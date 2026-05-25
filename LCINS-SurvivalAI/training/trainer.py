"""
training/trainer.py
-------------------
Core training / validation loop for the survival analysis pipeline.

Features:
  • Adam optimiser with the exact hyperparameters from the paper
  • Mini-batch Cox partial-likelihood loss
  • Per-epoch C-index tracking on the validation set
  • Checkpointing every epoch (full state) + best-model saving
  • Resume from any saved checkpoint
  • Early stopping
  • Loss and C-index curve saving
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.cox_loss        import CoxLoss
from training.early_stopping import EarlyStopping
from evaluation.metrics     import concordance_index
from evaluation.inference   import predict_tiles, aggregate_tile_scores
from utils.checkpoint       import save_checkpoint, load_checkpoint, get_latest_checkpoint
from utils.visualization    import plot_loss_curves, plot_metric_curve

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Manages the complete training lifecycle for one model run.

    Args:
        model:          VGG16Survival instance.
        train_loader:   DataLoader yielding tiles with patient labels.
        val_df:         Patient-level DataFrame for validation C-index.
        val_tile_map:   Tile-path map for validation cohort.
        config:         Config dataclass.
        run_name:       Unique identifier for this run (used in file names).
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_df:       pd.DataFrame,
        val_tile_map: Dict[str, List[str]],
        config,
        run_name:     str = "final",
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_df       = val_df
        self.val_tile_map = val_tile_map
        self.cfg          = config
        self.run_name     = run_name

        # Device
        self.device = self._resolve_device(config.device)
        self.model   = self.model.to(self.device)

        # Criterion
        self.criterion = CoxLoss(reduction="mean")

        # Optimiser (Adam with paper-specified hyperparameters)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

        # LR scheduler (ReduceLROnPlateau on val loss)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=7, verbose=True
        )

        # Early stopping
        best_ckpt = os.path.join(config.checkpoint_dir, f"{run_name}_best.pth")
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode="min",
            checkpoint_path=best_ckpt,
            verbose=True,
        )

        # Metric history
        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []
        self.val_cindex:   List[float] = []
        self.start_epoch   = 1

        # AMP scaler (CUDA only)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.device.type == "cuda")
        )

        # Resume if requested
        if config.resume:
            self._resume(config.resume)
        else:
            # Auto-resume from latest checkpoint if one exists
            latest = get_latest_checkpoint(
                config.checkpoint_dir, prefix=f"{run_name}_epoch_"
            )
            if latest:
                logger.info(f"Auto-resuming from latest checkpoint: {latest}")
                self._resume(latest)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self) -> Dict:
        """
        Run the complete training loop.

        Returns:
            Dict with final metrics and history.
        """
        n_epochs = self.cfg.n_epochs
        logger.info(
            f"Starting training [{self.run_name}]: "
            f"epochs {self.start_epoch}–{n_epochs}, "
            f"device={self.device}, "
            f"lr={self.cfg.lr}, "
            f"weight_decay={self.cfg.weight_decay}"
        )

        for epoch in range(self.start_epoch, n_epochs + 1):
            t0 = time.time()

            # Update sampler epoch (for patient-aware shuffling)
            if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                self.train_loader.batch_sampler.set_epoch(epoch)

            # ── Train one epoch ───────────────────────────────────────────────
            train_loss = self._train_epoch(epoch)
            self.train_losses.append(train_loss)

            # ── Validate ──────────────────────────────────────────────────────
            val_loss, val_ci = self._validate_epoch()
            self.val_losses.append(val_loss)
            self.val_cindex.append(val_ci)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val C-index: {val_ci:.4f} | "
                f"LR: {self._current_lr():.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # LR scheduler step
            self.scheduler.step(val_loss)

            # Epoch checkpoint
            epoch_ckpt = os.path.join(
                self.cfg.checkpoint_dir, f"{self.run_name}_epoch_{epoch:03d}.pth"
            )
            save_checkpoint(
                path=epoch_ckpt,
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                early_stopping=self.early_stopping,
                metrics={
                    "train_losses": self.train_losses,
                    "val_losses":   self.val_losses,
                    "val_cindex":   self.val_cindex,
                },
            )
            self._cleanup_old_checkpoints(keep_last=3)

            # Early stopping check (monitor val loss)
            if self.early_stopping.step(val_loss, self.model, epoch):
                logger.info("Early stopping triggered.")
                break

        # Restore best weights
        self.early_stopping.restore_best(self.model)

        # Save final curves
        self._save_curves()

        return {
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
            "val_cindex":   self.val_cindex,
            "best_val_loss": self.early_stopping.best_score,
            "best_epoch":    self.early_stopping.best_epoch,
            "best_val_ci":   max(self.val_cindex) if self.val_cindex else float("nan"),
        }

    # ── Training Step ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            imgs   = batch["image"].to(self.device, non_blocking=True)
            times  = batch["surv_time"].to(self.device, non_blocking=True)
            events = batch["event"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                risk_scores = self.model(imgs)
                loss        = self.criterion(risk_scores, times, events)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Epoch {epoch}: NaN/Inf loss — skipping batch.")
                continue

            self.scaler.scale(loss).backward()
            # Gradient clipping for stability
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # ── Validation Step ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate_epoch(self) -> Tuple[float, float]:
        """
        Compute validation loss and C-index.

        Val loss is approximated by running tiles through the model in
        eval mode and computing Cox loss over the aggregated (patient-mean) scores.
        """
        self.model.eval()

        # Get per-patient mean risk scores using tile inference
        tile_scores = predict_tiles(
            self.model, self.val_df, self.val_tile_map,
            device=str(self.device),
            batch_size=self.cfg.batch_size * 4,
            num_workers=self.cfg.num_workers,
        )
        slide_scores = aggregate_tile_scores(tile_scores, method=self.cfg.aggregation)

        risk_arr, time_arr, event_arr = [], [], []
        for _, row in self.val_df.iterrows():
            pid = str(row["patient_id"])
            if pid not in slide_scores:
                continue
            risk_arr.append(slide_scores[pid])
            time_arr.append(float(row["survival_time"]))
            event_arr.append(int(row["event"]))

        if not risk_arr:
            return 0.0, 0.5

        risk_t  = torch.tensor(risk_arr,  dtype=torch.float32)
        time_t  = torch.tensor(time_arr,  dtype=torch.float32)
        event_t = torch.tensor(event_arr, dtype=torch.float32)

        val_loss = self.criterion(risk_t, time_t, event_t).item()
        val_ci   = concordance_index(
            np.array(time_arr), np.array(event_arr), np.array(risk_arr)
        )
        return val_loss, val_ci

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resume(self, checkpoint_path: str) -> None:
        ckpt = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.early_stopping,
            device=str(self.device),
        )
        self.start_epoch = ckpt.get("epoch", 0) + 1
        if "metrics" in ckpt:
            m = ckpt["metrics"]
            self.train_losses = m.get("train_losses", [])
            self.val_losses   = m.get("val_losses",   [])
            self.val_cindex   = m.get("val_cindex",   [])
        logger.info(f"Resumed from epoch {self.start_epoch - 1}.")

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _save_curves(self) -> None:
        fig_dir = self.cfg.figures_dir
        os.makedirs(fig_dir, exist_ok=True)

        if self.train_losses and self.val_losses:
            plot_loss_curves(
                self.train_losses, self.val_losses,
                save_path=os.path.join(fig_dir, f"{self.run_name}_loss.png"),
                title=f"Loss — {self.run_name}",
            )
        if self.val_cindex:
            plot_metric_curve(
                self.val_cindex,
                save_path=os.path.join(fig_dir, f"{self.run_name}_cindex.png"),
                label="Val C-index",
                title=f"Validation C-index — {self.run_name}",
            )

    def _cleanup_old_checkpoints(self, keep_last: int = 3) -> None:
        """Remove epoch checkpoints older than the last `keep_last`."""
        ckpt_dir = self.cfg.checkpoint_dir
        prefix   = f"{self.run_name}_epoch_"
        files    = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith(prefix)],
            key=lambda x: int(x.replace(prefix, "").replace(".pth", ""))
        )
        for old in files[:-keep_last]:
            try:
                os.remove(os.path.join(ckpt_dir, old))
            except OSError:
                pass

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if device_str == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
