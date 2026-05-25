"""
training/cross_validation.py
-----------------------------
10-fold cross-validation within the training set (n ≈ 454 patients).

For each fold:
  • Train on ≈ 408 patients, validate on ≈ 46 patients.
  • Save per-fold best checkpoint.
  • Collect out-of-fold (OOF) predictions for all 454 training patients.

After CV:
  • Derive a risk threshold from OOF predictions.
  • Report per-fold and mean C-index.
  • Save OOF risk scores and CV summary.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config                  import Config
from data.dataset            import TileTrainDataset
from data.sampler            import PatientAwareBatchSampler, get_train_transform, get_val_transform
from models.vgg16_survival   import VGG16Survival
from training.trainer        import Trainer
from evaluation.metrics      import concordance_index, derive_risk_threshold, derive_optimal_threshold
from evaluation.inference    import predict_tiles, aggregate_tile_scores
from utils.visualization     import plot_cv_cindex

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Validation Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_cross_validation(
    train_df:    pd.DataFrame,
    tile_map:    Dict[str, List[str]],
    folds:       List[Dict],
    cfg:         Config,
) -> Tuple[float, float, float, pd.DataFrame]:
    """
    Execute 10-fold CV and collect out-of-fold risk predictions.

    Args:
        train_df:  Patient DataFrame for the full training set (n ≈ 454).
        tile_map:  Tile-path map (all cohort patients).
        folds:     List of fold dicts from DataSplitter.get_cv_folds().
        cfg:       Global Config.

    Returns:
        mean_ci:        Mean C-index across folds.
        std_ci:         Std of C-index across folds.
        threshold:      Risk threshold derived from OOF predictions.
        oof_df:         DataFrame: patient_id | oof_risk_score | survival_time | event.
    """
    fold_cindex: List[float] = []

    # OOF accumulator: patient_id → list of risk scores (one per fold this patient appeared in)
    oof_scores: Dict[str, List[float]] = {
        str(row["patient_id"]): [] for _, row in train_df.iterrows()
    }

    n_folds = len(folds)

    for fold_dict in folds:
        fold_i     = fold_dict["fold"]
        train_pids = set(fold_dict["train_pids"])
        val_pids   = set(fold_dict["val_pids"])

        logger.info(
            f"\n{'='*60}\n"
            f"  Cross-Validation Fold {fold_i + 1}/{n_folds}\n"
            f"  Train: {len(train_pids)} patients | Val: {len(val_pids)} patients\n"
            f"{'='*60}"
        )

        fold_train_df = train_df[train_df["patient_id"].isin(train_pids)].reset_index(drop=True)
        fold_val_df   = train_df[train_df["patient_id"].isin(val_pids)].reset_index(drop=True)

        # ── Build datasets ────────────────────────────────────────────────────
        train_transform = get_train_transform(cfg.tile_size)
        val_transform   = get_val_transform(cfg.tile_size)

        train_dataset = TileTrainDataset(
            fold_train_df, tile_map,
            transform=train_transform,
            max_tiles=cfg.max_tiles_per_slide,
        )

        sampler = PatientAwareBatchSampler(
            patient_index=train_dataset.patient_index,
            n_patients=cfg.n_patients_per_batch,
            n_tiles_per_patient=cfg.n_tiles_per_patient,
            shuffle=True,
            drop_last=True,
            seed=cfg.random_seed + fold_i,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        model = VGG16Survival(
            dropout_rate=cfg.dropout_rate,
            pretrained=cfg.pretrained,
        )

        # Override epochs for CV (lighter)
        fold_cfg          = _copy_config(cfg)
        fold_cfg.n_epochs = cfg.cv_epochs
        fold_cfg.resume   = None          # never auto-resume inside CV folds

        # ── Train ─────────────────────────────────────────────────────────────
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_df=fold_val_df,
            val_tile_map=tile_map,
            config=fold_cfg,
            run_name=f"cv_fold_{fold_i:02d}",
        )
        results = trainer.fit()

        logger.info(
            f"Fold {fold_i} complete | "
            f"best val loss: {results['best_val_loss']:.4f} | "
            f"best val C-index: {results['best_val_ci']:.4f}"
        )
        fold_cindex.append(results["best_val_ci"])

        # ── Collect OOF predictions for this fold's validation patients ────────
        logger.info(f"Collecting OOF predictions for fold {fold_i} …")
        tile_scores  = predict_tiles(
            model=model,
            patient_df=fold_val_df,
            tile_map=tile_map,
            device=cfg.device,
            batch_size=cfg.batch_size * 4,
            num_workers=cfg.num_workers,
        )
        slide_scores = aggregate_tile_scores(tile_scores, method=cfg.aggregation)
        for pid, sc in slide_scores.items():
            if pid in oof_scores:
                oof_scores[pid].append(sc)

    # ── Aggregate OOF ─────────────────────────────────────────────────────────
    oof_records = []
    for _, row in train_df.iterrows():
        pid    = str(row["patient_id"])
        scores = oof_scores.get(pid, [])
        if not scores:
            continue
        oof_records.append({
            "patient_id":    pid,
            "oof_risk_score": float(np.mean(scores)),
            "survival_time": float(row["survival_time"]),
            "event":         int(row["event"]),
        })
    oof_df = pd.DataFrame(oof_records)

    # ── Overall OOF C-index ───────────────────────────────────────────────────
    oof_ci = concordance_index(
        oof_df["survival_time"].values,
        oof_df["event"].values,
        oof_df["oof_risk_score"].values,
    )
    mean_ci = float(np.mean(fold_cindex))
    std_ci  = float(np.std(fold_cindex))

    logger.info(
        f"\n{'='*60}\n"
        f"  CV Complete ({n_folds} folds)\n"
        f"  Per-fold C-indices: {[f'{c:.3f}' for c in fold_cindex]}\n"
        f"  Mean C-index: {mean_ci:.4f} ± {std_ci:.4f}\n"
        f"  OOF C-index : {oof_ci:.4f}\n"
        f"{'='*60}"
    )

    # ── Risk threshold from OOF ───────────────────────────────────────────────
    threshold = derive_optimal_threshold(
        risk_scores=oof_df["oof_risk_score"].values,
        survival_times=oof_df["survival_time"].values,
        events=oof_df["event"].values,
    )
    logger.info(f"OOF-derived risk threshold: {threshold:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(cfg.results_dir, exist_ok=True)

    oof_df.to_csv(os.path.join(cfg.results_dir, "oof_risk_scores.csv"), index=False)

    cv_summary = {
        "fold_cindex": fold_cindex,
        "mean_cindex": mean_ci,
        "std_cindex":  std_ci,
        "oof_cindex":  oof_ci,
        "threshold":   threshold,
    }
    with open(os.path.join(cfg.results_dir, "cv_summary.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs(cfg.figures_dir, exist_ok=True)
    plot_cv_cindex(
        fold_cindex,
        save_path=os.path.join(cfg.figures_dir, "cv_cindex.png"),
        title=f"C-index per CV Fold (mean = {mean_ci:.3f} ± {std_ci:.3f})",
    )

    return mean_ci, std_ci, threshold, oof_df


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _copy_config(cfg: Config) -> Config:
    """Shallow copy of config (dataclass)."""
    import copy
    return copy.copy(cfg)
