"""
training/early_stopping.py
--------------------------
Early stopping monitor with best-model checkpointing.

Stops training when the monitored metric (default: validation loss) does not
improve by at least `min_delta` for `patience` consecutive epochs.
The best model weights are automatically saved and can be restored.
"""

import logging
import os
import shutil
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping with best-model saving.

    Args:
        patience:       Number of epochs without improvement before stopping.
        min_delta:      Minimum change to count as an improvement.
        mode:           "min" (loss) or "max" (C-index, AUC).
        checkpoint_path: Where to save the best model weights.
        verbose:        Print improvement messages.
    """

    def __init__(
        self,
        patience:        int   = 15,
        min_delta:       float = 1e-4,
        mode:            str   = "min",
        checkpoint_path: str   = "checkpoints/best_model.pth",
        verbose:         bool  = True,
    ):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.patience        = patience
        self.min_delta       = min_delta
        self.mode            = mode
        self.checkpoint_path = checkpoint_path
        self.verbose         = verbose

        self._counter        = 0
        self._best_score:  Optional[float] = None
        self._best_epoch:  int             = 0
        self.stopped       = False
        self.best_model_restored = False

        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    # ── Core API ──────────────────────────────────────────────────────────────

    def step(self, score: float, model: torch.nn.Module, epoch: int) -> bool:
        """
        Evaluate the new score and decide whether to stop.

        Args:
            score:  The monitored metric value for this epoch.
            model:  The model to checkpoint if this is the best score.
            epoch:  Current epoch number (for logging).

        Returns:
            True if training should stop, False otherwise.
        """
        improved = self._is_improvement(score)

        if improved:
            self._best_score = score
            self._best_epoch = epoch
            self._counter    = 0
            self._save_checkpoint(model)
            if self.verbose:
                logger.info(
                    f"EarlyStopping: new best score {score:.5f} at epoch {epoch}. "
                    f"Checkpoint saved → {self.checkpoint_path}"
                )
        else:
            self._counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: no improvement for {self._counter}/{self.patience} "
                    f"epochs. Best={self._best_score:.5f} @ epoch {self._best_epoch}."
                )

        if self._counter >= self.patience:
            self.stopped = True
            if self.verbose:
                logger.info(
                    f"EarlyStopping: patience exhausted — stopping at epoch {epoch}. "
                    f"Best score was {self._best_score:.5f} at epoch {self._best_epoch}."
                )
            return True

        return False

    def restore_best(self, model: torch.nn.Module) -> None:
        """Load the best-checkpoint weights back into the model."""
        if not os.path.exists(self.checkpoint_path):
            logger.warning("No best-model checkpoint found — weights unchanged.")
            return
        state = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        self.best_model_restored = True
        logger.info(
            f"EarlyStopping: restored best model from {self.checkpoint_path} "
            f"(best score {self._best_score:.5f} @ epoch {self._best_epoch})."
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    def _is_improvement(self, score: float) -> bool:
        if self._best_score is None:
            return True
        if self.mode == "min":
            return score < self._best_score - self.min_delta
        else:
            return score > self._best_score + self.min_delta

    def _save_checkpoint(self, model: torch.nn.Module) -> None:
        tmp = self.checkpoint_path + ".tmp"
        torch.save(model.state_dict(), tmp)
        shutil.move(tmp, self.checkpoint_path)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def counter(self) -> int:
        return self._counter

    def state_dict(self) -> dict:
        return {
            "counter":     self._counter,
            "best_score":  self._best_score,
            "best_epoch":  self._best_epoch,
            "stopped":     self.stopped,
        }

    def load_state_dict(self, state: dict) -> None:
        self._counter    = state["counter"]
        self._best_score = state["best_score"]
        self._best_epoch = state["best_epoch"]
        self.stopped     = state["stopped"]
