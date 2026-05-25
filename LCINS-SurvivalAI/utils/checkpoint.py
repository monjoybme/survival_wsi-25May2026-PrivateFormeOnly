"""
utils/checkpoint.py
-------------------
Utilities for saving and loading full training state:
  • model weights
  • optimiser state
  • scheduler state
  • early stopping state
  • epoch, loss history
"""

import logging
import os
import shutil
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path:           str,
    epoch:          int,
    model:          torch.nn.Module,
    optimizer:      torch.optim.Optimizer,
    scheduler:      Optional[Any]          = None,
    early_stopping: Optional[Any]          = None,
    metrics:        Optional[Dict]         = None,
    extra:          Optional[Dict]         = None,
) -> None:
    """
    Save a full training checkpoint to `path`.

    All state required to resume training is included so the run can be
    restarted from any epoch without loss of optimiser momentum etc.

    Args:
        path:            File path for the checkpoint (.pth).
        epoch:           Current epoch number (1-indexed).
        model:           The PyTorch model.
        optimizer:       The optimiser.
        scheduler:       LR scheduler (optional).
        early_stopping:  EarlyStopping instance (optional).
        metrics:         Dict of tracked metrics (e.g. train/val loss history).
        extra:           Any additional key-value pairs to store.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"

    payload: Dict[str, Any] = {
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()

    if early_stopping is not None:
        payload["early_stopping_state"] = early_stopping.state_dict()

    if metrics is not None:
        payload["metrics"] = metrics

    if extra is not None:
        payload.update(extra)

    torch.save(payload, tmp_path)
    shutil.move(tmp_path, path)
    logger.debug(f"Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path:           str,
    model:          torch.nn.Module,
    optimizer:      Optional[torch.optim.Optimizer] = None,
    scheduler:      Optional[Any]                    = None,
    early_stopping: Optional[Any]                    = None,
    device:         str                              = "cpu",
    strict:         bool                             = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore all state into the provided objects.

    Args:
        path:            Checkpoint file path.
        model:           Model to restore weights into.
        optimizer:       Optimiser to restore state into (optional).
        scheduler:       LR scheduler to restore state into (optional).
        early_stopping:  EarlyStopping instance to restore state into (optional).
        device:          Device to map tensors to.
        strict:          Strict model state_dict loading.

    Returns:
        The full checkpoint dict (contains epoch, metrics, etc.).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Loading checkpoint from {path} …")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state"], strict=strict)
    logger.info(f"  Model weights restored (epoch {ckpt.get('epoch', '?')}).")

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        logger.info("  Optimiser state restored.")

    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
        logger.info("  LR scheduler state restored.")

    if early_stopping is not None and "early_stopping_state" in ckpt:
        early_stopping.load_state_dict(ckpt["early_stopping_state"])
        logger.info("  EarlyStopping state restored.")

    return ckpt


def get_latest_checkpoint(checkpoint_dir: str, prefix: str = "epoch_") -> Optional[str]:
    """
    Scan `checkpoint_dir` for checkpoint files matching `prefix*.pth` and
    return the path to the one with the highest epoch number, or None if none exist.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    candidates = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith(prefix) and f.endswith(".pth")
    ]
    if not candidates:
        return None

    def _epoch_num(fname: str) -> int:
        try:
            return int(fname.replace(prefix, "").replace(".pth", ""))
        except ValueError:
            return -1

    latest = max(candidates, key=_epoch_num)
    return os.path.join(checkpoint_dir, latest)
