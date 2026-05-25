"""
data/sampler.py
---------------
Custom batch sampler that guarantees each mini-batch contains tiles from
multiple *different* patients, which is essential for the Cox partial-likelihood
loss to receive meaningful gradient signal (requires diverse survival times).

Also defines the albumentations transform pipelines used for training
and inference.
"""

import logging
import math
from typing import Dict, List, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Sampler

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_OK = True
except ImportError:
    ALBUMENTATIONS_OK = False
    logging.warning("albumentations not installed — using basic transforms.")

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Patient-Aware Batch Sampler
# ──────────────────────────────────────────────────────────────────────────────

class PatientAwareBatchSampler(Sampler):
    """
    Produces batches of tile indices such that each batch contains exactly
    `n_patients` patients and `n_tiles_per_patient` tiles per patient.

    Batch size = n_patients × n_tiles_per_patient.

    Args:
        patient_index:       Dict[patient_id → list of dataset indices].
        n_patients:          Number of patients per batch.
        n_tiles_per_patient: Number of tiles sampled per patient per batch.
        shuffle:             Whether to shuffle patient and tile order.
        drop_last:           Drop the last (incomplete) batch.
        seed:                RNG seed.
    """

    def __init__(
        self,
        patient_index: Dict[str, List[int]],
        n_patients:    int = 16,
        n_tiles_per_patient: int = 4,
        shuffle: bool  = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.patient_index   = patient_index
        self.n_patients      = n_patients
        self.n_tiles         = n_tiles_per_patient
        self.shuffle         = shuffle
        self.drop_last       = drop_last
        self.seed            = seed
        self.epoch           = 0

        # Only include patients with enough tiles
        self.valid_patients  = [
            p for p, idxs in patient_index.items() if len(idxs) >= 1
        ]
        self.n_valid         = len(self.valid_patients)

        n_batches_float = self.n_valid / n_patients
        self.n_batches = (
            math.floor(n_batches_float)
            if drop_last else math.ceil(n_batches_float)
        )

        logger.info(
            f"PatientAwareBatchSampler: {self.n_valid} patients, "
            f"batch = {n_patients} patients × {n_tiles_per_patient} tiles "
            f"= {n_patients * n_tiles_per_patient} tiles. "
            f"{self.n_batches} batches/epoch."
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)

        patients = list(self.valid_patients)
        if self.shuffle:
            rng.shuffle(patients)

        for batch_start in range(0, len(patients), self.n_patients):
            batch_patients = patients[batch_start: batch_start + self.n_patients]
            if self.drop_last and len(batch_patients) < self.n_patients:
                break

            batch_indices: List[int] = []
            for pid in batch_patients:
                pool  = self.patient_index[pid]
                k     = min(self.n_tiles, len(pool))
                chosen = rng.choice(pool, size=k, replace=False).tolist()
                batch_indices.extend(chosen)

            yield batch_indices


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation Pipelines
# ──────────────────────────────────────────────────────────────────────────────

# ImageNet normalisation constants
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transform(tile_size: int = 256) -> "A.Compose":
    """
    Training augmentation pipeline:
      • Random horizontal / vertical flips
      • Random 90° rotations
      • Colour jitter (brightness, contrast, saturation, hue)
      • ImageNet normalisation + ToTensorV2
    """
    if not ALBUMENTATIONS_OK:
        return _basic_train_transform(tile_size)

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.05,
            p=0.80,
        ),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.20),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform(tile_size: int = 256) -> "A.Compose":
    """
    Validation / inference transform: normalisation only, no augmentation.
    """
    if not ALBUMENTATIONS_OK:
        return _basic_val_transform(tile_size)

    return A.Compose([
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


# ── Fallback pure-PyTorch transforms (if albumentations is unavailable) ───────

import torch
import torchvision.transforms as T


def _basic_train_transform(tile_size: int):
    """Fallback training transform using torchvision."""
    class _Wrap:
        def __init__(self):
            self.t = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(90),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ])
        def __call__(self, image):
            return {"image": self.t(image)}
    return _Wrap()


def _basic_val_transform(tile_size: int):
    """Fallback inference transform using torchvision."""
    class _Wrap:
        def __init__(self):
            self.t = T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ])
        def __call__(self, image):
            return {"image": self.t(image)}
    return _Wrap()
