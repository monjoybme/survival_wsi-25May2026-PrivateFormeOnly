"""
data/dataset.py
---------------
PyTorch Dataset classes for the survival analysis pipeline.

  TileTrainDataset  – Training dataset: one tile per sample, paired with
                      the patient's survival label.

  SlideInferDataset – Inference dataset: all tiles from a single slide,
                      used to generate slide-level risk scores.
"""

import os
import logging
from typing import Dict, List, Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Training Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TileTrainDataset(Dataset):
    """
    Flat dataset where each item is (tile_image, survival_time, event, patient_id).

    Args:
        patient_df:   DataFrame with columns [patient_id, survival_time, event].
        tile_map:     Dict mapping patient_id → list of tile file paths.
        transform:    Albumentations or torchvision transform applied to each tile.
        max_tiles:    Cap on tiles per patient (None = no cap).
        cache_in_ram: If True, cache decoded PIL images in memory (fast but RAM-heavy).
    """

    def __init__(
        self,
        patient_df: pd.DataFrame,
        tile_map:   Dict[str, List[str]],
        transform:  Optional[Callable] = None,
        max_tiles:  Optional[int] = None,
        cache_in_ram: bool = False,
    ):
        self.transform    = transform
        self.cache_in_ram = cache_in_ram
        self._cache: Dict[str, np.ndarray] = {}

        # Build flat sample list [(tile_path, survival_time, event, patient_id)]
        self.samples: List[Tuple[str, float, int, str]] = []
        self.patient_index: Dict[str, List[int]] = {}  # patient_id → sample indices

        for _, row in patient_df.iterrows():
            pid = str(row["patient_id"])
            t   = float(row["survival_time"])
            e   = int(row["event"])

            tiles = tile_map.get(pid, [])
            if not tiles:
                logger.debug(f"Patient {pid} has no tiles — skipping.")
                continue

            if max_tiles and len(tiles) > max_tiles:
                rng   = np.random.default_rng(seed=abs(hash(pid)) % (2**31))
                tiles = rng.choice(tiles, size=max_tiles, replace=False).tolist()

            start_idx = len(self.samples)
            for tp in tiles:
                self.samples.append((tp, t, e, pid))
            self.patient_index[pid] = list(range(start_idx, len(self.samples)))

        self.n_patients = len(self.patient_index)
        logger.info(
            f"TileTrainDataset: {len(self.samples)} tiles "
            f"from {self.n_patients} patients."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tile_path, surv_time, event, pid = self.samples[idx]

        img = self._load_image(tile_path)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]          # albumentations → numpy HWC uint8
        else:
            img = torch.from_numpy(
                img.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        return {
            "image":      img,
            "surv_time":  torch.tensor(surv_time, dtype=torch.float32),
            "event":      torch.tensor(event,     dtype=torch.float32),
            "patient_id": pid,
        }

    def _load_image(self, path: str) -> np.ndarray:
        if self.cache_in_ram and path in self._cache:
            return self._cache[path].copy()
        img = np.array(Image.open(path).convert("RGB"))
        if self.cache_in_ram:
            self._cache[path] = img.copy()
        return img

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_patient_labels(self) -> pd.DataFrame:
        """Return a DataFrame with one row per patient (for C-index computation)."""
        seen = {}
        for tp, t, e, pid in self.samples:
            if pid not in seen:
                seen[pid] = {"patient_id": pid, "survival_time": t, "event": e}
        return pd.DataFrame(list(seen.values()))


# ──────────────────────────────────────────────────────────────────────────────
# Inference Dataset (single slide)
# ──────────────────────────────────────────────────────────────────────────────

class SlideInferDataset(Dataset):
    """
    Dataset over all tiles of a single slide for inference.

    Args:
        tile_paths:  List of tile file paths for one patient/slide.
        transform:   Inference transform (normalisation only, no augmentation).
    """

    def __init__(
        self,
        tile_paths: List[str],
        transform:  Optional[Callable] = None,
    ):
        self.tile_paths = tile_paths
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.tile_paths[idx]
        img  = np.array(Image.open(path).convert("RGB"))

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            img = torch.from_numpy(
                img.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        return {"image": img, "path": path}


# ──────────────────────────────────────────────────────────────────────────────
# Cohort Inference Dataset (all slides)
# ──────────────────────────────────────────────────────────────────────────────

class CohortInferDataset(Dataset):
    """
    Flat dataset for batch inference across all slides in a cohort.
    Returns one tile per sample along with its patient_id for aggregation.
    """

    def __init__(
        self,
        patient_df: pd.DataFrame,
        tile_map:   Dict[str, List[str]],
        transform:  Optional[Callable] = None,
    ):
        self.transform = transform
        self.samples: List[Tuple[str, str, float, int]] = []
        # (tile_path, patient_id, surv_time, event)

        for _, row in patient_df.iterrows():
            pid   = str(row["patient_id"])
            t     = float(row["survival_time"])
            e     = int(row["event"])
            tiles = tile_map.get(pid, [])
            for tp in tiles:
                self.samples.append((tp, pid, t, e))

        logger.info(
            f"CohortInferDataset: {len(self.samples)} tiles "
            f"from {patient_df['patient_id'].nunique()} patients."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        path, pid, t, e = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            img = torch.from_numpy(
                img.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        return {
            "image":      img,
            "patient_id": pid,
            "surv_time":  torch.tensor(t, dtype=torch.float32),
            "event":      torch.tensor(e, dtype=torch.float32),
        }
