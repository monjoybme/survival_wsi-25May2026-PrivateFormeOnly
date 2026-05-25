"""
config.py
---------
Central configuration for the WSI survival analysis pipeline.
All hyperparameters, paths, and settings are defined here.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Config:
    # ─── Data Paths ───────────────────────────────────────────────────────────
    # CSV with columns: patient_id, wsi_path, survival_time, event, cohort
    sherlock_csv: str = "data/sherlock_clinical.csv"
    msk_csv:      str = "data/msk_clinical.csv"

    # Where extracted tiles are stored: tiles_dir/{patient_id}/{x}_{y}.png
    tiles_dir:    str = "data/tiles"

    # Where train/val/fold split JSONs are saved
    splits_dir:   str = "data/splits"

    # ─── Tiling ───────────────────────────────────────────────────────────────
    tile_size:              int   = 256        # pixels
    target_magnification:   float = 20.0       # 20×
    tile_overlap:           int   = 0          # non-overlapping
    tissue_threshold:       float = 0.5        # min tissue fraction per tile
    max_tiles_per_slide:    int   = 2000       # cap to prevent memory issues

    # ─── Dataset Split ────────────────────────────────────────────────────────
    train_n:     int   = 454
    val_n:       int   = 141
    random_seed: int   = 42
    n_folds:     int   = 10                    # CV folds within training set

    # ─── Model ────────────────────────────────────────────────────────────────
    dropout_rate: float = 0.50
    pretrained:   bool  = True                 # ImageNet init for VGG-16

    # ─── Optimiser ────────────────────────────────────────────────────────────
    lr:           float = 1e-3
    beta1:        float = 0.9
    beta2:        float = 0.999
    eps:          float = 1e-8
    weight_decay: float = 4e-4                 # L2 regularisation

    # ─── Training ─────────────────────────────────────────────────────────────
    n_epochs:              int   = 150
    batch_size:            int   = 64          # tiles per mini-batch
    n_patients_per_batch:  int   = 16          # patients sampled per batch
    n_tiles_per_patient:   int   = 4           # tiles sampled per patient per batch
    num_workers:           int   = 4
    pin_memory:            bool  = True

    # ─── Early Stopping ───────────────────────────────────────────────────────
    patience:       int   = 15                 # epochs without improvement
    min_delta:      float = 1e-4               # minimum improvement to count

    # ─── Aggregation at Inference ─────────────────────────────────────────────
    # "mean" | "median" | "max"
    aggregation: str = "mean"

    # ─── Output Directories ───────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    results_dir:    str = "results"
    log_dir:        str = "logs"
    figures_dir:    str = "figures"

    # ─── Resume ───────────────────────────────────────────────────────────────
    resume: Optional[str] = None               # path to a .pth checkpoint

    # ─── Device ───────────────────────────────────────────────────────────────
    device: str = "cuda"                       # "cuda" | "mps" | "cpu"

    # ─── Cross-Validation ─────────────────────────────────────────────────────
    run_cv:       bool = True
    cv_epochs:    int  = 100                   # epochs per CV fold (lighter)

    def __post_init__(self):
        for d in [self.tiles_dir, self.splits_dir, self.checkpoint_dir,
                  self.results_dir, self.log_dir, self.figures_dir]:
            os.makedirs(d, exist_ok=True)

    @property
    def effective_batch_size(self) -> int:
        return self.n_patients_per_batch * self.n_tiles_per_patient
