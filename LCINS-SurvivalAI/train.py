"""
train.py
--------
Main entry point for training the survival analysis model.

Workflow:
  1. Load Sherlock-Lung clinical data
  2. Build/load tile map
  3. Patient-level train (n=454) / val (n=141) split
  4. [Optional] 10-fold CV within training set → derive risk threshold
  5. Retrain final model on complete training set (n=454)
  6. Save final checkpoint + training curves

Usage:
    # Full pipeline (CV + final training)
    python train.py

    # Skip CV, train directly
    python train.py --no_cv

    # Resume from a checkpoint
    python train.py --resume checkpoints/final_epoch_042.pth

    # Custom config overrides
    python train.py --lr 5e-4 --epochs 100 --batch_patients 8

Run `python train.py --help` for all options.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

# ── add project root to sys.path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config                      import Config
from data.splitter               import DataSplitter
from data.dataset                import TileTrainDataset
from data.sampler                import PatientAwareBatchSampler, get_train_transform, get_val_transform
from models.vgg16_survival       import VGG16Survival
from training.trainer            import Trainer
from training.cross_validation   import run_cross_validation
from utils.logger                import setup_logger


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train VGG-16 survival model on H&E WSI tiles (Cox loss)."
    )
    # ── Paths
    p.add_argument("--sherlock_csv",   default="data/sherlock_clinical.csv")
    p.add_argument("--msk_csv",        default="data/msk_clinical.csv")
    p.add_argument("--tiles_dir",      default="data/tiles")
    p.add_argument("--splits_dir",     default="data/splits")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--results_dir",    default="results")
    p.add_argument("--log_dir",        default="logs")
    p.add_argument("--figures_dir",    default="figures")

    # ── Tiling
    p.add_argument("--tile_size",   type=int,   default=256)
    p.add_argument("--max_tiles",   type=int,   default=2000)

    # ── Model
    p.add_argument("--dropout",     type=float, default=0.50)
    p.add_argument("--no_pretrain", action="store_true")

    # ── Optimiser
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=4e-4)

    # ── Training
    p.add_argument("--epochs",          type=int, default=150)
    p.add_argument("--cv_epochs",       type=int, default=100)
    p.add_argument("--batch_patients",  type=int, default=16,
                   help="Patients per mini-batch")
    p.add_argument("--tiles_per_patient", type=int, default=4,
                   help="Tiles sampled per patient per batch")
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--patience",        type=int, default=15)
    p.add_argument("--seed",            type=int, default=42)

    # ── Flags
    p.add_argument("--no_cv",     action="store_true",
                   help="Skip cross-validation and go straight to final training.")
    p.add_argument("--resume",    type=str, default=None,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--device",    type=str, default="cuda",
                   choices=["cuda", "mps", "cpu"])
    p.add_argument("--rebuild_tile_map", action="store_true",
                   help="Force rebuilding the tile-map cache.")
    p.add_argument("--force_split",      action="store_true",
                   help="Force re-creating the train/val split.")
    p.add_argument("--force_cv_folds",   action="store_true",
                   help="Force re-creating CV fold assignments.")
    p.add_argument("--aggregation", type=str, default="mean",
                   choices=["mean", "median", "max"])

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_config(args) -> Config:
    cfg = Config(
        sherlock_csv   = args.sherlock_csv,
        msk_csv        = args.msk_csv,
        tiles_dir      = args.tiles_dir,
        splits_dir     = args.splits_dir,
        tile_size      = args.tile_size,
        max_tiles_per_slide = args.max_tiles,
        dropout_rate   = args.dropout,
        pretrained     = not args.no_pretrain,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        n_epochs       = args.epochs,
        cv_epochs      = args.cv_epochs,
        n_patients_per_batch  = args.batch_patients,
        n_tiles_per_patient   = args.tiles_per_patient,
        num_workers    = args.num_workers,
        patience       = args.patience,
        random_seed    = args.seed,
        resume         = args.resume,
        device         = args.device,
        run_cv         = not args.no_cv,
        aggregation    = args.aggregation,
        checkpoint_dir = args.checkpoint_dir,
        results_dir    = args.results_dir,
        log_dir        = args.log_dir,
        figures_dir    = args.figures_dir,
    )
    return cfg


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = build_config(args)

    logger = setup_logger(log_dir=cfg.log_dir, name="train")
    logger.info("=" * 70)
    logger.info("  WSI Survival Analysis — Training Pipeline")
    logger.info("=" * 70)

    set_global_seed(cfg.random_seed)

    # ── 1. Data Splits ────────────────────────────────────────────────────────
    logger.info("\n[Step 1] Building dataset splits …")
    splitter = DataSplitter(
        sherlock_csv = cfg.sherlock_csv,
        msk_csv      = cfg.msk_csv,
        tiles_dir    = cfg.tiles_dir,
        splits_dir   = cfg.splits_dir,
        train_n      = cfg.train_n,
        n_folds      = cfg.n_folds,
        random_seed  = cfg.random_seed,
    )

    tile_map  = splitter.build_tile_map(force=args.rebuild_tile_map)
    train_df, val_df = splitter.get_train_val_split(force=args.force_split)

    logger.info(
        f"  Sherlock-Lung: train={len(train_df)}, val={len(val_df)}\n"
        f"  Total tiled patients in map: {len(tile_map)}"
    )

    # ── 2. Cross-Validation ───────────────────────────────────────────────────
    threshold = None
    if cfg.run_cv:
        logger.info("\n[Step 2] Running 10-fold cross-validation …")
        cv_folds = splitter.get_cv_folds(train_df, force=args.force_cv_folds)
        mean_ci, std_ci, threshold, oof_df = run_cross_validation(
            train_df=train_df,
            tile_map=tile_map,
            folds=cv_folds,
            cfg=cfg,
        )
        logger.info(
            f"  CV result: mean C-index = {mean_ci:.4f} ± {std_ci:.4f}\n"
            f"  OOF-derived threshold   = {threshold:.4f}"
        )
    else:
        logger.info("\n[Step 2] Skipping CV (--no_cv flag).")
        # Try to load threshold from a previous CV run
        cv_summary_path = os.path.join(cfg.results_dir, "cv_summary.json")
        if os.path.exists(cv_summary_path):
            with open(cv_summary_path) as f:
                cv_summary = json.load(f)
            threshold = cv_summary.get("threshold")
            logger.info(f"  Loaded threshold from previous CV: {threshold:.4f}")

    # ── 3. Final Model Training on Full Training Set ──────────────────────────
    logger.info("\n[Step 3] Training final model on complete training set (n=454) …")

    train_transform = get_train_transform(cfg.tile_size)

    full_train_dataset = TileTrainDataset(
        patient_df=train_df,
        tile_map=tile_map,
        transform=train_transform,
        max_tiles=cfg.max_tiles_per_slide,
    )

    sampler = PatientAwareBatchSampler(
        patient_index=full_train_dataset.patient_index,
        n_patients=cfg.n_patients_per_batch,
        n_tiles_per_patient=cfg.n_tiles_per_patient,
        shuffle=True,
        drop_last=True,
        seed=cfg.random_seed,
    )
    train_loader = torch.utils.data.DataLoader(
        full_train_dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    final_model = VGG16Survival(
        dropout_rate=cfg.dropout_rate,
        pretrained=cfg.pretrained,
    )

    trainer = Trainer(
        model=final_model,
        train_loader=train_loader,
        val_df=val_df,
        val_tile_map=tile_map,
        config=cfg,
        run_name="final",
    )

    results = trainer.fit()

    logger.info(
        f"\n  Final model training complete.\n"
        f"  Best epoch     : {results['best_epoch']}\n"
        f"  Best val loss  : {results['best_val_loss']:.4f}\n"
        f"  Best val C-idx : {results['best_val_ci']:.4f}"
    )

    # ── 4. Save threshold ─────────────────────────────────────────────────────
    if threshold is not None:
        threshold_path = os.path.join(cfg.results_dir, "risk_threshold.json")
        with open(threshold_path, "w") as f:
            json.dump({"threshold": threshold, "method": "optimal_logrank_oof"}, f, indent=2)
        logger.info(f"  Risk threshold saved → {threshold_path}")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  Training pipeline complete.")
    logger.info(f"  Best model     : {os.path.join(cfg.checkpoint_dir, 'final_best.pth')}")
    logger.info(f"  Loss curves    : {os.path.join(cfg.figures_dir,    'final_loss.png')}")
    logger.info(f"  C-index curve  : {os.path.join(cfg.figures_dir,    'final_cindex.png')}")
    logger.info(f"  Risk threshold : {threshold}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
