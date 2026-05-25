"""
test.py
-------
Evaluation / inference entry point.

Runs the trained model on:
  1. Internal validation set  (Sherlock-Lung held-out, n = 141)
  2. External validation set  (MSK cohort — if available)

For each cohort:
  • Tile-level inference  →  slide-level risk-score aggregation (mean)
  • C-index computation
  • Log-rank test + Kaplan-Meier curve (high vs. low risk)
  • Risk-score distribution plot
  • Results saved as CSV

Usage:
    # Evaluate on internal + external validation
    python test.py --checkpoint checkpoints/final_best.pth

    # Evaluate on a specific cohort only
    python test.py --checkpoint checkpoints/final_best.pth --cohort internal
    python test.py --checkpoint checkpoints/final_best.pth --cohort msk

    # Use a custom risk threshold (instead of reading from file)
    python test.py --checkpoint checkpoints/final_best.pth --threshold 0.123
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config                  import Config
from data.splitter           import DataSplitter
from models.vgg16_survival   import VGG16Survival
from evaluation.inference    import evaluate_cohort
from evaluation.metrics      import compute_all_metrics
from utils.logger            import setup_logger
from utils.checkpoint        import load_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate trained survival model on validation cohorts."
    )
    p.add_argument("--checkpoint",    required=True,
                   help="Path to the trained model checkpoint (.pth).")
    p.add_argument("--sherlock_csv",  default="data/sherlock_clinical.csv")
    p.add_argument("--msk_csv",       default="data/msk_clinical.csv")
    p.add_argument("--tiles_dir",     default="data/tiles")
    p.add_argument("--splits_dir",    default="data/splits")
    p.add_argument("--results_dir",   default="results")
    p.add_argument("--figures_dir",   default="figures")
    p.add_argument("--log_dir",       default="logs")
    p.add_argument("--threshold",     type=float, default=None,
                   help="Risk dichotomisation threshold. "
                        "If not given, loaded from results/risk_threshold.json or median.")
    p.add_argument("--cohort",        type=str, default="all",
                   choices=["all", "internal", "msk"],
                   help="Which cohort(s) to evaluate.")
    p.add_argument("--aggregation",   type=str, default="mean",
                   choices=["mean", "median", "max"])
    p.add_argument("--tile_size",     type=int, default=256)
    p.add_argument("--batch_size",    type=int, default=512)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--dropout",       type=float, default=0.50)
    p.add_argument("--device",        type=str, default="cuda",
                   choices=["cuda", "mps", "cpu"])
    p.add_argument("--rebuild_tile_map", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def resolve_device(device_str: str) -> str:
    if device_str == "cuda" and not torch.cuda.is_available():
        logging.getLogger("test").warning("CUDA not available — falling back to CPU.")
        return "cpu"
    if device_str == "mps" and not torch.backends.mps.is_available():
        logging.getLogger("test").warning("MPS not available — falling back to CPU.")
        return "cpu"
    return device_str


def load_threshold(results_dir: str, cli_threshold) -> float:
    if cli_threshold is not None:
        return float(cli_threshold)
    threshold_path = os.path.join(results_dir, "risk_threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            data = json.load(f)
        thr = float(data["threshold"])
        logging.getLogger("test").info(
            f"Loaded risk threshold from {threshold_path}: {thr:.4f}"
        )
        return thr
    logging.getLogger("test").warning(
        "No threshold file found and --threshold not provided. "
        "Will use median of each cohort's risk scores."
    )
    return None


def print_results_table(all_metrics: dict, logger) -> None:
    """Print a formatted summary table of evaluation results."""
    header = f"{'Cohort':<25} {'C-index':>10} {'Log-rank p':>12} {'N high':>8} {'N low':>8}"
    line   = "-" * len(header)
    logger.info("\n" + line)
    logger.info(header)
    logger.info(line)
    for split, m in all_metrics.items():
        p_str = f"{m['pvalue']:.4g}" if not (m['pvalue'] != m['pvalue']) else "N/A"
        logger.info(
            f"{split:<25} {m['c_index']:>10.4f} {p_str:>12} "
            f"{m['n_high']:>8} {m['n_low']:>8}"
        )
    logger.info(line + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    logger = setup_logger(log_dir=args.log_dir, name="test")

    logger.info("=" * 70)
    logger.info("  WSI Survival Analysis — Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Checkpoint : {args.checkpoint}")
    logger.info(f"  Cohort     : {args.cohort}")
    logger.info(f"  Aggregation: {args.aggregation}")

    device    = resolve_device(args.device)
    threshold = load_threshold(args.results_dir, args.threshold)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("\n[Step 1] Loading model …")
    model = VGG16Survival(dropout_rate=args.dropout, pretrained=False)
    load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    model.eval()
    logger.info("  Model loaded and set to eval mode.")

    # ── Load splits & tile map ────────────────────────────────────────────────
    logger.info("\n[Step 2] Loading data splits and tile map …")

    # Minimal config for splitter
    cfg = Config(
        sherlock_csv = args.sherlock_csv,
        msk_csv      = args.msk_csv,
        tiles_dir    = args.tiles_dir,
        splits_dir   = args.splits_dir,
        results_dir  = args.results_dir,
        figures_dir  = args.figures_dir,
    )

    splitter = DataSplitter(
        sherlock_csv = args.sherlock_csv,
        msk_csv      = args.msk_csv,
        tiles_dir    = args.tiles_dir,
        splits_dir   = args.splits_dir,
    )
    tile_map         = splitter.build_tile_map(force=args.rebuild_tile_map)
    _, internal_val_df = splitter.get_train_val_split()
    msk_df           = splitter.get_msk_patients() if args.cohort in ("all", "msk") else None

    logger.info(
        f"  Internal val : {len(internal_val_df)} patients\n"
        f"  MSK external : {len(msk_df) if msk_df is not None else 0} patients"
    )

    all_metrics: dict = {}
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # ── Evaluate: Internal Validation ─────────────────────────────────────────
    if args.cohort in ("all", "internal"):
        logger.info("\n[Step 3a] Evaluating on internal validation set …")
        results_df, metrics = evaluate_cohort(
            model=model,
            patient_df=internal_val_df,
            tile_map=tile_map,
            threshold=threshold,
            aggregation=args.aggregation,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_name="Internal Validation",
            results_dir=args.results_dir,
            figures_dir=args.figures_dir,
            tile_size=args.tile_size,
        )
        all_metrics["Internal Validation"] = metrics

    # ── Evaluate: MSK External Validation ─────────────────────────────────────
    if args.cohort in ("all", "msk") and msk_df is not None and len(msk_df) > 0:
        logger.info("\n[Step 3b] Evaluating on MSK external validation cohort …")
        results_df_msk, metrics_msk = evaluate_cohort(
            model=model,
            patient_df=msk_df,
            tile_map=tile_map,
            threshold=threshold,
            aggregation=args.aggregation,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_name="MSK External",
            results_dir=args.results_dir,
            figures_dir=args.figures_dir,
            tile_size=args.tile_size,
        )
        all_metrics["MSK External"] = metrics_msk
    elif args.cohort in ("all", "msk") and msk_df is None:
        logger.info("[Step 3b] MSK CSV not found — skipping external validation.")

    # ── Summary Table ─────────────────────────────────────────────────────────
    if all_metrics:
        print_results_table(all_metrics, logger)
        # Save JSON summary
        summary_path = os.path.join(args.results_dir, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Evaluation summary saved → {summary_path}")

    logger.info("\n" + "=" * 70)
    logger.info("  Evaluation complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
