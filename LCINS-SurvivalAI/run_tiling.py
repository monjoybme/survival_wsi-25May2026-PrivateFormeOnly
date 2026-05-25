"""
run_tiling.py
-------------
Standalone script to tile all WSIs before training.
Must be run BEFORE train.py.

Usage:
    python run_tiling.py --sherlock_csv data/sherlock_clinical.csv \
                         --msk_csv      data/msk_clinical.csv \
                         --out_dir      data/tiles

    # Tile only Sherlock-Lung (skip MSK for now)
    python run_tiling.py --sherlock_csv data/sherlock_clinical.csv

    # Force re-tiling (ignore existing tiles)
    python run_tiling.py --sherlock_csv data/sherlock_clinical.csv --no_resume
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess.tiling import tile_cohort
from utils.logger      import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description="Tile WSIs for survival analysis pipeline")
    p.add_argument("--sherlock_csv",   required=True,
                   help="Path to Sherlock-Lung clinical CSV (must have wsi_path column).")
    p.add_argument("--msk_csv",        default=None,
                   help="Path to MSK clinical CSV (optional).")
    p.add_argument("--out_dir",        default="data/tiles",
                   help="Output directory for tile PNGs.")
    p.add_argument("--tile_size",      type=int,   default=256)
    p.add_argument("--magnification",  type=float, default=20.0)
    p.add_argument("--tissue_threshold", type=float, default=0.5)
    p.add_argument("--max_tiles",      type=int,   default=2000)
    p.add_argument("--no_resume",      action="store_true",
                   help="Re-tile all slides even if already tiled.")
    p.add_argument("--log_dir",        default="logs")
    return p.parse_args()


def main():
    args   = parse_args()
    logger = setup_logger(log_dir=args.log_dir, name="tiling")

    logger.info("=" * 60)
    logger.info("  WSI Tiling Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Output dir   : {args.out_dir}")
    logger.info(f"  Tile size    : {args.tile_size} px @ {args.magnification}×")
    logger.info(f"  Max tiles    : {args.max_tiles}")
    logger.info(f"  Resume       : {not args.no_resume}")

    # ── Tile Sherlock-Lung ─────────────────────────────────────────────────────
    logger.info("\n[1/2] Tiling Sherlock-Lung cohort …")
    sherlock_manifest = tile_cohort(
        csv_path          = args.sherlock_csv,
        out_dir           = args.out_dir,
        tile_size         = args.tile_size,
        target_mag        = args.magnification,
        tissue_threshold  = args.tissue_threshold,
        max_tiles         = args.max_tiles,
        resume            = not args.no_resume,
    )
    n_patients  = len(sherlock_manifest)
    n_tiles     = sum(len(v) for v in sherlock_manifest.values())
    logger.info(
        f"  Sherlock-Lung: {n_patients} patients, {n_tiles:,} tiles extracted."
    )

    # ── Tile MSK (if provided) ─────────────────────────────────────────────────
    if args.msk_csv and os.path.exists(args.msk_csv):
        logger.info("\n[2/2] Tiling MSK cohort …")
        msk_manifest = tile_cohort(
            csv_path         = args.msk_csv,
            out_dir          = args.out_dir,
            tile_size        = args.tile_size,
            target_mag       = args.magnification,
            tissue_threshold = args.tissue_threshold,
            max_tiles        = args.max_tiles,
            resume           = not args.no_resume,
        )
        n_msk_p = len(msk_manifest)
        n_msk_t = sum(len(v) for v in msk_manifest.values())
        logger.info(f"  MSK: {n_msk_p} patients, {n_msk_t:,} tiles extracted.")
    else:
        logger.info("[2/2] No MSK CSV provided — skipping MSK tiling.")

    logger.info("\n" + "=" * 60)
    logger.info("  Tiling complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
