"""
preprocess/tiling.py
--------------------
Tiles Whole Slide Images (WSIs) into non-overlapping 256×256-pixel patches
at 20× magnification.  Only tiles with sufficient tissue coverage are kept.

Usage (standalone):
    python -m preprocess.tiling \
        --csv data/sherlock_clinical.csv \
        --out_dir data/tiles \
        --tile_size 256 \
        --magnification 20 \
        --tissue_threshold 0.5 \
        --max_tiles 2000
"""

import os
import argparse
import logging
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    logging.warning("openslide-python not installed. WSI reading unavailable.")


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Tissue Detection
# ──────────────────────────────────────────────────────────────────────────────

def get_tissue_mask(thumbnail: np.ndarray, threshold: int = 200) -> np.ndarray:
    """
    Generate binary tissue mask from an RGB thumbnail using Otsu thresholding
    in the grayscale channel.

    Returns a boolean mask (True = tissue).
    """
    gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
    # Invert so tissue is bright
    gray_inv = 255 - gray
    _, mask = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask.astype(bool)


def tile_has_tissue(tile: np.ndarray, tissue_threshold: float = 0.5) -> bool:
    """
    Check whether a tile contains enough tissue.
    Uses saturation channel in HSV to detect coloured (non-white) pixels.
    """
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    tissue_fraction = np.mean(sat > 20)
    return tissue_fraction >= tissue_threshold


# ──────────────────────────────────────────────────────────────────────────────
# Magnification Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_best_level(slide: "openslide.OpenSlide",
                   target_mag: float) -> Tuple[int, float]:
    """
    Return the best OpenSlide level for the requested magnification and
    the actual downsample factor to use within that level.
    """
    native_mag_str = slide.properties.get(
        openslide.PROPERTY_NAME_OBJECTIVE_POWER,
        slide.properties.get("openslide.objective-power", None)
    )
    if native_mag_str is None:
        # Fallback: assume 40× native
        native_mag = 40.0
        logger.warning("Could not read native magnification; assuming 40×.")
    else:
        native_mag = float(native_mag_str)

    desired_downsample = native_mag / target_mag
    level = slide.get_best_level_for_downsample(desired_downsample)
    actual_downsample = slide.level_downsamples[level]
    # Additional resize factor if the level's downsample != desired
    resize_factor = actual_downsample / desired_downsample
    return level, resize_factor, native_mag


# ──────────────────────────────────────────────────────────────────────────────
# Core Tiling Function
# ──────────────────────────────────────────────────────────────────────────────

def tile_wsi(wsi_path: str,
             patient_id: str,
             out_dir: str,
             tile_size: int = 256,
             target_mag: float = 20.0,
             tissue_threshold: float = 0.50,
             max_tiles: int = 2000) -> List[str]:
    """
    Extract non-overlapping tiles from a WSI and save them as PNGs.

    Args:
        wsi_path:          Path to the WSI file.
        patient_id:        Patient identifier (used for subdirectory name).
        out_dir:           Root output directory.
        tile_size:         Tile size in pixels at target magnification.
        target_mag:        Target magnification (20×).
        tissue_threshold:  Minimum tissue fraction per tile (0–1).
        max_tiles:         Maximum number of tiles to extract per slide.

    Returns:
        List of saved tile file paths.
    """
    if not OPENSLIDE_AVAILABLE:
        raise RuntimeError("openslide-python is required for WSI tiling.")

    slide = openslide.OpenSlide(wsi_path)
    level, resize_factor, native_mag = get_best_level(slide, target_mag)
    level_dims = slide.level_dimensions[level]

    # Effective tile size at the chosen level
    effective_tile = int(tile_size * resize_factor)

    # Thumbnail for tissue mask (scale-down ~64×)
    thumb_scale = 64
    thumb_w = max(1, level_dims[0] // thumb_scale)
    thumb_h = max(1, level_dims[1] // thumb_scale)
    thumbnail = np.array(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
    tissue_mask = get_tissue_mask(thumbnail)

    patient_dir = os.path.join(out_dir, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    saved_paths: List[str] = []
    cols = level_dims[0] // effective_tile
    rows = level_dims[1] // effective_tile

    # Collect candidate tile positions
    candidates = []
    for row in range(rows):
        for col in range(cols):
            # Check tissue mask at thumbnail resolution
            mx = int(col * thumb_scale * effective_tile / level_dims[0])
            my = int(row * thumb_scale * effective_tile / level_dims[1])
            mx = min(mx, tissue_mask.shape[1] - 1)
            my = min(my, tissue_mask.shape[0] - 1)
            if tissue_mask[my, mx]:
                candidates.append((row, col))

    # Shuffle and cap
    rng = np.random.default_rng(42)
    rng.shuffle(candidates)
    candidates = candidates[:max_tiles]

    for row, col in tqdm(candidates, desc=f"Tiling {patient_id}", leave=False):
        # Level-0 coordinates
        x_l0 = int(col * effective_tile * slide.level_downsamples[level])
        y_l0 = int(row * effective_tile * slide.level_downsamples[level])

        try:
            region = slide.read_region((x_l0, y_l0), level,
                                       (effective_tile, effective_tile))
            tile_rgb = np.array(region.convert("RGB"))
        except Exception as e:
            logger.debug(f"Skipping tile ({row},{col}): {e}")
            continue

        # Resize if needed
        if resize_factor != 1.0:
            tile_rgb = cv2.resize(tile_rgb, (tile_size, tile_size),
                                  interpolation=cv2.INTER_LINEAR)

        if not tile_has_tissue(tile_rgb, tissue_threshold):
            continue

        tile_path = os.path.join(patient_dir, f"{row}_{col}.png")
        Image.fromarray(tile_rgb).save(tile_path)
        saved_paths.append(tile_path)

    slide.close()
    logger.info(f"Patient {patient_id}: saved {len(saved_paths)} tiles → {patient_dir}")
    return saved_paths


# ──────────────────────────────────────────────────────────────────────────────
# Batch Tiling Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def tile_cohort(csv_path: str,
                out_dir: str,
                tile_size: int = 256,
                target_mag: float = 20.0,
                tissue_threshold: float = 0.50,
                max_tiles: int = 2000,
                resume: bool = True) -> dict:
    """
    Tile all WSIs listed in a clinical CSV file.

    CSV must contain columns: patient_id, wsi_path

    Returns a dict mapping patient_id → list of tile paths.
    """
    df = pd.read_csv(csv_path)
    required = {"patient_id", "wsi_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    manifest_path = os.path.join(out_dir, "tile_manifest.json")
    manifest: dict = {}
    if resume and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        logger.info(f"Resuming: {len(manifest)} patients already tiled.")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cohort tiling"):
        pid = str(row["patient_id"])
        wsi = str(row["wsi_path"])

        if pid in manifest:
            logger.debug(f"Skipping {pid} (already tiled).")
            continue

        if not os.path.exists(wsi):
            logger.warning(f"WSI not found for {pid}: {wsi}")
            manifest[pid] = []
            continue

        try:
            tiles = tile_wsi(wsi, pid, out_dir,
                             tile_size=tile_size,
                             target_mag=target_mag,
                             tissue_threshold=tissue_threshold,
                             max_tiles=max_tiles)
            manifest[pid] = tiles
        except Exception as e:
            logger.error(f"Error tiling {pid}: {e}")
            manifest[pid] = []

        # Save manifest after each slide so we can resume
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    logger.info(f"Tiling complete. Manifest saved to {manifest_path}")
    return manifest


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Tile WSIs for survival analysis")
    p.add_argument("--csv",          required=True, help="Clinical CSV path")
    p.add_argument("--out_dir",      default="data/tiles")
    p.add_argument("--tile_size",    type=int,   default=256)
    p.add_argument("--magnification",type=float, default=20.0)
    p.add_argument("--tissue_threshold", type=float, default=0.5)
    p.add_argument("--max_tiles",    type=int,   default=2000)
    p.add_argument("--no_resume",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    tile_cohort(
        csv_path=args.csv,
        out_dir=args.out_dir,
        tile_size=args.tile_size,
        target_mag=args.magnification,
        tissue_threshold=args.tissue_threshold,
        max_tiles=args.max_tiles,
        resume=not args.no_resume,
    )
