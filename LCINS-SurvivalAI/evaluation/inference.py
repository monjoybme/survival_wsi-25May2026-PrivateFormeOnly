"""
evaluation/inference.py
-----------------------
Inference engine: applies the trained model to every tile in a cohort and
aggregates tile-level log-hazard scores to produce slide-level risk scores.

Aggregation strategies (mean / median / max):
    mean  — primary strategy (paper default, numerically stable).
    median — robust to extreme-risk tiles.
    max    — upper-tail aggregation (captures highest-risk regions).
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import CohortInferDataset
from data.sampler  import get_val_transform
from evaluation.metrics import concordance_index, compute_all_metrics

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Tile-Level Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_tiles(
    model:       torch.nn.Module,
    patient_df:  pd.DataFrame,
    tile_map:    Dict[str, List[str]],
    device:      str  = "cuda",
    batch_size:  int  = 256,
    num_workers: int  = 4,
    tile_size:   int  = 256,
) -> Dict[str, np.ndarray]:
    """
    Run the model on every tile of every slide in patient_df and collect
    tile-level log-hazard scores.

    Args:
        model:      Trained VGG16Survival (in eval mode).
        patient_df: DataFrame with patient_id, survival_time, event.
        tile_map:   Dict[patient_id → list of tile paths].
        device:     Inference device.
        batch_size: Tiles per batch.
        num_workers: DataLoader workers.
        tile_size:  Expected input tile size (pixels).

    Returns:
        Dict[patient_id → np.ndarray of shape (n_tiles,)] of log-hazard scores.
    """
    model = model.to(device)
    model.eval()

    transform = get_val_transform(tile_size)
    dataset   = CohortInferDataset(patient_df, tile_map, transform=transform)

    if len(dataset) == 0:
        logger.warning("Inference dataset is empty — no tiles found.")
        return {}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    # patient_id → list of tile scores
    tile_scores: Dict[str, List[float]] = defaultdict(list)

    for batch in tqdm(loader, desc="Inference", leave=False):
        imgs  = batch["image"].to(device, non_blocking=True)
        pids  = batch["patient_id"]

        with torch.autocast(device_type=device.split(":")[0],
                            enabled=(device != "cpu")):
            scores = model(imgs)           # (B,) log-hazards

        scores_np = scores.float().cpu().numpy()
        for pid, sc in zip(pids, scores_np):
            tile_scores[pid].append(float(sc))

    return {pid: np.array(sc) for pid, sc in tile_scores.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Slide-Level Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_tile_scores(
    tile_scores:  Dict[str, np.ndarray],
    method:       str = "mean",
) -> Dict[str, float]:
    """
    Aggregate per-tile log-hazard scores into a single slide-level risk score.

    Args:
        tile_scores: Dict[patient_id → np.ndarray of tile log-hazards].
        method:      "mean" | "median" | "max".

    Returns:
        Dict[patient_id → scalar risk score].
    """
    agg_fn = {"mean": np.mean, "median": np.median, "max": np.max}.get(method)
    if agg_fn is None:
        raise ValueError(f"Unknown aggregation method: {method!r}")

    slide_scores: Dict[str, float] = {}
    for pid, scores in tile_scores.items():
        if len(scores) == 0:
            logger.warning(f"Patient {pid} has no tile scores — skipping.")
            continue
        slide_scores[pid] = float(agg_fn(scores))

    logger.info(
        f"Aggregated {len(slide_scores)} slides via {method!r}. "
        f"Score range: [{min(slide_scores.values()):.3f}, "
        f"{max(slide_scores.values()):.3f}]"
    )
    return slide_scores


# ──────────────────────────────────────────────────────────────────────────────
# End-to-End Cohort Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_cohort(
    model:          torch.nn.Module,
    patient_df:     pd.DataFrame,
    tile_map:       Dict[str, List[str]],
    threshold:      Optional[float]  = None,
    aggregation:    str              = "mean",
    device:         str              = "cuda",
    batch_size:     int              = 256,
    num_workers:    int              = 4,
    split_name:     str              = "Validation",
    results_dir:    str              = "results",
    figures_dir:    str              = "figures",
    tile_size:      int              = 256,
) -> Tuple[pd.DataFrame, dict]:
    """
    Full evaluation pipeline for a patient cohort:
      1. Tile-level inference
      2. Slide-level aggregation
      3. C-index + log-rank computation
      4. KM plot + risk distribution plot

    Args:
        model:       Trained model.
        patient_df:  DataFrame with patient_id, survival_time, event.
        tile_map:    Tile path mapping.
        threshold:   Risk dichotomisation threshold (derived from OOF if None).
        aggregation: "mean" | "median" | "max".
        device:      Torch device string.
        batch_size:  Inference batch size.
        num_workers: DataLoader workers.
        split_name:  Label for plots/logs (e.g. "Internal Val", "MSK").
        results_dir: Where to save the results CSV.
        figures_dir: Where to save plots.
        tile_size:   Expected tile size in pixels.

    Returns:
        results_df: DataFrame with patient_id, risk_score, survival_time, event.
        metrics:    Dict of evaluation metrics.
    """
    from utils.visualization import plot_kaplan_meier, plot_risk_distribution

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ── 1. Tile inference ─────────────────────────────────────────────────────
    tile_scores = predict_tiles(
        model, patient_df, tile_map,
        device=device, batch_size=batch_size,
        num_workers=num_workers, tile_size=tile_size,
    )

    # ── 2. Slide-level aggregation ────────────────────────────────────────────
    slide_scores = aggregate_tile_scores(tile_scores, method=aggregation)

    # ── 3. Build results DataFrame ────────────────────────────────────────────
    records = []
    for _, row in patient_df.iterrows():
        pid = str(row["patient_id"])
        if pid not in slide_scores:
            logger.warning(f"No score for patient {pid} — excluded.")
            continue
        records.append({
            "patient_id":    pid,
            "risk_score":    slide_scores[pid],
            "survival_time": float(row["survival_time"]),
            "event":         int(row["event"]),
        })

    results_df = pd.DataFrame(records)

    risk_arr  = results_df["risk_score"].values
    time_arr  = results_df["survival_time"].values
    event_arr = results_df["event"].values

    # ── 4. Risk threshold ─────────────────────────────────────────────────────
    if threshold is None:
        threshold = float(np.median(risk_arr))
        logger.info(f"No threshold provided; using median: {threshold:.4f}")

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    metrics = compute_all_metrics(
        risk_arr, time_arr, event_arr, threshold, split_name
    )

    # ── 6. Save results ───────────────────────────────────────────────────────
    tag = split_name.lower().replace(" ", "_")
    csv_path = os.path.join(results_dir, f"{tag}_risk_scores.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Risk scores saved → {csv_path}")

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    mask_high = risk_arr >= threshold
    mask_low  = ~mask_high

    if mask_high.sum() >= 2 and mask_low.sum() >= 2:
        plot_kaplan_meier(
            times_high=time_arr[mask_high],  events_high=event_arr[mask_high],
            times_low=time_arr[mask_low],    events_low=event_arr[mask_low],
            save_path=os.path.join(figures_dir, f"{tag}_km_curve.png"),
            title=f"KM Curve — {split_name}",
            pvalue=metrics.get("pvalue"),
        )

    plot_risk_distribution(
        risk_scores=risk_arr,
        events=event_arr,
        save_path=os.path.join(figures_dir, f"{tag}_risk_dist.png"),
        title=f"Risk Distribution — {split_name}",
        threshold=threshold,
    )

    return results_df, metrics
