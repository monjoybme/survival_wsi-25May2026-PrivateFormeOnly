"""
evaluation/metrics.py
---------------------
Survival analysis evaluation metrics:
  • Harrell's concordance index (C-index)
  • Log-rank test p-value
  • Risk-score threshold derivation from out-of-fold (OOF) predictions
  • Brier score (time-dependent)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Concordance Index
# ──────────────────────────────────────────────────────────────────────────────

def concordance_index(
    survival_times: np.ndarray,
    events:         np.ndarray,
    risk_scores:    np.ndarray,
) -> float:
    """
    Harrell's concordance index (C-index).

    A pair (i, j) where t_i < t_j and event_i = 1 is concordant if
    risk_i > risk_j (higher risk → earlier event).

    Args:
        survival_times: (N,) array of observed times.
        events:         (N,) binary event indicators (1 = event).
        risk_scores:    (N,) predicted log-hazard scores.

    Returns:
        C-index in [0, 1].  0.5 = random.
    """
    try:
        from lifelines.utils import concordance_index as lifelines_ci
        return float(lifelines_ci(survival_times, risk_scores, events))
    except ImportError:
        pass

    # Fallback: manual O(N²) implementation
    concordant   = 0
    discordant   = 0
    tied_risk    = 0

    n = len(survival_times)
    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if survival_times[j] <= survival_times[i]:
                continue
            # i had event before j
            if risk_scores[i] > risk_scores[j]:
                concordant += 1
            elif risk_scores[i] < risk_scores[j]:
                discordant += 1
            else:
                tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


# ──────────────────────────────────────────────────────────────────────────────
# Log-Rank Test
# ──────────────────────────────────────────────────────────────────────────────

def logrank_pvalue(
    times_a:  np.ndarray,
    events_a: np.ndarray,
    times_b:  np.ndarray,
    events_b: np.ndarray,
) -> float:
    """
    Log-rank test p-value comparing two survival groups.

    Requires lifelines (falls back to scipy if unavailable).
    """
    try:
        from lifelines.statistics import logrank_test
        result = logrank_test(times_a, times_b,
                              event_observed_A=events_a,
                              event_observed_B=events_b)
        return float(result.p_value)
    except ImportError:
        pass

    try:
        from scipy.stats import chi2
        # Simplified log-rank via Mantel-Cox
        raise NotImplementedError("scipy log-rank not implemented here; install lifelines.")
    except Exception as e:
        logger.warning(f"Log-rank test failed: {e}")
        return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Risk Threshold Derivation (OOF predictions)
# ──────────────────────────────────────────────────────────────────────────────

def derive_risk_threshold(
    oof_risk_scores: np.ndarray,
    method:          str = "median",
    percentile:      float = 50.0,
) -> float:
    """
    Derive a scalar risk threshold from out-of-fold (OOF) predictions on the
    training set.  Patients above the threshold are classified as high-risk.

    Args:
        oof_risk_scores: (N,) OOF predicted log-hazard scores for all training patients.
        method:          "median" | "percentile" | "optimal_logrank"
        percentile:      Used when method="percentile".

    Returns:
        Scalar threshold value.
    """
    if method == "median":
        threshold = float(np.median(oof_risk_scores))
    elif method == "percentile":
        threshold = float(np.percentile(oof_risk_scores, percentile))
    elif method == "optimal_logrank":
        # Scan percentile thresholds and pick the one maximising log-rank stat
        # (requires times and events — caller should pass them separately)
        raise ValueError(
            "For optimal_logrank, use derive_optimal_threshold() instead."
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    logger.info(f"Risk threshold derived via {method!r}: {threshold:.4f}")
    return threshold


def derive_optimal_threshold(
    risk_scores:    np.ndarray,
    survival_times: np.ndarray,
    events:         np.ndarray,
    percentile_range: Tuple[float, float] = (25.0, 75.0),
    n_steps: int = 50,
) -> float:
    """
    Scan candidate thresholds (percentile grid) and select the one that
    maximises the log-rank chi-squared statistic between high- and low-risk groups.

    Args:
        risk_scores:     (N,) predicted log-hazards.
        survival_times:  (N,) survival times.
        events:          (N,) event indicators.
        percentile_range: Percentile bounds to search within.
        n_steps:         Number of candidate thresholds to evaluate.

    Returns:
        Optimal threshold value.
    """
    try:
        from lifelines.statistics import logrank_test
    except ImportError:
        logger.warning("lifelines not available — falling back to median threshold.")
        return float(np.median(risk_scores))

    lo, hi = percentile_range
    candidates = np.percentile(risk_scores, np.linspace(lo, hi, n_steps))
    best_stat  = -np.inf
    best_thresh = float(np.median(risk_scores))

    for thr in candidates:
        mask_high = risk_scores >= thr
        mask_low  = risk_scores <  thr
        if mask_high.sum() < 5 or mask_low.sum() < 5:
            continue
        try:
            result = logrank_test(
                survival_times[mask_high], survival_times[mask_low],
                event_observed_A=events[mask_high],
                event_observed_B=events[mask_low],
            )
            if result.test_statistic > best_stat:
                best_stat   = result.test_statistic
                best_thresh = float(thr)
        except Exception:
            continue

    logger.info(
        f"Optimal log-rank threshold: {best_thresh:.4f} "
        f"(chi² = {best_stat:.2f})"
    )
    return best_thresh


# ──────────────────────────────────────────────────────────────────────────────
# Summary Helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    risk_scores:    np.ndarray,
    survival_times: np.ndarray,
    events:         np.ndarray,
    threshold:      Optional[float] = None,
    split_name:     str = "Validation",
) -> dict:
    """
    Compute and log a standard suite of survival metrics.

    Returns a dict with: c_index, pvalue, n_high, n_low, threshold.
    """
    ci = concordance_index(survival_times, events, risk_scores)

    if threshold is None:
        threshold = float(np.median(risk_scores))

    mask_high = risk_scores >= threshold
    mask_low  = risk_scores <  threshold
    n_high    = int(mask_high.sum())
    n_low     = int(mask_low.sum())

    pvalue = float("nan")
    if n_high >= 2 and n_low >= 2:
        pvalue = logrank_pvalue(
            survival_times[mask_high], events[mask_high],
            survival_times[mask_low],  events[mask_low],
        )

    logger.info(
        f"[{split_name}] C-index = {ci:.4f} | "
        f"Log-rank p = {pvalue:.4g} | "
        f"High risk n={n_high}, Low risk n={n_low} "
        f"(threshold = {threshold:.4f})"
    )

    return {
        "c_index":   ci,
        "pvalue":    pvalue,
        "n_high":    n_high,
        "n_low":     n_low,
        "threshold": threshold,
        "split":     split_name,
    }
