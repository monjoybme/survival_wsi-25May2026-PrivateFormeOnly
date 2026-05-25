"""
utils/visualization.py
-----------------------
Plotting utilities:
  • Loss / metric curves across epochs
  • Kaplan-Meier survival curves (high vs. low risk)
  • C-index bar charts across CV folds
  • Risk score distributions
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)

_STYLE  = "seaborn-v0_8-whitegrid"
_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]


# ──────────────────────────────────────────────────────────────────────────────
# Loss / Metric Curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(
    train_losses: List[float],
    val_losses:   List[float],
    save_path:    str,
    title:        str = "Training & Validation Loss",
    xlabel:       str = "Epoch",
    ylabel:       str = "Cox Negative Log-Likelihood",
) -> None:
    """
    Plot training and validation loss across epochs and save to disk.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, train_losses, color=_COLORS[0], lw=2, label="Train")
    ax.plot(epochs, val_losses,   color=_COLORS[1], lw=2, label="Validation",
            linestyle="--")

    best_epoch = int(np.argmin(val_losses)) + 1
    best_val   = min(val_losses)
    ax.axvline(best_epoch, color="gray", linestyle=":", lw=1.5,
               label=f"Best val @ epoch {best_epoch} ({best_val:.3f})")
    ax.scatter([best_epoch], [best_val], color=_COLORS[1], zorder=5, s=60)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Loss curve saved → {save_path}")


def plot_metric_curve(
    values:    List[float],
    save_path: str,
    label:     str = "C-index",
    title:     str = "Validation C-index per Epoch",
) -> None:
    """
    Plot a single metric across epochs (e.g. validation C-index).
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    epochs = range(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, values, color=_COLORS[2], lw=2, label=label)

    best_epoch = int(np.argmax(values)) + 1
    best_val   = max(values)
    ax.axvline(best_epoch, color="gray", linestyle=":", lw=1.5,
               label=f"Best @ epoch {best_epoch} ({best_val:.3f})")
    ax.scatter([best_epoch], [best_val], color=_COLORS[2], zorder=5, s=60)
    ax.axhline(0.5, color="black", linestyle="--", lw=1.0, alpha=0.5,
               label="Random (0.50)")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Metric curve saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Kaplan-Meier Curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_kaplan_meier(
    times_high:  np.ndarray,
    events_high: np.ndarray,
    times_low:   np.ndarray,
    events_low:  np.ndarray,
    save_path:   str,
    title:       str = "Kaplan-Meier: High vs. Low Risk",
    pvalue:      Optional[float] = None,
    time_unit:   str = "Days",
) -> None:
    """
    Plot KM curves for high- and low-risk groups.
    Optionally annotate with log-rank p-value.
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        logger.warning("lifelines not installed — skipping KM plot.")
        return

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(8, 6))

    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()

    kmf_high.fit(times_high, event_observed=events_high, label="High risk")
    kmf_low.fit (times_low,  event_observed=events_low,  label="Low risk")

    kmf_high.plot_survival_function(ax=ax, ci_show=True, color=_COLORS[1])
    kmf_low.plot_survival_function (ax=ax, ci_show=True, color=_COLORS[0])

    if pvalue is None:
        lr = logrank_test(
            times_high, times_low,
            event_observed_A=events_high,
            event_observed_B=events_low,
        )
        pvalue = lr.p_value

    p_str = f"p = {pvalue:.4f}" if pvalue >= 0.0001 else "p < 0.0001"
    ax.text(
        0.70, 0.85, p_str,
        transform=ax.transAxes,
        fontsize=13, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )
    ax.text(
        0.70, 0.77,
        f"n_high={len(times_high)}, n_low={len(times_low)}",
        transform=ax.transAxes, fontsize=10, color="gray",
    )

    ax.set_xlabel(f"Time ({time_unit})", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"KM plot saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Risk Score Distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_risk_distribution(
    risk_scores: np.ndarray,
    events:      np.ndarray,
    save_path:   str,
    title:       str = "Risk Score Distribution",
    threshold:   Optional[float] = None,
) -> None:
    """
    Histogram of predicted risk scores, coloured by event status.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(risk_scores[events == 1], bins=40, alpha=0.7,
            color=_COLORS[1], label="Event (death)", density=True)
    ax.hist(risk_scores[events == 0], bins=40, alpha=0.7,
            color=_COLORS[0], label="Censored",      density=True)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", lw=2,
                   label=f"Threshold ({threshold:.3f})")

    ax.set_xlabel("Predicted Log-Hazard (Risk Score)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Risk distribution saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CV Fold C-index Bar Chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_cv_cindex(
    fold_cindex: List[float],
    save_path:   str,
    title:       str = "C-index per CV Fold",
) -> None:
    """
    Bar chart of C-index per CV fold with mean ± std annotation.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    n    = len(fold_cindex)
    mean = np.mean(fold_cindex)
    std  = np.std(fold_cindex)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), 5))
    bars = ax.bar(range(n), fold_cindex, color=_COLORS[2], alpha=0.85,
                  edgecolor="white", linewidth=1.2)
    ax.axhline(mean, color=_COLORS[1], linestyle="--", lw=2,
               label=f"Mean C-index: {mean:.3f} ± {std:.3f}")
    ax.axhline(0.5,  color="gray",     linestyle=":",  lw=1.5, alpha=0.6,
               label="Random (0.50)")

    for bar, val in zip(bars, fold_cindex):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("CV Fold", fontsize=12)
    ax.set_ylabel("C-index", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"Fold {i}" for i in range(n)], rotation=30)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"CV C-index chart saved → {save_path}")
