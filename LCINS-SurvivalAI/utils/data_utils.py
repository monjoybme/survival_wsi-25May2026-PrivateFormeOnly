"""
utils/data_utils.py
-------------------
Utility functions for:
  • Validating clinical CSV files before training
  • Generating template / example CSV files
  • Summarising cohort statistics (event rate, follow-up, etc.)
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Required / Optional CSV Columns
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS  = {"patient_id", "wsi_path", "survival_time", "event"}
OPTIONAL_COLS  = {"age", "sex", "stage", "histology", "cohort"}


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_clinical_csv(
    csv_path: str,
    check_files: bool = False,
    cohort_name: str  = "Cohort",
) -> pd.DataFrame:
    """
    Validate a clinical CSV and print a summary.  Raises ValueError on
    critical issues (missing required columns, all-zero event vector, etc.).

    Args:
        csv_path:     Path to the CSV.
        check_files:  If True, verify that wsi_path values exist on disk.
        cohort_name:  Label for log messages.

    Returns:
        Cleaned DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"[{cohort_name}] Loaded CSV: {len(df)} rows, {len(df.columns)} columns.")

    # ── Required columns ──────────────────────────────────────────────────────
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"[{cohort_name}] Missing required columns: {missing}\n"
            f"  Present columns: {list(df.columns)}"
        )

    df = df.copy()
    df["patient_id"]    = df["patient_id"].astype(str)
    df["survival_time"] = pd.to_numeric(df["survival_time"], errors="coerce")
    df["event"]         = pd.to_numeric(df["event"],         errors="coerce").astype("Int64")

    # ── Missing values ────────────────────────────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=["survival_time", "event"])
    df = df[df["survival_time"] > 0]
    df = df[df["event"].isin([0, 1])]
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(f"[{cohort_name}] Dropped {n_dropped} rows with invalid labels.")

    # ── Duplicates ────────────────────────────────────────────────────────────
    n_dups = df["patient_id"].duplicated().sum()
    if n_dups:
        logger.warning(f"[{cohort_name}] {n_dups} duplicate patient_id rows — keeping first.")
        df = df.drop_duplicates(subset="patient_id", keep="first")

    # ── Event sanity ──────────────────────────────────────────────────────────
    event_rate = df["event"].mean()
    if event_rate == 0:
        raise ValueError(f"[{cohort_name}] All events are 0 — no outcomes to learn from.")
    if event_rate == 1:
        logger.warning(f"[{cohort_name}] All events are 1 — no censored cases.")

    # ── File existence check ───────────────────────────────────────────────────
    if check_files:
        missing_wsi = df[~df["wsi_path"].apply(os.path.exists)]
        if not missing_wsi.empty:
            logger.warning(
                f"[{cohort_name}] {len(missing_wsi)} WSI files not found on disk:\n"
                + "\n".join(f"  {r['patient_id']}: {r['wsi_path']}"
                            for _, r in missing_wsi.head(5).iterrows())
                + (" ..." if len(missing_wsi) > 5 else "")
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        f"[{cohort_name}] Summary:\n"
        f"  Patients        : {len(df)}\n"
        f"  Event rate      : {event_rate:.2%}\n"
        f"  Median surv time: {df['survival_time'].median():.1f}\n"
        f"  Surv time range : [{df['survival_time'].min():.1f}, "
        f"{df['survival_time'].max():.1f}]"
    )
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Template CSV Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_template_csv(
    out_path: str,
    n_patients: int = 10,
    cohort:     str = "sherlock",
) -> None:
    """
    Generate a template clinical CSV file with dummy data.

    IMPORTANT: Replace wsi_path values with actual file paths before running
    the tiling or training pipelines.

    Args:
        out_path:   Where to save the template CSV.
        n_patients: Number of example rows.
        cohort:     "sherlock" or "msk".
    """
    rng = np.random.default_rng(0)
    df  = pd.DataFrame({
        "patient_id":    [f"{cohort.upper()}_{i+1:04d}" for i in range(n_patients)],
        "wsi_path":      [f"/path/to/wsi/{cohort}_{i+1:04d}.svs" for i in range(n_patients)],
        "survival_time": rng.uniform(30, 2000, n_patients).round(1),
        "event":         rng.integers(0, 2, n_patients),
        "age":           rng.integers(45, 85, n_patients),
        "sex":           rng.choice(["M", "F"], n_patients),
        "stage":         rng.choice(["I", "II", "III", "IV"], n_patients),
    })
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(
        f"Template CSV saved → {out_path}\n"
        f"  *** Replace 'wsi_path' with real WSI file paths before use. ***"
    )
    print(df.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    # validate subcommand
    v = sub.add_parser("validate", help="Validate a clinical CSV.")
    v.add_argument("csv_path")
    v.add_argument("--check_files", action="store_true")
    v.add_argument("--cohort", default="Cohort")

    # template subcommand
    t = sub.add_parser("template", help="Generate a template CSV.")
    t.add_argument("out_path")
    t.add_argument("--n", type=int, default=10)
    t.add_argument("--cohort", default="sherlock")

    args = p.parse_args()

    if args.cmd == "validate":
        validate_clinical_csv(args.csv_path,
                              check_files=args.check_files,
                              cohort_name=args.cohort)
    elif args.cmd == "template":
        generate_template_csv(args.out_path, n_patients=args.n, cohort=args.cohort)
    else:
        p.print_help()
