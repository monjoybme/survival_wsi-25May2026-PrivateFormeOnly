"""
data/splitter.py
----------------
Handles patient-level dataset splitting:
  1. Sherlock-Lung: 454 training / 141 held-out internal validation
  2. 10-fold stratified CV within the 454-patient training set
  3. MSK cohort: used only for external validation (no split needed)

All splits are saved as JSON so that the exact partition can be reproduced
and inspected without re-running this script.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _event_strata(df: pd.DataFrame, n_bins: int = 4) -> np.ndarray:
    """
    Create strata for stratified K-Fold: combine event indicator with
    quantile-binned survival time so that each fold has a similar
    event-rate and time distribution.
    """
    time_bin = pd.qcut(df["survival_time"], q=n_bins, labels=False, duplicates="drop")
    strata = df["event"].astype(str) + "_" + time_bin.astype(str)
    # Encode as integers
    return pd.Categorical(strata).codes


# ──────────────────────────────────────────────────────────────────────────────
# Main Splitter
# ──────────────────────────────────────────────────────────────────────────────

class DataSplitter:
    """
    Manages all dataset splits for the survival analysis pipeline.

    Args:
        sherlock_csv:  Path to Sherlock-Lung clinical CSV.
        msk_csv:       Path to MSK clinical CSV (external validation only).
        tiles_dir:     Root directory containing patient tile folders.
        splits_dir:    Directory where split JSONs are saved.
        train_n:       Target number of training patients (≈454).
        n_folds:       Number of CV folds within the training set.
        random_seed:   RNG seed for reproducibility.
    """

    SPLIT_FILE   = "train_val_split.json"
    CV_FILE      = "cv_folds.json"
    MSK_FILE     = "msk_patients.json"
    TILE_MAP_FILE = "tile_map.json"

    def __init__(
        self,
        sherlock_csv: str,
        msk_csv: Optional[str],
        tiles_dir: str,
        splits_dir: str,
        train_n: int = 454,
        n_folds: int = 10,
        random_seed: int = 42,
    ):
        self.sherlock_csv = sherlock_csv
        self.msk_csv      = msk_csv
        self.tiles_dir    = tiles_dir
        self.splits_dir   = splits_dir
        self.train_n      = train_n
        self.n_folds      = n_folds
        self.random_seed  = random_seed

        os.makedirs(splits_dir, exist_ok=True)

        self._sherlock_df: Optional[pd.DataFrame] = None
        self._msk_df:      Optional[pd.DataFrame] = None

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_sherlock(self) -> pd.DataFrame:
        if self._sherlock_df is None:
            df = pd.read_csv(self.sherlock_csv)
            df = self._validate_and_clean(df, "Sherlock-Lung")
            self._sherlock_df = df
        return self._sherlock_df

    def _load_msk(self) -> Optional[pd.DataFrame]:
        if self.msk_csv is None or not os.path.exists(self.msk_csv):
            return None
        if self._msk_df is None:
            df = pd.read_csv(self.msk_csv)
            df = self._validate_and_clean(df, "MSK")
            self._msk_df = df
        return self._msk_df

    @staticmethod
    def _validate_and_clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
        required = {"patient_id", "survival_time", "event"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")

        df = df.copy()
        df["patient_id"]    = df["patient_id"].astype(str)
        df["survival_time"] = pd.to_numeric(df["survival_time"], errors="coerce")
        df["event"]         = pd.to_numeric(df["event"],         errors="coerce").astype(int)

        before = len(df)
        df = df.dropna(subset=["survival_time", "event"])
        df = df[df["survival_time"] > 0]
        if len(df) < before:
            logger.warning(f"{name}: dropped {before - len(df)} rows with invalid labels.")

        df = df.drop_duplicates(subset="patient_id")
        logger.info(f"{name}: {len(df)} patients loaded.")
        return df.reset_index(drop=True)

    # ── Tile Map ──────────────────────────────────────────────────────────────

    def build_tile_map(self, force: bool = False) -> Dict[str, List[str]]:
        """
        Scan tiles_dir and build {patient_id: [tile_paths]} mapping.
        Cached to disk.
        """
        cache = os.path.join(self.splits_dir, self.TILE_MAP_FILE)
        if os.path.exists(cache) and not force:
            with open(cache) as f:
                tile_map = json.load(f)
            logger.info(f"Tile map loaded ({len(tile_map)} patients).")
            return tile_map

        tile_map: Dict[str, List[str]] = {}
        tiles_root = Path(self.tiles_dir)
        if not tiles_root.exists():
            logger.warning(f"tiles_dir does not exist: {self.tiles_dir}")
            return tile_map

        for patient_dir in sorted(tiles_root.iterdir()):
            if patient_dir.is_dir():
                tiles = sorted(str(p) for p in patient_dir.glob("*.png"))
                if tiles:
                    tile_map[patient_dir.name] = tiles

        with open(cache, "w") as f:
            json.dump(tile_map, f, indent=2)
        logger.info(f"Tile map built: {len(tile_map)} patients, saved to {cache}")
        return tile_map

    # ── Train / Val Split ─────────────────────────────────────────────────────

    def get_train_val_split(
        self,
        force: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split Sherlock-Lung into training (n=454) and internal validation (n=141).

        The split is saved to disk; subsequent calls load it unless force=True.
        Only patients with at least one tile are included.
        """
        split_path = os.path.join(self.splits_dir, self.SPLIT_FILE)

        if os.path.exists(split_path) and not force:
            with open(split_path) as f:
                split = json.load(f)
            df = self._load_sherlock()
            train_df = df[df["patient_id"].isin(split["train"])].reset_index(drop=True)
            val_df   = df[df["patient_id"].isin(split["val"])].reset_index(drop=True)
            logger.info(f"Loaded existing split: train={len(train_df)}, val={len(val_df)}")
            return train_df, val_df

        df = self._load_sherlock()

        # Filter to patients who actually have tiles
        tile_map = self.build_tile_map()
        has_tiles = set(tile_map.keys())
        df_tiled  = df[df["patient_id"].isin(has_tiles)].reset_index(drop=True)
        n_available = len(df_tiled)

        if n_available < self.train_n + 1:
            logger.warning(
                f"Only {n_available} tiled patients available; "
                f"using 80/20 split instead of {self.train_n}/{len(df) - self.train_n}."
            )
            train_frac = 0.80
        else:
            train_frac = self.train_n / n_available

        # Stratified shuffle split
        rng   = np.random.default_rng(self.random_seed)
        strata = _event_strata(df_tiled)
        pids   = df_tiled["patient_id"].values
        idx    = np.arange(n_available)
        rng.shuffle(idx)

        # Maintain approximate event-rate across splits
        train_idx, val_idx = self._stratified_split(
            idx, strata, train_frac, rng
        )

        train_pids = pids[train_idx].tolist()
        val_pids   = pids[val_idx].tolist()

        split = {"train": train_pids, "val": val_pids}
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)

        train_df = df_tiled.iloc[train_idx].reset_index(drop=True)
        val_df   = df_tiled.iloc[val_idx].reset_index(drop=True)

        logger.info(
            f"Split saved → train: {len(train_df)} | val: {len(val_df)} "
            f"(event rates: {train_df['event'].mean():.2f} / {val_df['event'].mean():.2f})"
        )
        return train_df, val_df

    @staticmethod
    def _stratified_split(
        idx: np.ndarray,
        strata: np.ndarray,
        train_frac: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stratified split preserving event rate.
        """
        train_idx_list, val_idx_list = [], []
        unique_strata = np.unique(strata)
        for s in unique_strata:
            s_idx = idx[strata[idx] == s]
            rng.shuffle(s_idx)
            n_train = max(1, int(len(s_idx) * train_frac))
            train_idx_list.extend(s_idx[:n_train].tolist())
            val_idx_list.extend(s_idx[n_train:].tolist())
        return np.array(train_idx_list), np.array(val_idx_list)

    # ── 10-Fold CV ────────────────────────────────────────────────────────────

    def get_cv_folds(
        self,
        train_df: pd.DataFrame,
        force: bool = False,
    ) -> List[Dict]:
        """
        Generate 10-fold CV splits within the training set.

        Each fold dictionary contains:
            {"fold": int, "train_pids": [...], "val_pids": [...]}

        Out-of-fold predictions cover the entire training set.
        """
        cv_path = os.path.join(self.splits_dir, self.CV_FILE)

        if os.path.exists(cv_path) and not force:
            with open(cv_path) as f:
                folds = json.load(f)
            logger.info(f"Loaded existing {len(folds)}-fold CV splits.")
            return folds

        pids   = train_df["patient_id"].values
        strata = _event_strata(train_df)

        skf    = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                 random_state=self.random_seed)
        folds  = []
        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(pids, strata)):
            folds.append({
                "fold":      fold_i,
                "train_pids": pids[tr_idx].tolist(),
                "val_pids":   pids[va_idx].tolist(),
            })
            logger.info(
                f"Fold {fold_i}: train={len(tr_idx)} "
                f"(~{len(tr_idx)}) | val={len(va_idx)}"
            )

        with open(cv_path, "w") as f:
            json.dump(folds, f, indent=2)
        logger.info(f"CV folds saved to {cv_path}")
        return folds

    # ── MSK (External Validation) ─────────────────────────────────────────────

    def get_msk_patients(self) -> Optional[pd.DataFrame]:
        """Return MSK patient DataFrame for external validation."""
        df = self._load_msk()
        if df is None:
            logger.info("No MSK cohort CSV provided.")
            return None

        tile_map = self.build_tile_map()
        has_tiles = set(tile_map.keys())
        df_tiled  = df[df["patient_id"].isin(has_tiles)].reset_index(drop=True)
        logger.info(f"MSK external validation: {len(df_tiled)} tiled patients.")

        msk_path = os.path.join(self.splits_dir, self.MSK_FILE)
        with open(msk_path, "w") as f:
            json.dump(df_tiled["patient_id"].tolist(), f, indent=2)
        return df_tiled
