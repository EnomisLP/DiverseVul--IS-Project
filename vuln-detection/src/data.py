"""
Data loading, splitting, and fold generation for the Vulnerability Detection System.

This module is the SINGLE source of truth for all data partitioning.
It must be the first thing called in any notebook or script — before any
EDA, feature extraction, or model training.

Strict ordering enforced here:
  1. load_rdiversevul()      — read raw CSV from data/raw/
  2. make_holdout_split()    — stratified 20% holdout, saved to data/splits/
  3. get_cv_folds()          — StratifiedKFold on the remaining 80% only

The holdout split is saved as numpy index arrays so every notebook and both
tracks use byte-for-byte identical partitions regardless of call order.

Public API:
  - load_rdiversevul        : read and validate the raw dataset
  - make_holdout_split      : one-time stratified holdout split + persist
  - load_holdout_split      : reload saved split indices (used by all notebooks)
  - get_cv_folds            : yield (train_idx, val_idx) for each CV fold
  - get_texts_and_labels    : extract (X, y) arrays from a DataFrame subset
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils import get_logger, load_config

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column name constants  (adjust if RDiverseVul uses different field names)
# ---------------------------------------------------------------------------

COL_FUNC   = "func"        # raw C/C++ function source text
COL_TARGET = "target"      # binary label: 1 = vulnerable, 0 = non-vulnerable
COL_CWE    = "cwe_id"      # CWE identifier string, e.g. "CWE-119"
COL_PROJECT = "project"    # originating open-source project

# Primary CWE categories tracked in this study
PRIMARY_CWES = {"CWE-119", "CWE-787", "CWE-476", "CWE-125"}

# Expected split files written to data/splits/
_SPLIT_DIR          = Path("data/splits")
_TRAIN_IDX_FILE     = _SPLIT_DIR / "train_idx.npy"
_HOLDOUT_IDX_FILE   = _SPLIT_DIR / "holdout_idx.npy"
_SPLIT_META_FILE    = _SPLIT_DIR / "split_meta.json"


# ---------------------------------------------------------------------------
# 1. Dataset loading
# ---------------------------------------------------------------------------

def load_rdiversevul(
    data_dir: Union[str, Path] = "data/raw",
    *,
    filename: Optional[str] = None,
    drop_duplicates: bool = True,
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load the RDiverseVul dataset from ``data/raw/``.

    Searches *data_dir* for a CSV or JSON Lines file.  If *filename* is
    provided, that specific file is loaded; otherwise the first ``.csv`` or
    ``.jsonl`` file found is used.

    Post-load validation:
      - Required columns are present.
      - Label column contains only {0, 1}.
      - Duplicate rows optionally removed.
      - Basic statistics logged for sanity-checking.

    Args:
        data_dir:         Path to ``data/raw/`` (default ``"data/raw"``).
        filename:         Specific filename to load.  Auto-detected if ``None``.
        drop_duplicates:  Remove exact duplicate rows (default ``True``).
        required_columns: Columns that must be present.  Defaults to
                          ``[COL_FUNC, COL_TARGET]``.

    Returns:
        :class:`pandas.DataFrame` with at least ``func`` and ``target`` columns.

    Raises:
        FileNotFoundError: If no suitable file is found in *data_dir*.
        ValueError:        If required columns are missing or labels are invalid.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"data/raw/ directory not found at '{data_dir}'. "
            "Please load RDiverseVul manually into data/raw/ as described in "
            "the project README (data is gitignored and not committed)."
        )

    # --- locate the file ---
    if filename is not None:
        fpath = data_dir / filename
        if not fpath.exists():
            raise FileNotFoundError(f"Specified file not found: {fpath}")
    else:
        candidates = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"No .csv or .jsonl file found in '{data_dir}'. "
                "Download RDiverseVul and place it there."
            )
        fpath = candidates[0]
        if len(candidates) > 1:
            _log.warning(
                "Multiple files found in data/raw/; loading '%s'. "
                "Pass filename= to be explicit.",
                fpath.name,
            )

    _log.info("Loading dataset from: %s", fpath)

    # --- read ---
    if fpath.suffix == ".csv":
        df = pd.read_csv(fpath, low_memory=False)
    elif fpath.suffix == ".jsonl":
        df = pd.read_json(fpath, lines=True)
    else:
        raise ValueError(f"Unsupported file extension: {fpath.suffix}")

    _log.info("Raw shape: %s", df.shape)

    # --- validate columns ---
    if required_columns is None:
        required_columns = [COL_FUNC, COL_TARGET]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from dataset: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # --- validate labels ---
    unique_labels = set(df[COL_TARGET].dropna().unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Label column '{COL_TARGET}' must contain only {{0, 1}}. "
            f"Found: {unique_labels}"
        )

    # --- drop duplicates ---
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=[COL_FUNC, COL_TARGET])
        removed = before - len(df)
        if removed > 0:
            _log.info("Removed %d duplicate rows (before=%d, after=%d).", removed, before, len(df))

    # --- drop rows with null text or labels ---
    null_mask = df[COL_FUNC].isna() | df[COL_TARGET].isna()
    if null_mask.any():
        _log.warning("Dropping %d rows with null func or target.", null_mask.sum())
        df = df[~null_mask].reset_index(drop=True)

    # --- reset index so integer positions are contiguous ---
    df = df.reset_index(drop=True)

    # --- log class distribution ---
    counts = df[COL_TARGET].value_counts().sort_index()
    total  = len(df)
    _log.info(
        "Class distribution — vulnerable (1): %d (%.1f%%)  |  "
        "non-vulnerable (0): %d (%.1f%%)",
        counts.get(1, 0), 100 * counts.get(1, 0) / total,
        counts.get(0, 0), 100 * counts.get(0, 0) / total,
    )

    if COL_CWE in df.columns:
        cwe_counts = df[df[COL_TARGET] == 1][COL_CWE].value_counts()
        primary_counts = {c: int(cwe_counts.get(c, 0)) for c in PRIMARY_CWES}
        _log.info("Primary CWE counts: %s", primary_counts)

    return df


# ---------------------------------------------------------------------------
# 2. Holdout split — create once, persist, reload
# ---------------------------------------------------------------------------

def make_holdout_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    split_dir: Union[str, Path] = _SPLIT_DIR,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a stratified holdout split and persist indices to disk.

    Must be called **exactly once** per project lifecycle, before any EDA,
    feature fitting, or model training.  Subsequent calls should use
    :func:`load_holdout_split` to guarantee identical partitions across
    all experiments.

    The indices are integer positions into *df* (i.e. ``df.iloc[train_idx]``),
    not DataFrame index values, so they remain valid after ``reset_index``.

    Args:
        df:           Full dataset DataFrame from :func:`load_rdiversevul`.
        test_size:    Fraction of data for holdout (default 0.2 = 20%).
        random_state: RNG seed (default 42).
        split_dir:    Directory to write index arrays (default ``data/splits/``).
        overwrite:    If ``False`` (default) and split files already exist,
                      raises :class:`FileExistsError` to prevent accidental
                      re-splitting.

    Returns:
        Tuple ``(train_idx, holdout_idx)`` of integer numpy arrays.

    Raises:
        FileExistsError: If split files exist and *overwrite* is ``False``.
        ValueError:      If *df* has fewer than 10 rows or is missing labels.
    """
    split_dir = Path(split_dir)
    train_file   = split_dir / "train_idx.npy"
    holdout_file = split_dir / "holdout_idx.npy"
    meta_file    = split_dir / "split_meta.json"

    if train_file.exists() and not overwrite:
        raise FileExistsError(
            f"Split files already exist in '{split_dir}'. "
            "Use load_holdout_split() to reload them, or pass overwrite=True "
            "only if you intentionally want to re-split (this invalidates all "
            "previously saved metrics)."
        )

    if len(df) < 10:
        raise ValueError("DataFrame has fewer than 10 rows — cannot split.")

    labels = df[COL_TARGET].to_numpy(dtype=np.int64)
    all_idx = np.arange(len(df))

    train_idx, holdout_idx = train_test_split(
        all_idx,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    # verify stratification held
    train_pos_rate   = labels[train_idx].mean()
    holdout_pos_rate = labels[holdout_idx].mean()
    _log.info(
        "Split created — train: %d samples (%.1f%% vuln) | "
        "holdout: %d samples (%.1f%% vuln)",
        len(train_idx), 100 * train_pos_rate,
        len(holdout_idx), 100 * holdout_pos_rate,
    )

    # persist
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(train_file,   train_idx)
    np.save(holdout_file, holdout_idx)

    meta = {
        "test_size":          test_size,
        "random_state":       random_state,
        "n_total":            len(df),
        "n_train":            len(train_idx),
        "n_holdout":          len(holdout_idx),
        "train_pos_rate":     float(train_pos_rate),
        "holdout_pos_rate":   float(holdout_pos_rate),
    }
    with meta_file.open("w") as fh:
        json.dump(meta, fh, indent=2)

    _log.info("Split indices saved to '%s'.", split_dir)
    return train_idx, holdout_idx


def load_holdout_split(
    split_dir: Union[str, Path] = _SPLIT_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reload the holdout split indices created by :func:`make_holdout_split`.

    This is what every notebook and both tracks call after the initial
    ``00_setup.ipynb`` run.

    Args:
        split_dir: Directory containing ``train_idx.npy`` and
                   ``holdout_idx.npy`` (default ``data/splits/``).

    Returns:
        Tuple ``(train_idx, holdout_idx)`` of integer numpy arrays.

    Raises:
        FileNotFoundError: If split files have not been created yet.
    """
    split_dir    = Path(split_dir)
    train_file   = split_dir / "train_idx.npy"
    holdout_file = split_dir / "holdout_idx.npy"
    meta_file    = split_dir / "split_meta.json"

    for f in (train_file, holdout_file):
        if not f.exists():
            raise FileNotFoundError(
                f"Split file not found: '{f}'. "
                "Run make_holdout_split() in notebook 00_setup.ipynb first."
            )

    train_idx   = np.load(train_file)
    holdout_idx = np.load(holdout_file)

    if meta_file.exists():
        with meta_file.open() as fh:
            meta = json.load(fh)
        _log.info(
            "Loaded split — train: %d | holdout: %d (test_size=%.2f, seed=%d)",
            meta["n_train"], meta["n_holdout"],
            meta["test_size"], meta["random_state"],
        )
    else:
        _log.info(
            "Loaded split — train: %d | holdout: %d",
            len(train_idx), len(holdout_idx),
        )

    return train_idx, holdout_idx


# ---------------------------------------------------------------------------
# 3. CV fold generation — operates on train split ONLY
# ---------------------------------------------------------------------------

def get_cv_folds(
    train_idx: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield (fold_train_idx, fold_val_idx) pairs for stratified K-fold CV.

    Operates **only on the training partition** (indices from
    :func:`load_holdout_split`).  The holdout set is never touched here.

    The yielded indices are positions into the **original full DataFrame**,
    so callers can do ``df.iloc[fold_train_idx]`` directly.

    Args:
        train_idx:    Integer indices of the training partition (from
                      :func:`load_holdout_split`).
        labels:       Full label array for the entire dataset (all rows).
                      Only entries at *train_idx* positions are used for
                      stratification.
        n_splits:     Number of CV folds (default 5).
        random_state: RNG seed (default 42).

    Yields:
        ``(fold_train_idx, fold_val_idx)`` — both are integer arrays indexing
        into the full dataset DataFrame.

    Example::

        train_idx, holdout_idx = load_holdout_split()
        labels = df[COL_TARGET].to_numpy()

        for fold, (tr_idx, val_idx) in enumerate(get_cv_folds(train_idx, labels)):
            X_tr  = df.iloc[tr_idx][COL_FUNC].tolist()
            X_val = df.iloc[val_idx][COL_FUNC].tolist()
            y_tr  = labels[tr_idx]
            y_val = labels[val_idx]
            # fit TF-IDF on X_tr ONLY, then transform X_val
    """
    train_labels = labels[train_idx]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (local_tr, local_val) in enumerate(skf.split(train_idx, train_labels)):
        # local_tr / local_val are positions *within* train_idx, not the df
        fold_train_idx = train_idx[local_tr]
        fold_val_idx   = train_idx[local_val]

        fold_pos_rate_tr  = labels[fold_train_idx].mean()
        fold_pos_rate_val = labels[fold_val_idx].mean()

        _log.info(
            "Fold %d/%d — train: %d (%.1f%% vuln) | val: %d (%.1f%% vuln)",
            fold + 1, n_splits,
            len(fold_train_idx), 100 * fold_pos_rate_tr,
            len(fold_val_idx),   100 * fold_pos_rate_val,
        )

        yield fold_train_idx, fold_val_idx


# ---------------------------------------------------------------------------
# 4. Convenience extractor
# ---------------------------------------------------------------------------

def get_texts_and_labels(
    df: pd.DataFrame,
    idx: np.ndarray,
    *,
    text_col: str = COL_FUNC,
    label_col: str = COL_TARGET,
) -> Tuple[List[str], np.ndarray]:
    """Extract raw text strings and integer labels for a given index set.

    Args:
        df:        Full dataset DataFrame.
        idx:       Integer position array (e.g. from :func:`get_cv_folds`).
        text_col:  Column containing raw C/C++ source text.
        label_col: Column containing binary labels.

    Returns:
        Tuple ``(texts, labels)`` where *texts* is a list of strings and
        *labels* is an int64 numpy array.
    """
    subset  = df.iloc[idx]
    texts   = subset[text_col].tolist()
    labels  = subset[label_col].to_numpy(dtype=np.int64)
    return texts, labels


# ---------------------------------------------------------------------------
# 5. Config-aware convenience loader
# ---------------------------------------------------------------------------

def load_split_config(
    config_path: Union[str, Path] = "configs/track_a.yaml",
) -> dict:
    """Read CV / split parameters from a track config file.

    Returns a plain dict with keys ``n_splits``, ``holdout_size``,
    ``random_state`` — identical across both track configs.

    Args:
        config_path: Path to any track config YAML.

    Returns:
        Dict of split parameters.
    """
    cfg = load_config(config_path)
    return {
        "n_splits":     cfg.cv.n_splits,
        "test_size":    cfg.cv.test_size,
        "random_state": cfg.cv.random_state,
    }