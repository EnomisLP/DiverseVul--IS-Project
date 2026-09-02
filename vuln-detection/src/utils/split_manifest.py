"""
Project-grouped, stratified rotating 5-fold split manifest for all experiments.

Every experiment reuses the SAME manifest: one StratifiedGroupKFold(n_splits=5)
over the whole dataset, grouped by project, stratified by label. Each row gets
a fold in {0..4}; there is no separate fixed holdout. Every fold takes a turn
as the test set (the other 4 are train), and final metrics are the mean and
standard deviation across the 5 outer folds. This module prevents project
overlap across folds; it does not silently remove duplicates or relabel samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


MANIFEST_VERSION = "project-grouped-rotating-5fold-v2"
DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42

REQUIRED_COLUMNS = ("source_row_id", "label", "project")


@dataclass(frozen=True)
class SplitConfig:
    """Configuration used to create the rotating fold manifest."""

    n_splits: int = DEFAULT_N_SPLITS
    random_state: int = DEFAULT_RANDOM_STATE
    shuffle: bool = True
    source_id_column: str = "source_row_id"
    label_column: str = "label"
    group_column: str = "project"


@dataclass(frozen=True)
class ManifestPaths:
    """Locations of the artifacts produced by ``save_manifest_artifacts``."""

    csv_path: Path
    parquet_path: Path
    summary_path: Path
    metadata_path: Path


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error listing any missing required column."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {sorted(frame.columns.tolist())}"
        )


def _prepare_split_frame(
    frame: pd.DataFrame,
    config: SplitConfig,
) -> pd.DataFrame:
    """
    Validate and prepare the minimal dataframe required by the splitter.

    The returned dataframe preserves original row order. This is important
    because scikit-learn split indices are positional.
    """
    _require_columns(
        frame,
        (
            config.source_id_column,
            config.label_column,
            config.group_column,
        ),
    )

    split_frame = frame[
        [
            config.source_id_column,
            config.label_column,
            config.group_column,
        ]
    ].copy()

    split_frame = split_frame.rename(
        columns={
            config.source_id_column: "source_row_id",
            config.label_column: "label",
            config.group_column: "project",
        }
    )

    if split_frame["source_row_id"].isna().any():
        raise ValueError("source_row_id contains missing values.")

    if split_frame["source_row_id"].duplicated().any():
        duplicate_count = int(split_frame["source_row_id"].duplicated().sum())
        raise ValueError(
            "source_row_id must uniquely identify every sample. "
            f"Found {duplicate_count:,} duplicate identifier(s)."
        )

    if split_frame["label"].isna().any():
        raise ValueError("label contains missing values.")

    numeric_labels = pd.to_numeric(split_frame["label"], errors="raise")
    if not numeric_labels.isin([0, 1]).all():
        observed = sorted(pd.unique(numeric_labels).tolist())
        raise ValueError(
            f"Requires binary labels 0 and 1. Observed label values: {observed[:20]}"
        )
    split_frame["label"] = numeric_labels.astype("int8")

    if split_frame["project"].isna().any():
        raise ValueError(
            "project contains missing values. Project-aware CV requires a valid "
            "group identifier for every sample."
        )

    split_frame["project"] = split_frame["project"].astype(str).str.strip()
    if (split_frame["project"] == "").any():
        raise ValueError("project contains empty-string group identifiers.")

    if split_frame["label"].nunique() != 2:
        raise ValueError(
            "Both classes must be present before creating cross-validation folds."
        )

    if split_frame["project"].nunique() < config.n_splits:
        raise ValueError(
            f"At least {config.n_splits} unique projects are required for "
            f"{config.n_splits}-fold grouped cross-validation; found "
            f"{split_frame['project'].nunique()}."
        )

    return split_frame


def dataset_split_fingerprint(
    frame: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
) -> str:
    """
    Create a deterministic fingerprint for the row IDs, labels, and projects.
    """
    split_frame = _prepare_split_frame(frame, config)

    stable = split_frame.sort_values("source_row_id", kind="stable").reset_index(
        drop=True
    )
    row_hashes = pd.util.hash_pandas_object(
        stable[["source_row_id", "label", "project"]],
        index=False,
    ).values

    return sha256(row_hashes.tobytes()).hexdigest()


def create_project_grouped_manifest(
    frame: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
) -> pd.DataFrame:
    """Assign every row to one of n_splits rotating test folds via StratifiedGroupKFold."""
    split_frame = _prepare_split_frame(frame, config)

    splitter = StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=config.shuffle,
        random_state=config.random_state if config.shuffle else None,
    )

    fold_assignment = np.full(shape=len(split_frame), fill_value=-1, dtype=np.int16)
    dummy_x = np.zeros(shape=(len(split_frame), 1), dtype=np.uint8)

    for fold_id, (_, test_positions) in enumerate(
        splitter.split(
            X=dummy_x,
            y=split_frame["label"].to_numpy(),
            groups=split_frame["project"].to_numpy(),
        )
    ):
        if np.any(fold_assignment[test_positions] != -1):
            raise RuntimeError(
                "A sample was assigned to multiple test folds. This should never happen."
            )
        fold_assignment[test_positions] = fold_id

    if np.any(fold_assignment == -1):
        unassigned = int(np.sum(fold_assignment == -1))
        raise RuntimeError(
            f"{unassigned:,} sample(s) were not assigned to any test fold."
        )

    manifest = split_frame.copy()
    manifest["fold"] = fold_assignment.astype(int)

    assert_manifest_integrity(manifest, config=config)
    return manifest


def summarize_manifest(
    manifest: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
) -> pd.DataFrame:
    """Return fold-level test statistics and class-prevalence diagnostics."""
    _require_columns(manifest, ("source_row_id", "label", "project", "fold"))

    global_positive_rate = float(manifest["label"].mean())
    total_rows = len(manifest)
    total_projects = manifest["project"].nunique()

    rows: list[dict] = []
    for fold_id in sorted(manifest["fold"].unique().tolist()):
        test_frame = manifest.loc[manifest["fold"] == fold_id]
        train_frame = manifest.loc[manifest["fold"] != fold_id]

        test_projects = set(test_frame["project"])
        train_projects = set(train_frame["project"])
        overlap_projects = test_projects.intersection(train_projects)

        rows.append(
            {
                "fold": int(fold_id),
                "test_rows": int(len(test_frame)),
                "test_vulnerable": int((test_frame["label"] == 1).sum()),
                "test_non_vulnerable": int((test_frame["label"] == 0).sum()),
                "test_positive_rate": float(test_frame["label"].mean()),
                "positive_rate_delta_from_global": float(
                    test_frame["label"].mean() - global_positive_rate
                ),
                "test_unique_projects": int(test_frame["project"].nunique()),
                "train_rows": int(len(train_frame)),
                "train_unique_projects": int(train_frame["project"].nunique()),
                "train_test_project_overlap": int(len(overlap_projects)),
                "test_row_share": float(len(test_frame) / total_rows),
                "test_project_share": float(test_frame["project"].nunique() / total_projects),
            }
        )

    summary = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    return summary


def assert_manifest_integrity(
    manifest: pd.DataFrame,
    config: SplitConfig = SplitConfig(),
) -> None:
    """Raise an error if a manifest violates the project-grouped split rules."""
    _require_columns(manifest, ("source_row_id", "label", "project", "fold"))

    if manifest.empty:
        raise ValueError("Manifest is empty.")

    if manifest["source_row_id"].isna().any():
        raise ValueError("Manifest contains missing source_row_id values.")

    if manifest["source_row_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate source_row_id values.")

    labels = pd.to_numeric(manifest["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Manifest labels must be exactly 0 or 1.")

    folds = pd.to_numeric(manifest["fold"], errors="raise")
    expected_folds = set(range(config.n_splits))
    actual_folds = set(folds.astype(int).unique().tolist())
    if actual_folds != expected_folds:
        raise ValueError(
            f"Manifest fold IDs must be {sorted(expected_folds)}; "
            f"found {sorted(actual_folds)}."
        )

    if (folds < 0).any() or (folds >= config.n_splits).any():
        raise ValueError("Manifest includes an invalid fold ID.")

    for fold_id in range(config.n_splits):
        test_projects = set(manifest.loc[manifest["fold"] == fold_id, "project"])
        train_projects = set(manifest.loc[manifest["fold"] != fold_id, "project"])

        overlap = test_projects.intersection(train_projects)
        if overlap:
            sample = sorted(overlap)[:10]
            raise ValueError(
                f"Project leakage detected in fold {fold_id}. "
                f"Example overlapping project IDs: {sample}"
            )

        test_labels = set(
            manifest.loc[manifest["fold"] == fold_id, "label"].astype(int).tolist()
        )
        if test_labels != {0, 1}:
            raise ValueError(
                f"Fold {fold_id} does not contain both classes. "
                f"Observed test labels: {sorted(test_labels)}"
            )


def save_manifest_artifacts(
    manifest: pd.DataFrame,
    output_dir: Path | str,
    config: SplitConfig = SplitConfig(),
    dataset_fingerprint: Optional[str] = None,
    normalized_dataset_path: Optional[Path | str] = None,
) -> ManifestPaths:
    """Save the reusable manifest and its audit artifacts."""
    assert_manifest_integrity(manifest, config=config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"project_grouped_{config.n_splits}fold"
    csv_path = output_dir / f"{stem}_manifest.csv"
    parquet_path = output_dir / f"{stem}_manifest.parquet"
    summary_path = output_dir / f"{stem}_fold_summary.csv"
    metadata_path = output_dir / f"{stem}_metadata.json"

    manifest_to_save = manifest.sort_values("source_row_id", kind="stable").reset_index(
        drop=True
    )
    summary = summarize_manifest(manifest_to_save, config=config)

    manifest_to_save.to_csv(csv_path, index=False)
    manifest_to_save.to_parquet(parquet_path, index=False)
    summary.to_csv(summary_path, index=False)

    if dataset_fingerprint is None:
        dataset_fingerprint = dataset_split_fingerprint(manifest_to_save, config=config)

    metadata = {
        "manifest_version": MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "splitter": "StratifiedGroupKFold",
        "config": asdict(config),
        "dataset_split_fingerprint": dataset_fingerprint,
        "normalized_dataset_path": (
            str(normalized_dataset_path) if normalized_dataset_path is not None else None
        ),
        "rows": int(len(manifest_to_save)),
        "vulnerable_rows": int((manifest_to_save["label"] == 1).sum()),
        "non_vulnerable_rows": int((manifest_to_save["label"] == 0).sum()),
        "global_positive_rate": float(manifest_to_save["label"].mean()),
        "unique_projects": int(manifest_to_save["project"].nunique()),
        "integrity_checks": {
            "all_rows_assigned_once": True,
            "project_overlap_between_train_and_test": False,
            "both_classes_present_in_each_test_fold": True,
        },
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return ManifestPaths(
        csv_path=csv_path,
        parquet_path=parquet_path,
        summary_path=summary_path,
        metadata_path=metadata_path,
    )


def load_manifest(
    manifest_path: Path | str,
    config: SplitConfig = SplitConfig(),
) -> pd.DataFrame:
    """Load a previously created manifest and verify its integrity."""
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    if manifest_path.suffix.lower() == ".parquet":
        manifest = pd.read_parquet(manifest_path)
    elif manifest_path.suffix.lower() == ".csv":
        manifest = pd.read_csv(manifest_path)
    else:
        raise ValueError(
            "Manifest must be a .csv or .parquet file; "
            f"received: {manifest_path.suffix}"
        )

    assert_manifest_integrity(manifest, config=config)
    return manifest


def apply_manifest(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    source_id_column: str = "source_row_id",
) -> pd.DataFrame:
    """Attach the fold assignment to a dataframe by merging on source_row_id."""
    _require_columns(frame, (source_id_column,))
    _require_columns(manifest, ("source_row_id", "fold", "label", "project"))

    source = frame.copy()
    if source_id_column != "source_row_id":
        source = source.rename(columns={source_id_column: "source_row_id"})

    if source["source_row_id"].duplicated().any():
        raise ValueError("Input frame has duplicate source_row_id values.")

    manifest_columns = ["source_row_id", "fold", "label", "project"]
    merged = source.merge(
        manifest[manifest_columns],
        on="source_row_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_manifest"),
    )

    if merged["fold"].isna().any():
        missing = int(merged["fold"].isna().sum())
        raise ValueError(
            f"{missing:,} source row(s) have no fold assignment in the manifest. "
            "Use the same normalized dataset that generated the manifest."
        )

    if "label_manifest" in merged.columns:
        mismatch = merged["label"].astype(int) != merged["label_manifest"].astype(int)
        if mismatch.any():
            raise ValueError(
                "Label mismatch between input dataframe and manifest. "
                "Do not reuse the manifest with modified labels."
            )
        merged = merged.drop(columns=["label_manifest"])

    if "project_manifest" in merged.columns:
        mismatch = (
            merged["project"].astype(str).str.strip()
            != merged["project_manifest"].astype(str).str.strip()
        )
        if mismatch.any():
            raise ValueError(
                "Project mismatch between input dataframe and manifest. "
                "Do not reuse the manifest with modified project IDs."
            )
        merged = merged.drop(columns=["project_manifest"])

    merged["fold"] = merged["fold"].astype(int)
    return merged


__all__ = [
    "DEFAULT_N_SPLITS",
    "DEFAULT_RANDOM_STATE",
    "MANIFEST_VERSION",
    "ManifestPaths",
    "SplitConfig",
    "apply_manifest",
    "assert_manifest_integrity",
    "create_project_grouped_manifest",
    "dataset_split_fingerprint",
    "load_manifest",
    "save_manifest_artifacts",
    "summarize_manifest",
]