"""
CS1-EXP2-MLP: train-only TF-IDF + TruncatedSVD + static features + shallow MLP.

This module implements the third Case Study 1 candidate after:
    CS1-EXP0-LR  : word/character TF-IDF + SGD Logistic Regression
    CS1-EXP1-RF  : TF-IDF + TruncatedSVD + static features + Random Forest

CS1-EXP2-MLP follows the approved shallow neural-network design:
    310 dense inputs (256 SVD + 54 deterministic static features)
        -> Linear(128) + ReLU
        -> Linear(64) + ReLU
        -> Linear(1) logits

Important evaluation protocol
-----------------------------
- The frozen outer 20% project holdout remains unused by ``run_exp2`` and
  ``run_exp2_profile_fold``.
- The same frozen 5-fold project-grouped development manifest as EXP-0 and
  EXP-1 is reused unchanged.
- In every outer development fold, an additional project-disjoint validation
  partition is created only from that fold's training projects.
- TF-IDF, SVD, static-feature scaling, class weighting, early stopping, and
  threshold selection are all fit or selected without access to the outer
  held-out project fold.
- The MLP uses ``BCEWithLogitsLoss(pos_weight=negative/positive)`` calculated
  from the inner optimisation partition only. Scores are therefore risk/ranking
  scores, not calibrated real-world vulnerability probabilities.

The final-holdout function is intentionally separate and must be called only
once after all Case Study 1 candidates have been compared on development OOF
results.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import time
from typing import Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "CS1-EXP2-MLP requires PyTorch. Install a compatible torch build "
        "before importing this module."
    ) from exc

from ..evaluation import (
    EvaluationConfig,
    compute_binary_metrics,
    evaluate_oof_predictions,
    save_evaluation_artifacts,
)
from ..split_manifest import (
    SplitConfig,
    apply_manifest,
    assert_manifest_integrity,
)
from ..exp1.static_features import FEATURE_COLUMNS


EXP2_VERSION = "cs1-exp2-svd-static-mlp-v1-inner-val-holdout"


@dataclass(frozen=True)
class Exp2Config:
    """Declared reproducible configuration for CS1-EXP2-MLP."""

    experiment_name: str = "cs1_exp2_mlp"
    code_column: str = "normalized_code"
    source_id_column: str = "source_row_id"
    label_column: str = "label"
    project_column: str = "project"
    fold_column: str = "fold"

    # Frozen outer development-CV manifest configuration.
    n_splits: int = 5
    random_state: int = 42
    decision_threshold: float = 0.50

    # Same lexical representation used by EXP-0 and EXP-1.
    word_ngram_range: tuple[int, int] = (1, 3)
    word_min_df: int = 3
    word_max_df: float = 0.995
    word_max_features: int = 50_000

    char_analyzer: str = "char"
    char_ngram_range: tuple[int, int] = (3, 4)
    char_min_df: int = 8
    char_max_df: float = 0.995
    char_max_features: int = 60_000

    lowercase: bool = False
    sublinear_tf: bool = True
    tfidf_norm: str = "l2"

    # TruncatedSVD is fitted only on the inner optimisation partition.
    svd_n_components: int = 256
    svd_algorithm: str = "randomized"
    svd_n_iter: int = 5
    svd_n_oversamples: int = 10

    # Inner project-disjoint validation for early stopping and threshold choice.
    inner_validation_n_splits: int = 5
    inner_validation_seed_start: int = 10_000
    inner_validation_seed_count: int = 64
    inner_validation_min_row_share: float = 0.08
    inner_validation_max_row_share: float = 0.35

    # Proposal-aligned 3-layer MLP: 128 -> 64 -> 1.
    hidden_dim_1: int = 128
    hidden_dim_2: int = 64
    max_epochs: int = 15
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    batch_size_gpu: int = 2_048
    batch_size_cpu: int = 1_024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Class imbalance is handled inside BCEWithLogitsLoss using neg/pos from
    # the inner optimisation partition only.
    use_pos_weight: bool = True

    # Threshold is selected only from inner validation scores using a fixed grid.
    validation_threshold_min: float = 0.01
    validation_threshold_max: float = 0.99
    validation_threshold_step: float = 0.01
    validation_threshold_objective: str = "f1"

    # ``auto`` uses CUDA when available, otherwise CPU.  The sklearn TF-IDF
    # and SVD stages are CPU operations; the MLP itself uses this device.
    device: str = "auto"
    dataloader_num_workers: int = 0
    verbose: bool = True


@dataclass(frozen=True)
class Exp2Artifacts:
    """Artifacts generated by a complete official CS1-EXP2-MLP run."""

    config_json: Path
    fold_training_csv: Path
    training_history_csv: Path
    validation_operating_metrics_json: Path
    validation_operating_fold_metrics_csv: Path
    validation_operating_fold_summary_csv: Path
    run_metadata_json: Path
    evaluation_paths: object


class ShallowMLP(nn.Module):
    """Proposal-aligned feed-forward binary classifier returning logits."""

    def __init__(self, input_dim: int, hidden_dim_1: int, hidden_dim_2: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(dim=1)


def _log(message: str, enabled: bool = True) -> None:
    if enabled:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)


def _format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    return f"{seconds:.1f}s" if seconds < 60 else f"{seconds / 60:.2f} min"


def _fold_label(fold_id: int, config: Exp2Config) -> str:
    """Return a readable label for development folds and final-holdout work."""
    if int(fold_id) < 0:
        return "FINAL HOLDOUT"
    return f"Fold {int(fold_id) + 1}/{config.n_splits}"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {sorted(frame.columns.tolist())}"
        )


def _validate_config(config: Exp2Config) -> None:
    if config.n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if not 0.0 < float(config.decision_threshold) < 1.0:
        raise ValueError("decision_threshold must lie strictly between 0 and 1.")
    if config.word_min_df < 1 or config.char_min_df < 1:
        raise ValueError("word_min_df and char_min_df must be >= 1.")
    if config.word_max_features < 1 or config.char_max_features < 1:
        raise ValueError("word_max_features and char_max_features must be >= 1.")
    if config.svd_n_components < 2:
        raise ValueError("svd_n_components must be >= 2.")
    if config.svd_algorithm != "randomized":
        raise ValueError("This experiment is declared for randomized TruncatedSVD.")
    if config.svd_n_iter < 1 or config.svd_n_oversamples < 1:
        raise ValueError("SVD iteration and oversampling values must be >= 1.")
    if config.inner_validation_n_splits < 2:
        raise ValueError("inner_validation_n_splits must be >= 2.")
    if config.inner_validation_seed_count < 1:
        raise ValueError("inner_validation_seed_count must be >= 1.")
    if not 0.0 < config.inner_validation_min_row_share < 1.0:
        raise ValueError("inner_validation_min_row_share must lie in (0, 1).")
    if not 0.0 < config.inner_validation_max_row_share < 1.0:
        raise ValueError("inner_validation_max_row_share must lie in (0, 1).")
    if config.inner_validation_min_row_share >= config.inner_validation_max_row_share:
        raise ValueError("Inner-validation min row share must be smaller than max.")
    if config.hidden_dim_1 < 1 or config.hidden_dim_2 < 1:
        raise ValueError("MLP hidden dimensions must be positive.")
    if config.max_epochs < 1:
        raise ValueError("max_epochs must be >= 1.")
    if config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be >= 1.")
    if config.batch_size_gpu < 1 or config.batch_size_cpu < 1:
        raise ValueError("Batch sizes must be >= 1.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be >= 0.")
    if not (
        0.0
        < config.validation_threshold_min
        < config.validation_threshold_max
        < 1.0
    ):
        raise ValueError("Invalid validation threshold range.")
    if config.validation_threshold_step <= 0.0:
        raise ValueError("validation_threshold_step must be positive.")
    if config.validation_threshold_objective != "f1":
        raise ValueError("Only validation_threshold_objective='f1' is supported.")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: 'auto', 'cpu', 'cuda'.")
    if config.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be >= 0.")


def _validate_dataset_with_folds(
    frame: pd.DataFrame,
    config: Exp2Config,
) -> pd.DataFrame:
    _require_columns(
        frame,
        [
            config.source_id_column,
            config.code_column,
            config.label_column,
            config.project_column,
            config.fold_column,
        ],
    )

    checked = frame.copy()
    if checked[config.source_id_column].isna().any():
        raise ValueError("source_row_id contains missing values.")
    if checked[config.source_id_column].duplicated().any():
        raise ValueError("source_row_id contains duplicate values.")

    checked[config.code_column] = checked[config.code_column].fillna("").astype(str)
    if checked[config.code_column].str.strip().eq("").any():
        raise ValueError(
            "EXP-2 received empty normalized_code rows. Use the normalized "
            "dataset that passed the Case Study 1 integrity checks."
        )

    labels = pd.to_numeric(checked[config.label_column], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("label must contain only 0 and 1.")
    checked[config.label_column] = labels.astype("int8")

    checked[config.project_column] = (
        checked[config.project_column].fillna("").astype(str).str.strip()
    )
    if checked[config.project_column].eq("").any():
        raise ValueError("project contains missing or empty identifiers.")

    folds = pd.to_numeric(checked[config.fold_column], errors="raise").astype(int)
    expected = set(range(config.n_splits))
    observed = set(folds.unique().tolist())
    if observed != expected:
        raise ValueError(
            f"Expected folds {sorted(expected)}; observed {sorted(observed)}."
        )
    checked[config.fold_column] = folds.astype("int8")
    return checked


def _validate_static_feature_frame(
    static_features_frame: pd.DataFrame,
    config: Exp2Config,
) -> pd.DataFrame:
    required = [config.source_id_column, *FEATURE_COLUMNS]
    _require_columns(static_features_frame, required)

    static_checked = static_features_frame[required].copy()
    if static_checked[config.source_id_column].isna().any():
        raise ValueError("Static feature cache has missing source_row_id values.")
    if static_checked[config.source_id_column].duplicated().any():
        raise ValueError("Static feature cache has duplicate source_row_id values.")

    for column in FEATURE_COLUMNS:
        static_checked[column] = pd.to_numeric(
            static_checked[column], errors="raise"
        ).astype(np.float32)

    values = static_checked[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Static feature cache contains NaN or infinite values.")
    if (values < 0).any():
        raise ValueError("Static feature cache contains negative proxy values.")

    return static_checked.set_index(config.source_id_column, drop=True)


def _prepare_dataset_and_static_lookup(
    normalized_frame: pd.DataFrame,
    static_features_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    config: Exp2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the frozen inner-CV manifest and validate static-cache alignment."""
    assert_manifest_integrity(
        manifest,
        config=SplitConfig(
            n_splits=config.n_splits,
            random_state=config.random_state,
            source_id_column=config.source_id_column,
            label_column=config.label_column,
            group_column=config.project_column,
        ),
    )

    dataset = apply_manifest(
        normalized_frame,
        manifest,
        source_id_column=config.source_id_column,
    )
    dataset = _validate_dataset_with_folds(dataset, config)
    static_lookup = _validate_static_feature_frame(static_features_frame, config)

    dataset_ids = pd.Index(dataset[config.source_id_column])
    missing_static = dataset_ids.difference(static_lookup.index)
    if len(missing_static):
        raise ValueError(
            "Static feature cache is missing IDs required by the development "
            f"frame: {len(missing_static):,}. The cache may legitimately include "
            "extra IDs from the sealed outer holdout."
        )

    for fold_id in range(config.n_splits):
        held_out_projects = set(
            dataset.loc[
                dataset[config.fold_column] == fold_id,
                config.project_column,
            ]
        )
        train_projects = set(
            dataset.loc[
                dataset[config.fold_column] != fold_id,
                config.project_column,
            ]
        )
        overlap = held_out_projects.intersection(train_projects)
        if overlap:
            raise RuntimeError(
                f"Project leakage detected in outer fold {fold_id}; "
                f"examples: {sorted(overlap)[:10]}"
            )

    return dataset, static_lookup


def _build_word_vectorizer(config: Exp2Config) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=config.word_ngram_range,
        min_df=config.word_min_df,
        max_df=config.word_max_df,
        max_features=config.word_max_features,
        lowercase=config.lowercase,
        sublinear_tf=config.sublinear_tf,
        norm=config.tfidf_norm,
        dtype=np.float32,
    )


def _build_char_vectorizer(config: Exp2Config) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=config.char_analyzer,
        ngram_range=config.char_ngram_range,
        min_df=config.char_min_df,
        max_df=config.char_max_df,
        max_features=config.char_max_features,
        lowercase=config.lowercase,
        sublinear_tf=config.sublinear_tf,
        norm=config.tfidf_norm,
        dtype=np.float32,
    )


def _fit_lexical_matrices(
    fit_code: pd.Series,
    validation_code: pd.Series,
    test_code: pd.Series,
    config: Exp2Config,
    fold_id: int,
):
    """Fit lexical TF-IDF only on inner-fit code and transform val/test code."""
    label = _fold_label(fold_id, config)
    word_vectorizer = _build_word_vectorizer(config)
    char_vectorizer = _build_char_vectorizer(config)

    _log(f"{label} | fitting inner-train word TF-IDF...", config.verbose)
    started = time.perf_counter()
    x_fit_word = word_vectorizer.fit_transform(fit_code)
    x_validation_word = word_vectorizer.transform(validation_code)
    x_test_word = word_vectorizer.transform(test_code)
    word_seconds = float(time.perf_counter() - started)
    _log(
        f"{label} | word TF-IDF done in {_format_seconds(word_seconds)} "
        f"({x_fit_word.shape[1]:,} features).",
        config.verbose,
    )

    _log(f"{label} | fitting inner-train character TF-IDF...", config.verbose)
    started = time.perf_counter()
    x_fit_char = char_vectorizer.fit_transform(fit_code)
    x_validation_char = char_vectorizer.transform(validation_code)
    x_test_char = char_vectorizer.transform(test_code)
    char_seconds = float(time.perf_counter() - started)
    _log(
        f"{label} | character TF-IDF done in {_format_seconds(char_seconds)} "
        f"({x_fit_char.shape[1]:,} features).",
        config.verbose,
    )

    _log(f"{label} | joining sparse lexical matrices...", config.verbose)
    started = time.perf_counter()
    x_fit_lexical = hstack([x_fit_word, x_fit_char], format="csr", dtype=np.float32)
    x_validation_lexical = hstack(
        [x_validation_word, x_validation_char], format="csr", dtype=np.float32
    )
    x_test_lexical = hstack(
        [x_test_word, x_test_char], format="csr", dtype=np.float32
    )
    sparse_join_seconds = float(time.perf_counter() - started)

    metadata = {
        "word_features": int(x_fit_word.shape[1]),
        "char_features": int(x_fit_char.shape[1]),
        "lexical_features": int(x_fit_lexical.shape[1]),
        "word_tfidf_seconds": word_seconds,
        "char_tfidf_seconds": char_seconds,
        "sparse_join_seconds": sparse_join_seconds,
        "vectorization_seconds": float(word_seconds + char_seconds + sparse_join_seconds),
    }

    return (
        x_fit_lexical,
        x_validation_lexical,
        x_test_lexical,
        word_vectorizer,
        char_vectorizer,
        metadata,
    )


def _fit_and_transform_svd(
    x_fit_lexical,
    x_validation_lexical,
    x_test_lexical,
    config: Exp2Config,
    fold_id: int,
):
    """Fit SVD only on inner-fit sparse lexical features."""
    if config.svd_n_components >= x_fit_lexical.shape[1]:
        raise ValueError(
            "svd_n_components must be lower than lexical feature count; "
            f"received {config.svd_n_components} for "
            f"{x_fit_lexical.shape[1]} lexical features."
        )

    label = _fold_label(fold_id, config)
    svd = TruncatedSVD(
        n_components=config.svd_n_components,
        algorithm=config.svd_algorithm,
        n_iter=config.svd_n_iter,
        n_oversamples=config.svd_n_oversamples,
        random_state=config.random_state + fold_id,
    )

    _log(
        f"{label} | fitting inner-train TruncatedSVD "
        f"({config.svd_n_components} components)...",
        config.verbose,
    )
    started = time.perf_counter()
    x_fit_svd = svd.fit_transform(x_fit_lexical)
    x_validation_svd = svd.transform(x_validation_lexical)
    x_test_svd = svd.transform(x_test_lexical)
    svd_seconds = float(time.perf_counter() - started)

    x_fit_svd = np.asarray(x_fit_svd, dtype=np.float32)
    x_validation_svd = np.asarray(x_validation_svd, dtype=np.float32)
    x_test_svd = np.asarray(x_test_svd, dtype=np.float32)
    explained_variance = float(np.sum(svd.explained_variance_ratio_))

    _log(
        f"{label} | TruncatedSVD done in {_format_seconds(svd_seconds)} "
        f"(explained variance ratio sum={explained_variance:.4f}).",
        config.verbose,
    )

    return (
        x_fit_svd,
        x_validation_svd,
        x_test_svd,
        svd,
        svd_seconds,
        explained_variance,
    )


def _get_static_matrix(
    static_lookup: pd.DataFrame,
    source_ids: pd.Series,
) -> np.ndarray:
    """Fetch static features in exactly the requested source-ID order."""
    matrix = static_lookup.loc[
        pd.Index(source_ids), FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32, copy=True)
    if matrix.shape[1] != len(FEATURE_COLUMNS):
        raise RuntimeError("Static feature matrix has an unexpected column count.")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Static feature matrix contains NaN or infinite values.")
    return matrix


def _fit_scaler_and_transform(
    x_fit: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, float]:
    """Fit StandardScaler only on inner-fit dense features."""
    started = time.perf_counter()
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    x_fit_scaled = scaler.fit_transform(x_fit).astype(np.float32, copy=False)
    x_validation_scaled = scaler.transform(x_validation).astype(np.float32, copy=False)
    x_test_scaled = scaler.transform(x_test).astype(np.float32, copy=False)
    scaler_seconds = float(time.perf_counter() - started)

    for name, matrix in (
        ("fit", x_fit_scaled),
        ("validation", x_validation_scaled),
        ("test", x_test_scaled),
    ):
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"Scaled {name} feature matrix contains NaN/inf.")

    return x_fit_scaled, x_validation_scaled, x_test_scaled, scaler, scaler_seconds


def _select_inner_validation_split(
    outer_train_frame: pd.DataFrame,
    config: Exp2Config,
    outer_fold_id: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Select one deterministic, project-disjoint inner validation split.

    Candidate splits are ranked only by row-share and class-prevalence balance;
    no candidate model scores are consulted.  This is safe because selection
    occurs solely inside the current outer-fold training partition.
    """
    labels = outer_train_frame[config.label_column].to_numpy(dtype=np.int8)
    groups = outer_train_frame[config.project_column].astype(str).to_numpy()
    total_rows = len(outer_train_frame)
    global_positive_rate = float(labels.mean())
    target_share = 1.0 / float(config.inner_validation_n_splits)

    candidates: list[tuple[tuple[float, float, float, int, int], np.ndarray, np.ndarray, dict]] = []

    for seed_offset in range(config.inner_validation_seed_count):
        split_seed = (
            config.inner_validation_seed_start
            + config.random_state
            + outer_fold_id * 10_000
            + seed_offset
        )
        splitter = StratifiedGroupKFold(
            n_splits=config.inner_validation_n_splits,
            shuffle=True,
            random_state=split_seed,
        )

        try:
            split_pairs = list(splitter.split(np.zeros(total_rows), labels, groups))
        except ValueError:
            continue

        for validation_fold_id, (fit_indices, validation_indices) in enumerate(split_pairs):
            y_fit = labels[fit_indices]
            y_validation = labels[validation_indices]
            if set(np.unique(y_fit).tolist()) != {0, 1}:
                continue
            if set(np.unique(y_validation).tolist()) != {0, 1}:
                continue

            fit_projects = set(groups[fit_indices])
            validation_projects = set(groups[validation_indices])
            if fit_projects.intersection(validation_projects):
                continue

            validation_share = float(len(validation_indices) / total_rows)
            if not (
                config.inner_validation_min_row_share
                <= validation_share
                <= config.inner_validation_max_row_share
            ):
                continue

            validation_positive_rate = float(y_validation.mean())
            fit_positive_rate = float(y_fit.mean())
            key = (
                abs(validation_share - target_share),
                abs(validation_positive_rate - global_positive_rate),
                abs(fit_positive_rate - global_positive_rate),
                int(split_seed),
                int(validation_fold_id),
            )
            metadata = {
                "inner_validation_split_seed": int(split_seed),
                "inner_validation_candidate_fold": int(validation_fold_id),
                "inner_fit_rows": int(len(fit_indices)),
                "inner_validation_rows": int(len(validation_indices)),
                "inner_validation_row_share": validation_share,
                "inner_fit_positive_rate": fit_positive_rate,
                "inner_validation_positive_rate": validation_positive_rate,
                "inner_fit_projects": int(len(fit_projects)),
                "inner_validation_projects": int(len(validation_projects)),
                "inner_train_validation_project_overlap": 0,
            }
            candidates.append((key, fit_indices, validation_indices, metadata))

    if not candidates:
        raise RuntimeError(
            "Could not create a valid inner project-disjoint validation split "
            f"for outer fold {outer_fold_id}. Consider widening only the "
            "predeclared row-share constraints, then rerun the MLP profile."
        )

    candidates.sort(key=lambda item: item[0])
    _, fit_indices, validation_indices, metadata = candidates[0]
    metadata["inner_validation_candidates_considered"] = int(len(candidates))
    return fit_indices, validation_indices, metadata


def _resolve_device(config: Exp2Config) -> torch.device:
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but PyTorch cannot access a CUDA GPU."
            )
        return torch.device("cuda")
    if config.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_loader(
    features: np.ndarray,
    labels: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    config: Exp2Config,
) -> DataLoader:
    feature_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    if labels is None:
        dataset = TensorDataset(feature_tensor)
    else:
        label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        dataset = TensorDataset(feature_tensor, label_tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=config.dataloader_num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def _predict_scores(
    model: ShallowMLP,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
    config: Exp2Config,
) -> np.ndarray:
    loader = _build_loader(
        features=features,
        labels=None,
        batch_size=batch_size,
        shuffle=False,
        seed=config.random_state,
        device=device,
        config=config,
    )
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_features,) in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            logits = model(batch_features)
            probabilities.append(
                torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
            )
    scores = np.concatenate(probabilities, axis=0)
    if not np.isfinite(scores).all() or (scores < 0.0).any() or (scores > 1.0).any():
        raise RuntimeError("MLP produced invalid probability scores.")
    return scores


def _weighted_loss_on_features(
    model: ShallowMLP,
    features: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    config: Exp2Config,
) -> float:
    loader = _build_loader(
        features=features,
        labels=labels,
        batch_size=batch_size,
        shuffle=False,
        seed=config.random_state,
        device=device,
        config=config,
    )
    model.eval()
    loss_sum = 0.0
    n_rows = 0
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_loss = criterion(model(batch_features), batch_labels)
            rows = int(batch_labels.shape[0])
            loss_sum += float(batch_loss.item()) * rows
            n_rows += rows
    return float(loss_sum / n_rows) if n_rows else float("nan")


def _binary_metrics_from_decisions(
    labels: Sequence[int] | np.ndarray | pd.Series,
    scores: Sequence[float] | np.ndarray | pd.Series,
    decisions: Sequence[int] | np.ndarray | pd.Series,
    threshold: Optional[float],
) -> dict:
    """Compute ranking and operating metrics from supplied binary decisions."""
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=float)
    y_pred = np.asarray(decisions, dtype=np.int8)

    if len(y_true) == 0 or not (len(y_true) == len(y_score) == len(y_pred)):
        raise ValueError("labels, scores, and decisions must be aligned and non-empty.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("labels and decisions must be binary.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    npv_denominator = tn + fn

    return {
        "n_samples": int(len(y_true)),
        "vulnerable_1": int((y_true == 1).sum()),
        "non_vulnerable_0": int((y_true == 0).sum()),
        "positive_rate": float(y_true.mean()),
        "threshold": None if threshold is None else float(threshold),
        "average_precision_pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "negative_predictive_value": float(tn / npv_denominator) if npv_denominator else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_positive": int(y_pred.sum()),
        "predicted_positive_rate": float(y_pred.mean()),
    }


def _validation_threshold_candidates(config: Exp2Config) -> np.ndarray:
    grid = np.arange(
        config.validation_threshold_min,
        config.validation_threshold_max + config.validation_threshold_step / 2.0,
        config.validation_threshold_step,
        dtype=float,
    )
    return np.clip(grid, 0.0, 1.0)


def _select_validation_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    config: Exp2Config,
) -> tuple[float, dict]:
    """Select maximum-F1 threshold using only inner-validation predictions."""
    if not np.isfinite(scores).all() or (scores < 0.0).any() or (scores > 1.0).any():
        raise ValueError("Validation scores must be finite probabilities in [0, 1].")

    candidates = _validation_threshold_candidates(config)
    best: Optional[tuple[tuple[float, float, int], dict]] = None
    for threshold in candidates:
        decisions = (scores >= threshold).astype(np.int8)
        metrics = _binary_metrics_from_decisions(
            labels=labels,
            scores=scores,
            decisions=decisions,
            threshold=float(threshold),
        )
        # Stable tie-break: better F1, then better precision, then fewer alerts.
        key = (metrics["f1"], metrics["precision"], -metrics["predicted_positive"])
        if best is None or key > best[0]:
            best = (key, metrics)

    assert best is not None
    selected = dict(best[1])
    selected["selection_source"] = "inner_project_validation_predictions"
    selected["selection_objective"] = config.validation_threshold_objective
    selected["candidate_count"] = int(len(candidates))
    selected["candidate_min"] = float(candidates.min())
    selected["candidate_max"] = float(candidates.max())
    selected["selected_at_grid_boundary"] = bool(
        np.isclose(selected["threshold"], candidates.min())
        or np.isclose(selected["threshold"], candidates.max())
    )
    return float(selected["threshold"]), selected


def _train_mlp_with_early_stopping(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: Exp2Config,
    fold_seed: int,
) -> tuple[ShallowMLP, pd.DataFrame, dict, np.ndarray, torch.device]:
    """Train the MLP on inner-fit rows and early-stop by validation PR-AUC."""
    if set(np.unique(y_fit).tolist()) != {0, 1}:
        raise ValueError("Inner optimisation partition must contain both classes.")
    if set(np.unique(y_validation).tolist()) != {0, 1}:
        raise ValueError("Inner validation partition must contain both classes.")

    _set_reproducible_seed(fold_seed)
    device = _resolve_device(config)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    batch_size = config.batch_size_gpu if device.type == "cuda" else config.batch_size_cpu
    model = ShallowMLP(
        input_dim=int(x_fit.shape[1]),
        hidden_dim_1=config.hidden_dim_1,
        hidden_dim_2=config.hidden_dim_2,
    ).to(device)

    negative_count = int((y_fit == 0).sum())
    positive_count = int((y_fit == 1).sum())
    pos_weight_value = (
        float(negative_count / positive_count) if config.use_pos_weight else 1.0
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    fit_loader = _build_loader(
        features=x_fit,
        labels=y_fit,
        batch_size=batch_size,
        shuffle=True,
        seed=fold_seed,
        device=device,
        config=config,
    )

    history_rows: list[dict] = []
    best_state = deepcopy(model.state_dict())
    best_validation_pr_auc = float("-inf")
    best_validation_loss = float("inf")
    best_epoch = 0
    no_improvement_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_row_count = 0

        for batch_features, batch_labels in fit_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            rows = int(batch_labels.shape[0])
            train_loss_sum += float(loss.item()) * rows
            train_row_count += rows

        train_loss = float(train_loss_sum / train_row_count)
        validation_scores = _predict_scores(
            model=model,
            features=x_validation,
            device=device,
            batch_size=batch_size,
            config=config,
        )
        validation_pr_auc = float(average_precision_score(y_validation, validation_scores))
        validation_loss = _weighted_loss_on_features(
            model=model,
            features=x_validation,
            labels=y_validation,
            criterion=criterion,
            device=device,
            batch_size=batch_size,
            config=config,
        )

        improved = (
            validation_pr_auc > best_validation_pr_auc + config.early_stopping_min_delta
            or (
                abs(validation_pr_auc - best_validation_pr_auc)
                <= config.early_stopping_min_delta
                and validation_loss < best_validation_loss
            )
        )

        if improved:
            best_validation_pr_auc = validation_pr_auc
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        history_rows.append(
            {
                "epoch": int(epoch),
                "train_weighted_bce_loss": train_loss,
                "validation_weighted_bce_loss": validation_loss,
                "validation_pr_auc": validation_pr_auc,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "is_best_epoch": bool(improved),
            }
        )

        _log(
            f"Epoch {epoch:02d}/{config.max_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={validation_loss:.4f} | "
            f"val_PR-AUC={validation_pr_auc:.4f}",
            config.verbose,
        )

        if no_improvement_epochs >= config.early_stopping_patience:
            _log(
                f"Early stopping at epoch {epoch}; restoring epoch {best_epoch}.",
                config.verbose,
            )
            break

    model.load_state_dict(best_state)
    best_validation_scores = _predict_scores(
        model=model,
        features=x_validation,
        device=device,
        batch_size=batch_size,
        config=config,
    )
    training_seconds = float(time.perf_counter() - started)

    metadata = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "batch_size": int(batch_size),
        "positive_class_weight": float(pos_weight_value),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history_rows)),
        "best_validation_pr_auc": float(best_validation_pr_auc),
        "best_validation_weighted_bce_loss": float(best_validation_loss),
        "mlp_training_seconds": float(training_seconds),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    return (
        model,
        pd.DataFrame(history_rows),
        metadata,
        best_validation_scores,
        device,
    )


def _train_mlp_fixed_epochs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    config: Exp2Config,
    seed: int,
) -> tuple[ShallowMLP, dict, torch.device]:
    """Train a final MLP on all supplied rows for a preselected epoch count."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if set(np.unique(y_train).tolist()) != {0, 1}:
        raise ValueError("Final training partition must contain both classes.")

    _set_reproducible_seed(seed)
    device = _resolve_device(config)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    batch_size = config.batch_size_gpu if device.type == "cuda" else config.batch_size_cpu
    model = ShallowMLP(
        input_dim=int(x_train.shape[1]),
        hidden_dim_1=config.hidden_dim_1,
        hidden_dim_2=config.hidden_dim_2,
    ).to(device)

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    pos_weight_value = float(negatives / positives) if config.use_pos_weight else 1.0
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = _build_loader(
        features=x_train,
        labels=y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        device=device,
        config=config,
    )

    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

    return model, {
        "final_retrain_epochs": int(epochs),
        "final_retrain_seconds": float(time.perf_counter() - started),
        "positive_class_weight": float(pos_weight_value),
        "device": str(device),
        "batch_size": int(batch_size),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }, device


def _run_single_fold(
    outer_train_frame: pd.DataFrame,
    outer_test_frame: pd.DataFrame,
    static_lookup: pd.DataFrame,
    fold_id: int,
    config: Exp2Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Fit one project-grouped outer fold and score its held-out projects."""
    label = f"Fold {fold_id + 1}/{config.n_splits}"
    started = time.perf_counter()
    _log(
        f"{label} started | outer-train={len(outer_train_frame):,}, "
        f"outer-test={len(outer_test_frame):,}, "
        f"outer-train projects={outer_train_frame[config.project_column].nunique():,}, "
        f"outer-test projects={outer_test_frame[config.project_column].nunique():,}.",
        config.verbose,
    )

    fit_indices, validation_indices, inner_metadata = _select_inner_validation_split(
        outer_train_frame=outer_train_frame,
        config=config,
        outer_fold_id=fold_id,
    )
    fit_frame = outer_train_frame.iloc[fit_indices].reset_index(drop=True)
    validation_frame = outer_train_frame.iloc[validation_indices].reset_index(drop=True)

    _log(
        f"{label} | inner project split: fit={len(fit_frame):,} rows / "
        f"{fit_frame[config.project_column].nunique():,} projects; "
        f"validation={len(validation_frame):,} rows / "
        f"{validation_frame[config.project_column].nunique():,} projects.",
        config.verbose,
    )

    (
        x_fit_lexical,
        x_validation_lexical,
        x_test_lexical,
        _word_vectorizer,
        _char_vectorizer,
        lexical_metadata,
    ) = _fit_lexical_matrices(
        fit_code=fit_frame[config.code_column],
        validation_code=validation_frame[config.code_column],
        test_code=outer_test_frame[config.code_column],
        config=config,
        fold_id=fold_id,
    )

    (
        x_fit_svd,
        x_validation_svd,
        x_test_svd,
        _svd,
        svd_seconds,
        svd_variance_ratio,
    ) = _fit_and_transform_svd(
        x_fit_lexical=x_fit_lexical,
        x_validation_lexical=x_validation_lexical,
        x_test_lexical=x_test_lexical,
        config=config,
        fold_id=fold_id,
    )

    _log(f"{label} | loading deterministic static features...", config.verbose)
    static_started = time.perf_counter()
    x_fit_static = _get_static_matrix(static_lookup, fit_frame[config.source_id_column])
    x_validation_static = _get_static_matrix(
        static_lookup, validation_frame[config.source_id_column]
    )
    x_test_static = _get_static_matrix(
        static_lookup, outer_test_frame[config.source_id_column]
    )
    static_lookup_seconds = float(time.perf_counter() - static_started)

    concatenate_started = time.perf_counter()
    x_fit = np.hstack([x_fit_svd, x_fit_static]).astype(np.float32, copy=False)
    x_validation = np.hstack([x_validation_svd, x_validation_static]).astype(
        np.float32, copy=False
    )
    x_test = np.hstack([x_test_svd, x_test_static]).astype(np.float32, copy=False)
    dense_concat_seconds = float(time.perf_counter() - concatenate_started)

    expected_features = config.svd_n_components + len(FEATURE_COLUMNS)
    if x_fit.shape[1] != expected_features:
        raise RuntimeError(
            f"Expected {expected_features} dense inputs; received {x_fit.shape[1]}."
        )

    _log(f"{label} | fitting inner-train StandardScaler...", config.verbose)
    x_fit, x_validation, x_test, _scaler, scaler_seconds = _fit_scaler_and_transform(
        x_fit=x_fit,
        x_validation=x_validation,
        x_test=x_test,
    )

    y_fit = fit_frame[config.label_column].to_numpy(dtype=np.int8)
    y_validation = validation_frame[config.label_column].to_numpy(dtype=np.int8)
    y_test = outer_test_frame[config.label_column].to_numpy(dtype=np.int8)

    _log(
        f"{label} | training shallow MLP (128 -> 64 -> 1) with early stopping...",
        config.verbose,
    )
    (
        model,
        training_history,
        mlp_metadata,
        validation_scores,
        device,
    ) = _train_mlp_with_early_stopping(
        x_fit=x_fit,
        y_fit=y_fit,
        x_validation=x_validation,
        y_validation=y_validation,
        config=config,
        fold_seed=config.random_state + 1_000 + fold_id,
    )

    selected_threshold, validation_threshold_metrics = _select_validation_threshold(
        labels=y_validation,
        scores=validation_scores,
        config=config,
    )
    _log(
        f"{label} | inner-validation F1 threshold={selected_threshold:.3f} "
        f"(val F1={validation_threshold_metrics['f1']:.4f}, "
        f"val PR-AUC={validation_threshold_metrics['average_precision_pr_auc']:.4f}).",
        config.verbose,
    )

    _log(f"{label} | scoring outer held-out projects...", config.verbose)
    prediction_started = time.perf_counter()
    batch_size = config.batch_size_gpu if device.type == "cuda" else config.batch_size_cpu
    y_score = _predict_scores(
        model=model,
        features=x_test,
        device=device,
        batch_size=batch_size,
        config=config,
    )
    y_pred_validation_threshold = (y_score >= selected_threshold).astype(np.int8)
    prediction_seconds = float(time.perf_counter() - prediction_started)

    predictions = outer_test_frame[
        [
            config.source_id_column,
            config.fold_column,
            config.label_column,
            config.project_column,
        ]
    ].copy().rename(
        columns={
            config.source_id_column: "source_row_id",
            config.fold_column: "fold",
            config.label_column: "label",
            config.project_column: "project",
        }
    )
    predictions["y_score"] = y_score.astype(np.float64)
    predictions["selected_validation_threshold"] = float(selected_threshold)
    predictions["y_pred_validation_threshold"] = y_pred_validation_threshold

    training_history = training_history.copy()
    training_history.insert(0, "fold", int(fold_id))

    total_seconds = float(time.perf_counter() - started)
    metadata = {
        "fold": int(fold_id),
        "outer_train_rows": int(len(outer_train_frame)),
        "outer_test_rows": int(len(outer_test_frame)),
        "outer_train_vulnerable": int((outer_train_frame[config.label_column] == 1).sum()),
        "outer_test_vulnerable": int((outer_test_frame[config.label_column] == 1).sum()),
        "outer_train_positive_rate": float(outer_train_frame[config.label_column].mean()),
        "outer_test_positive_rate": float(outer_test_frame[config.label_column].mean()),
        "outer_train_unique_projects": int(outer_train_frame[config.project_column].nunique()),
        "outer_test_unique_projects": int(outer_test_frame[config.project_column].nunique()),
        **inner_metadata,
        **lexical_metadata,
        "svd_components": int(config.svd_n_components),
        "svd_seconds": float(svd_seconds),
        "svd_explained_variance_ratio_sum": float(svd_variance_ratio),
        "static_features": int(len(FEATURE_COLUMNS)),
        "static_lookup_seconds": float(static_lookup_seconds),
        "dense_feature_concatenation_seconds": float(dense_concat_seconds),
        "scaler_seconds": float(scaler_seconds),
        "final_dense_features": int(expected_features),
        **mlp_metadata,
        "selected_validation_threshold": float(selected_threshold),
        "validation_average_precision_pr_auc": float(
            validation_threshold_metrics["average_precision_pr_auc"]
        ),
        "validation_precision": float(validation_threshold_metrics["precision"]),
        "validation_recall": float(validation_threshold_metrics["recall"]),
        "validation_f1": float(validation_threshold_metrics["f1"]),
        "validation_mcc": float(validation_threshold_metrics["mcc"]),
        "validation_threshold_at_grid_boundary": bool(
            validation_threshold_metrics["selected_at_grid_boundary"]
        ),
        "prediction_seconds": float(prediction_seconds),
        "total_fold_seconds": float(total_seconds),
        "optimizer": "AdamW",
        "score_min": float(np.min(y_score)),
        "score_max": float(np.max(y_score)),
        "score_mean": float(np.mean(y_score)),
    }

    _log(f"{label} completed in {_format_seconds(total_seconds)}.", config.verbose)

    del (
        x_fit_lexical,
        x_validation_lexical,
        x_test_lexical,
        x_fit_svd,
        x_validation_svd,
        x_test_svd,
        x_fit_static,
        x_validation_static,
        x_test_static,
        x_fit,
        x_validation,
        x_test,
        model,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return predictions, training_history, metadata, validation_threshold_metrics


def _evaluate_foldwise_validation_thresholds(predictions: pd.DataFrame) -> dict:
    """Evaluate OOF decisions made with thresholds selected on inner validation only."""
    required = {
        "fold",
        "label",
        "y_score",
        "y_pred_validation_threshold",
        "selected_validation_threshold",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Missing validation operating-point columns: {sorted(missing)}")

    fold_rows: list[dict] = []
    for fold_id, frame in predictions.groupby("fold", sort=True):
        selected_threshold_values = frame["selected_validation_threshold"].unique()
        if len(selected_threshold_values) != 1:
            raise RuntimeError(f"Fold {fold_id} has inconsistent selected thresholds.")
        metric = _binary_metrics_from_decisions(
            labels=frame["label"],
            scores=frame["y_score"],
            decisions=frame["y_pred_validation_threshold"],
            threshold=float(selected_threshold_values[0]),
        )
        metric["fold"] = int(fold_id)
        metric["test_unique_projects"] = int(frame["project"].nunique())
        fold_rows.append(metric)

    fold_metrics = pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)
    pooled_metrics = _binary_metrics_from_decisions(
        labels=predictions["label"],
        scores=predictions["y_score"],
        decisions=predictions["y_pred_validation_threshold"],
        threshold=None,
    )
    pooled_metrics["threshold_strategy"] = "per_fold_inner_project_validation_f1"
    pooled_metrics["selected_threshold_min"] = float(
        predictions["selected_validation_threshold"].min()
    )
    pooled_metrics["selected_threshold_max"] = float(
        predictions["selected_validation_threshold"].max()
    )
    pooled_metrics["selected_threshold_mean"] = float(
        predictions["selected_validation_threshold"].mean()
    )

    rows: list[dict] = []
    for column in [
        "average_precision_pr_auc",
        "precision",
        "recall",
        "f1",
        "mcc",
        "false_positive_rate",
        "predicted_positive_rate",
        "threshold",
    ]:
        values = fold_metrics[column].astype(float)
        rows.append(
            {
                "metric": column,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    return {
        "pooled_metrics": pooled_metrics,
        "fold_metrics": fold_metrics,
        "fold_summary": pd.DataFrame(rows),
    }


def run_exp2_profile_fold(
    normalized_frame: pd.DataFrame,
    static_features_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    fold_id: int = 0,
    config: Exp2Config = Exp2Config(),
) -> dict:
    """Run one outer development fold to validate MLP feasibility and workflow."""
    _validate_config(config)
    if fold_id not in range(config.n_splits):
        raise ValueError(f"fold_id must be in [0, {config.n_splits - 1}], got {fold_id}.")

    _log(
        f"CS1-EXP2 profiling mode: running Fold {fold_id + 1}/{config.n_splits} only.",
        config.verbose,
    )
    dataset, static_lookup = _prepare_dataset_and_static_lookup(
        normalized_frame=normalized_frame,
        static_features_frame=static_features_frame,
        manifest=manifest,
        config=config,
    )
    outer_train_frame = dataset.loc[
        dataset[config.fold_column] != fold_id
    ].reset_index(drop=True)
    outer_test_frame = dataset.loc[
        dataset[config.fold_column] == fold_id
    ].reset_index(drop=True)

    predictions, history, metadata, validation_metrics = _run_single_fold(
        outer_train_frame=outer_train_frame,
        outer_test_frame=outer_test_frame,
        static_lookup=static_lookup,
        fold_id=fold_id,
        config=config,
    )

    threshold = float(predictions["selected_validation_threshold"].iloc[0])
    profile_metrics = _binary_metrics_from_decisions(
        labels=predictions["label"],
        scores=predictions["y_score"],
        decisions=predictions["y_pred_validation_threshold"],
        threshold=threshold,
    )
    profile_metrics["threshold_strategy"] = "inner_project_validation_f1"
    default_threshold_metrics = compute_binary_metrics(
        labels=predictions["label"],
        scores=predictions["y_score"],
        threshold=config.decision_threshold,
    )

    _log(
        "Profiling run complete. These metrics are descriptive for one outer "
        "development fold only; they are not the official 5-fold EXP-2 result.",
        config.verbose,
    )

    del outer_train_frame, outer_test_frame, dataset, static_lookup
    gc.collect()
    return {
        "fold_id": int(fold_id),
        "predictions": predictions,
        "training_history": history,
        "training_metadata": pd.DataFrame([metadata]),
        "validation_metrics": validation_metrics,
        "profile_metrics": profile_metrics,
        "default_threshold_metrics": default_threshold_metrics,
    }


def _save_exp2_artifacts(
    results: Mapping[str, object],
    output_dir: Path | str,
    config: Exp2Config,
    additional_metadata: Optional[Mapping[str, object]] = None,
) -> Exp2Artifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_predictions = results["oof_predictions"]
    fold_training = results["fold_training"]
    training_history = results["training_history"]
    evaluation = results["evaluation"]
    validation_operating_evaluation = results["validation_operating_evaluation"]

    if not isinstance(oof_predictions, pd.DataFrame):
        raise TypeError("oof_predictions must be a DataFrame.")
    if not isinstance(fold_training, pd.DataFrame):
        raise TypeError("fold_training must be a DataFrame.")
    if not isinstance(training_history, pd.DataFrame):
        raise TypeError("training_history must be a DataFrame.")

    safe_name = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in config.experiment_name.strip().lower().replace(" ", "_")
    )

    config_json = output_dir / f"{safe_name}_config.json"
    fold_training_csv = output_dir / f"{safe_name}_fold_training.csv"
    training_history_csv = output_dir / f"{safe_name}_training_history.csv"
    validation_operating_metrics_json = (
        output_dir / f"{safe_name}_validation_operating_pooled_metrics.json"
    )
    validation_operating_fold_metrics_csv = (
        output_dir / f"{safe_name}_validation_operating_fold_metrics.csv"
    )
    validation_operating_fold_summary_csv = (
        output_dir / f"{safe_name}_validation_operating_fold_summary.csv"
    )
    run_metadata_json = output_dir / f"{safe_name}_run_metadata.json"

    with config_json.open("w", encoding="utf-8") as file:
        json.dump({"exp2_version": EXP2_VERSION, "config": asdict(config)}, file, indent=2)

    fold_training.to_csv(fold_training_csv, index=False)
    training_history.to_csv(training_history_csv, index=False)
    with validation_operating_metrics_json.open("w", encoding="utf-8") as file:
        json.dump(validation_operating_evaluation["pooled_metrics"], file, indent=2)
    validation_operating_evaluation["fold_metrics"].to_csv(
        validation_operating_fold_metrics_csv, index=False
    )
    validation_operating_evaluation["fold_summary"].to_csv(
        validation_operating_fold_summary_csv, index=False
    )

    evaluation_paths = save_evaluation_artifacts(
        evaluation=evaluation,
        output_dir=output_dir,
        experiment_name=config.experiment_name,
        config=EvaluationConfig(
            threshold=config.decision_threshold,
            expected_n_folds=config.n_splits,
        ),
        additional_metadata={
            "exp2_version": EXP2_VERSION,
            "model": "PyTorch shallow MLP 310 -> 128 -> 64 -> 1 with AdamW",
            "representation": (
                "inner-train-only word/character TF-IDF, inner-train-only "
                "TruncatedSVD, 54 deterministic static features, and "
                "inner-train-only StandardScaler"
            ),
            "class_imbalance_strategy": (
                "BCEWithLogitsLoss positive-class weight calculated as "
                "negative/positive from the inner optimisation partition"
            ),
            "model_selection_strategy": (
                "project-disjoint inner validation selects early-stopping epoch "
                "by PR-AUC and F1 threshold on validation scores only"
            ),
            "score_interpretation": (
                "Sigmoid risk score; not a calibrated real-world vulnerability probability"
            ),
            **dict(additional_metadata or {}),
        },
    )

    metadata = {
        "exp2_version": EXP2_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config.experiment_name,
        "config": asdict(config),
        "n_oof_predictions": int(len(oof_predictions)),
        "folds_completed": sorted(oof_predictions["fold"].astype(int).unique().tolist()),
        "total_vectorization_seconds": float(fold_training["vectorization_seconds"].sum()),
        "total_svd_seconds": float(fold_training["svd_seconds"].sum()),
        "total_mlp_training_seconds": float(fold_training["mlp_training_seconds"].sum()),
        "total_prediction_seconds": float(fold_training["prediction_seconds"].sum()),
        "total_runtime_seconds": float(fold_training["total_fold_seconds"].sum()),
        "device_by_fold": {
            str(int(row.fold)): str(row.device)
            for row in fold_training[["fold", "device"]].itertuples(index=False)
        },
        "additional_metadata": dict(additional_metadata or {}),
    }
    with run_metadata_json.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)

    return Exp2Artifacts(
        config_json=config_json,
        fold_training_csv=fold_training_csv,
        training_history_csv=training_history_csv,
        validation_operating_metrics_json=validation_operating_metrics_json,
        validation_operating_fold_metrics_csv=validation_operating_fold_metrics_csv,
        validation_operating_fold_summary_csv=validation_operating_fold_summary_csv,
        run_metadata_json=run_metadata_json,
        evaluation_paths=evaluation_paths,
    )


def run_exp2(
    normalized_frame: pd.DataFrame,
    static_features_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    config: Exp2Config = Exp2Config(),
    output_dir: Optional[Path | str] = None,
    additional_metadata: Optional[Mapping[str, object]] = None,
) -> dict:
    """Run the official 5-fold project-aware development-CV MLP experiment."""
    _validate_config(config)
    device = _resolve_device(config)
    _log(
        f"CS1-EXP2 official run started: {config.n_splits}-fold grouped CV.",
        config.verbose,
    )
    _log(
        f"Configuration: lexical<= {config.word_max_features + config.char_max_features:,}, "
        f"SVD={config.svd_n_components}, static={len(FEATURE_COLUMNS)}, "
        f"MLP={config.hidden_dim_1}->{config.hidden_dim_2}->1, device={device}.",
        config.verbose,
    )

    dataset, static_lookup = _prepare_dataset_and_static_lookup(
        normalized_frame=normalized_frame,
        static_features_frame=static_features_frame,
        manifest=manifest,
        config=config,
    )

    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    training_rows: list[dict] = []
    validation_rows: list[dict] = []
    started = time.perf_counter()

    for fold_id in range(config.n_splits):
        outer_train_frame = dataset.loc[
            dataset[config.fold_column] != fold_id
        ].reset_index(drop=True)
        outer_test_frame = dataset.loc[
            dataset[config.fold_column] == fold_id
        ].reset_index(drop=True)
        if outer_train_frame.empty or outer_test_frame.empty:
            raise RuntimeError(f"Fold {fold_id} has an empty partition.")
        if set(outer_test_frame[config.label_column].unique().tolist()) != {0, 1}:
            raise RuntimeError(f"Outer fold {fold_id} does not contain both classes.")

        predictions, history, metadata, validation_metrics = _run_single_fold(
            outer_train_frame=outer_train_frame,
            outer_test_frame=outer_test_frame,
            static_lookup=static_lookup,
            fold_id=fold_id,
            config=config,
        )
        prediction_frames.append(predictions)
        history_frames.append(history)
        training_rows.append(metadata)
        validation_metrics = dict(validation_metrics)
        validation_metrics["outer_fold"] = int(fold_id)
        validation_rows.append(validation_metrics)

        del outer_train_frame, outer_test_frame
        gc.collect()

    oof_predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        .sort_values("source_row_id", kind="stable")
        .reset_index(drop=True)
    )
    training_history = (
        pd.concat(history_frames, ignore_index=True)
        .sort_values(["fold", "epoch"])
        .reset_index(drop=True)
    )
    fold_training = pd.DataFrame(training_rows).sort_values("fold").reset_index(drop=True)
    validation_selection_metrics = (
        pd.DataFrame(validation_rows).sort_values("outer_fold").reset_index(drop=True)
    )

    if len(oof_predictions) != len(dataset):
        raise RuntimeError(
            "OOF prediction count does not match development rows: "
            f"{len(oof_predictions):,} vs {len(dataset):,}."
        )
    if oof_predictions["source_row_id"].duplicated().any():
        raise RuntimeError("Duplicate source_row_id detected in MLP OOF predictions.")

    # Fixed 0.50 is preserved as a diagnostic. MLP operational metrics below
    # use thresholds selected solely from each fold's inner validation scores.
    evaluation = evaluate_oof_predictions(
        oof_predictions,
        config=EvaluationConfig(
            threshold=config.decision_threshold,
            expected_n_folds=config.n_splits,
        ),
    )
    validation_operating_evaluation = _evaluate_foldwise_validation_thresholds(
        oof_predictions
    )

    total_runtime_seconds = float(time.perf_counter() - started)
    _log(
        f"CS1-EXP2 official run completed in {_format_seconds(total_runtime_seconds)}.",
        config.verbose,
    )

    results = {
        "oof_predictions": evaluation["predictions"],
        "fold_training": fold_training,
        "training_history": training_history,
        "validation_selection_metrics": validation_selection_metrics,
        "evaluation": evaluation,
        "validation_operating_evaluation": validation_operating_evaluation,
        "total_runtime_seconds": total_runtime_seconds,
    }
    if output_dir is not None:
        results["artifacts"] = _save_exp2_artifacts(
            results=results,
            output_dir=output_dir,
            config=config,
            additional_metadata=additional_metadata,
        )
        _log(f"Artifacts saved to: {Path(output_dir)}", config.verbose)
    return results


def _validate_final_holdout_partitions(
    dev_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    config: Exp2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly validate the frozen development and outer-holdout partitions."""
    required = [
        config.source_id_column,
        config.code_column,
        config.label_column,
        config.project_column,
    ]
    _require_columns(dev_frame, required)
    _require_columns(holdout_frame, required)

    dev_clean = dev_frame.copy()
    holdout_clean = holdout_frame.copy()
    for name, frame in (("development", dev_clean), ("outer holdout", holdout_clean)):
        if frame.empty:
            raise ValueError(f"{name} partition is empty.")
        if frame[config.source_id_column].isna().any() or frame[config.source_id_column].duplicated().any():
            raise ValueError(f"{name} partition has invalid source_row_id values.")
        frame[config.code_column] = frame[config.code_column].fillna("").astype(str)
        if frame[config.code_column].str.strip().eq("").any():
            raise ValueError(f"{name} partition contains empty code rows.")
        frame[config.project_column] = frame[config.project_column].fillna("").astype(str).str.strip()
        if frame[config.project_column].eq("").any():
            raise ValueError(f"{name} partition contains empty project IDs.")
        labels = pd.to_numeric(frame[config.label_column], errors="raise")
        if not labels.isin([0, 1]).all() or set(labels.unique().tolist()) != {0, 1}:
            raise ValueError(f"{name} partition must contain both binary classes.")
        frame[config.label_column] = labels.astype("int8")

    source_overlap = set(dev_clean[config.source_id_column]).intersection(
        set(holdout_clean[config.source_id_column])
    )
    if source_overlap:
        raise RuntimeError("source_row_id leakage between development and outer holdout.")
    project_overlap = set(dev_clean[config.project_column]).intersection(
        set(holdout_clean[config.project_column])
    )
    if project_overlap:
        raise RuntimeError(
            "Project leakage between development and outer holdout; examples: "
            f"{sorted(project_overlap)[:10]}"
        )
    return dev_clean, holdout_clean


def run_exp2_final_holdout(
    dev_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    static_features_frame: pd.DataFrame,
    config: Exp2Config = Exp2Config(),
    output_dir: Optional[Path | str] = None,
    additional_metadata: Optional[Mapping[str, object]] = None,
) -> dict:
    """
    Run one final MLP evaluation on the sealed outer holdout.

    First, a project-disjoint validation partition inside development chooses the
    early-stopping epoch and threshold. Then preprocessing and the MLP are
    retrained on all development rows for that selected epoch count before the
    outer holdout is scored once.
    """
    _validate_config(config)
    dev_clean, holdout_clean = _validate_final_holdout_partitions(
        dev_frame=dev_frame,
        holdout_frame=holdout_frame,
        config=config,
    )
    static_lookup = _validate_static_feature_frame(static_features_frame, config)
    required_ids = pd.Index(
        pd.concat(
            [dev_clean[config.source_id_column], holdout_clean[config.source_id_column]],
            ignore_index=True,
        )
    )
    missing_static = required_ids.difference(static_lookup.index)
    if len(missing_static):
        raise ValueError(f"Static cache is missing {len(missing_static):,} final-evaluation IDs.")

    _log("FINAL HOLDOUT | selecting epoch and threshold inside development only...", config.verbose)
    fit_indices, validation_indices, inner_metadata = _select_inner_validation_split(
        outer_train_frame=dev_clean,
        config=config,
        outer_fold_id=99_999,
    )
    fit_frame = dev_clean.iloc[fit_indices].reset_index(drop=True)
    validation_frame = dev_clean.iloc[validation_indices].reset_index(drop=True)

    (
        x_fit_lexical,
        x_val_lexical,
        _unused_test_lexical,
        _word_validation,
        _char_validation,
        _validation_lexical_metadata,
    ) = _fit_lexical_matrices(
        fit_code=fit_frame[config.code_column],
        validation_code=validation_frame[config.code_column],
        test_code=holdout_clean[config.code_column],
        config=config,
        fold_id=-1,
    )
    (
        x_fit_svd,
        x_val_svd,
        x_holdout_svd,
        _validation_svd,
        _validation_svd_seconds,
        _validation_variance,
    ) = _fit_and_transform_svd(
        x_fit_lexical=x_fit_lexical,
        x_validation_lexical=x_val_lexical,
        x_test_lexical=_unused_test_lexical,
        config=config,
        fold_id=-1,
    )
    x_fit_static = _get_static_matrix(static_lookup, fit_frame[config.source_id_column])
    x_val_static = _get_static_matrix(static_lookup, validation_frame[config.source_id_column])
    x_holdout_static = _get_static_matrix(static_lookup, holdout_clean[config.source_id_column])
    x_fit = np.hstack([x_fit_svd, x_fit_static]).astype(np.float32, copy=False)
    x_val = np.hstack([x_val_svd, x_val_static]).astype(np.float32, copy=False)
    x_holdout = np.hstack([x_holdout_svd, x_holdout_static]).astype(np.float32, copy=False)
    x_fit, x_val, _x_holdout_temp, _validation_scaler, _ = _fit_scaler_and_transform(
        x_fit=x_fit, x_validation=x_val, x_test=x_holdout
    )
    y_fit = fit_frame[config.label_column].to_numpy(dtype=np.int8)
    y_val = validation_frame[config.label_column].to_numpy(dtype=np.int8)
    (
        _validation_model,
        _validation_history,
        validation_training_metadata,
        validation_scores,
        _validation_device,
    ) = _train_mlp_with_early_stopping(
        x_fit=x_fit,
        y_fit=y_fit,
        x_validation=x_val,
        y_validation=y_val,
        config=config,
        fold_seed=config.random_state + 99_999,
    )
    selected_threshold, validation_threshold_metrics = _select_validation_threshold(
        labels=y_val, scores=validation_scores, config=config
    )
    selected_epochs = int(validation_training_metadata["best_epoch"])

    _log(
        f"FINAL HOLDOUT | retraining on all development rows for {selected_epochs} epoch(s)...",
        config.verbose,
    )
    (
        x_dev_lexical,
        _unused_validation_lexical,
        x_final_holdout_lexical,
        word_vectorizer,
        char_vectorizer,
        final_lexical_metadata,
    ) = _fit_lexical_matrices(
        fit_code=dev_clean[config.code_column],
        validation_code=dev_clean[config.code_column].iloc[:1],
        test_code=holdout_clean[config.code_column],
        config=config,
        fold_id=-2,
    )
    (
        x_dev_svd,
        _unused_validation_svd,
        x_final_holdout_svd,
        svd,
        svd_seconds,
        svd_variance,
    ) = _fit_and_transform_svd(
        x_fit_lexical=x_dev_lexical,
        x_validation_lexical=_unused_validation_lexical,
        x_test_lexical=x_final_holdout_lexical,
        config=config,
        fold_id=-2,
    )
    x_dev_static = _get_static_matrix(static_lookup, dev_clean[config.source_id_column])
    x_final_holdout_static = _get_static_matrix(static_lookup, holdout_clean[config.source_id_column])
    x_dev = np.hstack([x_dev_svd, x_dev_static]).astype(np.float32, copy=False)
    x_final_holdout = np.hstack([x_final_holdout_svd, x_final_holdout_static]).astype(np.float32, copy=False)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    x_dev = scaler.fit_transform(x_dev).astype(np.float32, copy=False)
    x_final_holdout = scaler.transform(x_final_holdout).astype(np.float32, copy=False)

    y_dev = dev_clean[config.label_column].to_numpy(dtype=np.int8)
    y_holdout = holdout_clean[config.label_column].to_numpy(dtype=np.int8)
    final_model, final_training_metadata, final_device = _train_mlp_fixed_epochs(
        x_train=x_dev,
        y_train=y_dev,
        epochs=selected_epochs,
        config=config,
        seed=config.random_state + 199_999,
    )
    batch_size = config.batch_size_gpu if final_device.type == "cuda" else config.batch_size_cpu
    y_score = _predict_scores(
        final_model, x_final_holdout, final_device, batch_size, config
    )
    y_pred = (y_score >= selected_threshold).astype(np.int8)
    metrics = _binary_metrics_from_decisions(
        labels=y_holdout, scores=y_score, decisions=y_pred, threshold=selected_threshold
    )
    metrics["threshold_strategy"] = "development_inner_project_validation_f1"

    predictions = pd.DataFrame(
        {
            "source_row_id": holdout_clean[config.source_id_column].to_numpy(),
            "label": y_holdout,
            "project": holdout_clean[config.project_column].to_numpy(),
            "y_score": y_score,
            "selected_validation_threshold": float(selected_threshold),
            "y_pred": y_pred,
        }
    )

    results = {
        "experiment": config.experiment_name,
        "model": final_model,
        "predictions": predictions,
        "metrics": metrics,
        "selected_threshold": float(selected_threshold),
        "validation_threshold_metrics": validation_threshold_metrics,
        "validation_selection_metadata": {
            **inner_metadata,
            **validation_training_metadata,
        },
        "final_training_metadata": final_training_metadata,
        "vector_metadata": {
            **final_lexical_metadata,
            "svd_seconds": float(svd_seconds),
            "svd_explained_variance_ratio_sum": float(svd_variance),
            "final_dense_features": int(config.svd_n_components + len(FEATURE_COLUMNS)),
        },
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(
            c if c.isalnum() or c in {"_", "-"} else "_"
            for c in config.experiment_name.strip().lower().replace(" ", "_")
        )
        predictions_path = output_dir / f"{safe_name}_holdout_predictions.parquet"
        metrics_path = output_dir / f"{safe_name}_holdout_metrics.json"
        model_path = output_dir / f"{safe_name}_final_model_state.pt"
        preprocessing_path = output_dir / f"{safe_name}_final_preprocessing.joblib"
        predictions.to_parquet(predictions_path, index=False)
        torch.save(final_model.state_dict(), model_path)
        joblib.dump(
            {
                "word_vectorizer": word_vectorizer,
                "char_vectorizer": char_vectorizer,
                "svd": svd,
                "scaler": scaler,
                "feature_columns": FEATURE_COLUMNS,
                "input_dim": int(config.svd_n_components + len(FEATURE_COLUMNS)),
            },
            preprocessing_path,
        )
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "exp2_version": EXP2_VERSION,
                    "config": asdict(config),
                    "metrics": metrics,
                    "selected_threshold": float(selected_threshold),
                    "validation_threshold_metrics": validation_threshold_metrics,
                    "validation_selection_metadata": results["validation_selection_metadata"],
                    "final_training_metadata": final_training_metadata,
                    "additional_metadata": dict(additional_metadata or {}),
                },
                file,
                indent=2,
                default=str,
            )
        results["artifact_paths"] = {
            "predictions": predictions_path,
            "metrics": metrics_path,
            "model_state": model_path,
            "preprocessing": preprocessing_path,
        }
        _log(f"EXP-2 final-holdout artifacts saved to: {output_dir}", config.verbose)

    return results


__all__ = [
    "EXP2_VERSION",
    "Exp2Artifacts",
    "Exp2Config",
    "ShallowMLP",
    "run_exp2",
    "run_exp2_profile_fold",
    "run_exp2_final_holdout",
]
