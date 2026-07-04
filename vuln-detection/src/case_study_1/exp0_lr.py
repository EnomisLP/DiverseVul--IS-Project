"""
CS1-EXP0: word + character TF-IDF with class-weighted Logistic Regression.

For each project-aware outer fold:
1. Fit word and character TF-IDF only on training projects.
2. Train class-weighted L2 Logistic Regression.
3. Score the untouched held-out projects.
4. Return one out-of-fold prediction for every source row.

The raw LR score is stored as ``y_score``. Because class_weight='balanced'
is used, treat it as a vulnerability-risk ranking score rather than a calibrated
real-world probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
import warnings
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .evaluation import EvaluationConfig, evaluate_oof_predictions, save_evaluation_artifacts
from .split_manifest import SplitConfig, apply_manifest, assert_manifest_integrity


EXP0_VERSION = "cs1-exp0-lr-v1"


@dataclass(frozen=True)
class Exp0Config:
    """Declared, reproducible configuration for CS1-EXP0-LR."""

    experiment_name: str = "cs1_exp0_lr"
    code_column: str = "normalized_code"
    source_id_column: str = "source_row_id"
    label_column: str = "label"
    project_column: str = "project"
    fold_column: str = "fold"

    n_splits: int = 5
    random_state: int = 42
    decision_threshold: float = 0.50

    # Lexical/API token sequences.
    word_ngram_range: tuple[int, int] = (1, 3)
    word_min_df: int = 2
    word_max_df: float = 0.995
    word_max_features: int = 100_000

    # Punctuation and local C/C++ syntax patterns.
    char_analyzer: str = "char"
    char_ngram_range: tuple[int, int] = (3, 5)
    char_min_df: int = 3
    char_max_df: float = 0.995
    char_max_features: int = 150_000

    lowercase: bool = False
    sublinear_tf: bool = True
    tfidf_norm: str = "l2"

    # Logistic Regression.
    logistic_c: float = 1.0
    logistic_penalty: str = "l2"
    logistic_solver: str = "saga"
    logistic_class_weight: str = "balanced"
    logistic_max_iter: int = 1000
    logistic_tol: float = 1e-3

    top_features_per_direction: int = 30


@dataclass(frozen=True)
class Exp0Artifacts:
    config_json: Path
    fold_training_csv: Path
    top_features_csv: Path
    run_metadata_json: Path
    evaluation_paths: object


def _validate_config(config: Exp0Config) -> None:
    if config.n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if not 0.0 < float(config.decision_threshold) < 1.0:
        raise ValueError("decision_threshold must lie strictly between 0 and 1.")
    if config.word_ngram_range[0] < 1 or config.word_ngram_range[1] < config.word_ngram_range[0]:
        raise ValueError("word_ngram_range is invalid.")
    if config.char_ngram_range[0] < 1 or config.char_ngram_range[1] < config.char_ngram_range[0]:
        raise ValueError("char_ngram_range is invalid.")
    if config.word_min_df < 1 or config.char_min_df < 1:
        raise ValueError("min_df values must be >= 1.")
    if config.word_max_features < 1 or config.char_max_features < 1:
        raise ValueError("max_features values must be >= 1.")
    if config.logistic_c <= 0 or config.logistic_max_iter < 1 or config.logistic_tol <= 0:
        raise ValueError("Invalid Logistic Regression hyperparameters.")


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available: {sorted(frame.columns.tolist())}")


def _validate_dataset(frame: pd.DataFrame, config: Exp0Config) -> pd.DataFrame:
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

    if checked[config.source_id_column].isna().any() or checked[config.source_id_column].duplicated().any():
        raise ValueError("source_row_id must be present and unique.")

    checked[config.code_column] = checked[config.code_column].fillna("").astype(str)
    if (checked[config.code_column].str.strip() == "").any():
        raise ValueError("normalized_code contains empty samples.")

    labels = pd.to_numeric(checked[config.label_column], errors="raise")
    if not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError("EXP-0 requires binary labels containing both 0 and 1.")
    checked[config.label_column] = labels.astype("int8")

    checked[config.project_column] = checked[config.project_column].astype(str).str.strip()
    if (checked[config.project_column] == "").any():
        raise ValueError("project contains empty identifiers.")

    folds = pd.to_numeric(checked[config.fold_column], errors="raise").astype(int)
    expected = set(range(config.n_splits))
    if set(folds.unique().tolist()) != expected:
        raise ValueError(f"Expected fold IDs {sorted(expected)}; got {sorted(set(folds.tolist()))}.")
    checked[config.fold_column] = folds.astype("int8")

    return checked


def _make_word_vectorizer(config: Exp0Config) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        # Include one-character identifiers such as i and x.
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


def _make_char_vectorizer(config: Exp0Config) -> TfidfVectorizer:
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


def _make_model(config: Exp0Config) -> LogisticRegression:
    return LogisticRegression(
        C=config.logistic_c,
        penalty=config.logistic_penalty,
        solver=config.logistic_solver,
        class_weight=config.logistic_class_weight,
        max_iter=config.logistic_max_iter,
        tol=config.logistic_tol,
        random_state=config.random_state,
    )


def _top_features(
    model: LogisticRegression,
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    fold_id: int,
    top_n: int,
) -> pd.DataFrame:
    """Return strongest positive/negative coefficient features for one fold."""
    coefficients = np.asarray(model.coef_).reshape(-1)
    word_features = word_vectorizer.get_feature_names_out().astype(str)
    char_features = char_vectorizer.get_feature_names_out().astype(str)

    names = np.concatenate([np.char.add("word::", word_features), np.char.add("char::", char_features)])
    kinds = np.concatenate([np.repeat("word", len(word_features)), np.repeat("char", len(char_features))])

    if len(coefficients) != len(names):
        raise RuntimeError("Model coefficients do not align with TF-IDF features.")

    n = min(int(top_n), len(coefficients))
    rows: list[dict] = []
    groups = [
        ("vulnerable_associated", np.argsort(coefficients)[-n:][::-1]),
        ("non_vulnerable_associated", np.argsort(coefficients)[:n]),
    ]
    for direction, indices in groups:
        for rank, idx in enumerate(indices, start=1):
            rows.append(
                {
                    "fold": int(fold_id),
                    "direction": direction,
                    "rank": int(rank),
                    "feature_type": str(kinds[idx]),
                    "feature": str(names[idx]),
                    "coefficient": float(coefficients[idx]),
                    "abs_coefficient": float(abs(coefficients[idx])),
                }
            )
    return pd.DataFrame(rows)


def _run_one_fold(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    fold_id: int,
    config: Exp0Config,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit one outer fold and return held-out predictions plus diagnostics."""
    start_vectorization = time.perf_counter()
    word_vectorizer = _make_word_vectorizer(config)
    char_vectorizer = _make_char_vectorizer(config)

    train_code = train_frame[config.code_column]
    test_code = test_frame[config.code_column]

    # Critical: fit occurs only on train_code.
    x_train_word = word_vectorizer.fit_transform(train_code)
    x_test_word = word_vectorizer.transform(test_code)
    x_train_char = char_vectorizer.fit_transform(train_code)
    x_test_char = char_vectorizer.transform(test_code)

    x_train = hstack([x_train_word, x_train_char], format="csr", dtype=np.float32)
    x_test = hstack([x_test_word, x_test_char], format="csr", dtype=np.float32)
    vectorization_seconds = float(time.perf_counter() - start_vectorization)

    model = _make_model(config)
    y_train = train_frame[config.label_column].to_numpy(dtype=np.int8)

    fit_start = time.perf_counter()
    convergence_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    for warning_record in caught:
        if issubclass(warning_record.category, ConvergenceWarning):
            convergence_messages.append(str(warning_record.message))
    fit_seconds = float(time.perf_counter() - fit_start)

    predict_start = time.perf_counter()
    y_score = model.predict_proba(x_test)[:, 1]
    predict_seconds = float(time.perf_counter() - predict_start)

    predictions = test_frame[
        [config.source_id_column, config.fold_column, config.label_column, config.project_column]
    ].copy()
    predictions = predictions.rename(
        columns={
            config.source_id_column: "source_row_id",
            config.fold_column: "fold",
            config.label_column: "label",
            config.project_column: "project",
        }
    )
    predictions["y_score"] = y_score.astype(np.float64)

    top_features = _top_features(
        model=model,
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        fold_id=fold_id,
        top_n=config.top_features_per_direction,
    )

    diagnostics = {
        "fold": int(fold_id),
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_vulnerable": int((train_frame[config.label_column] == 1).sum()),
        "test_vulnerable": int((test_frame[config.label_column] == 1).sum()),
        "train_positive_rate": float(train_frame[config.label_column].mean()),
        "test_positive_rate": float(test_frame[config.label_column].mean()),
        "train_unique_projects": int(train_frame[config.project_column].nunique()),
        "test_unique_projects": int(test_frame[config.project_column].nunique()),
        "word_features": int(len(word_vectorizer.get_feature_names_out())),
        "char_features": int(len(char_vectorizer.get_feature_names_out())),
        "total_features": int(x_train.shape[1]),
        "train_nonzero_entries": int(x_train.nnz),
        "test_nonzero_entries": int(x_test.nnz),
        "vectorization_seconds": vectorization_seconds,
        "model_fit_seconds": fit_seconds,
        "prediction_seconds": predict_seconds,
        "logistic_n_iter": int(np.max(model.n_iter_)),
        "convergence_warning_count": int(len(convergence_messages)),
        "convergence_warning_messages": " | ".join(convergence_messages),
        "score_min": float(np.min(y_score)),
        "score_max": float(np.max(y_score)),
        "score_mean": float(np.mean(y_score)),
    }

    # Release memory before the next fold; the matrices can be large.
    del x_train_word, x_test_word, x_train_char, x_test_char
    del x_train, x_test, word_vectorizer, char_vectorizer, model
    gc.collect()

    return predictions, top_features, diagnostics


def _save_results(
    results: Mapping[str, object],
    output_dir: Path | str,
    config: Exp0Config,
    additional_metadata: Optional[Mapping[str, object]],
) -> Exp0Artifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(
        ch if ch.isalnum() or ch in {"_", "-"} else "_"
        for ch in config.experiment_name.strip().lower().replace(" ", "_")
    )

    config_json = output_dir / f"{safe_name}_config.json"
    fold_training_csv = output_dir / f"{safe_name}_fold_training.csv"
    top_features_csv = output_dir / f"{safe_name}_top_features.csv"
    run_metadata_json = output_dir / f"{safe_name}_run_metadata.json"

    with config_json.open("w", encoding="utf-8") as file:
        json.dump({"exp0_version": EXP0_VERSION, "config": asdict(config)}, file, indent=2)

    fold_training = results["fold_training"]
    top_features = results["top_features"]
    evaluation = results["evaluation"]
    if not isinstance(fold_training, pd.DataFrame) or not isinstance(top_features, pd.DataFrame):
        raise TypeError("EXP-0 result tables must be DataFrames.")
    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping.")

    fold_training.to_csv(fold_training_csv, index=False)
    top_features.to_csv(top_features_csv, index=False)

    evaluation_paths = save_evaluation_artifacts(
        evaluation=evaluation,
        output_dir=output_dir,
        experiment_name=config.experiment_name,
        config=EvaluationConfig(
            threshold=config.decision_threshold,
            expected_n_folds=config.n_splits,
        ),
        additional_metadata={
            "exp0_version": EXP0_VERSION,
            "model": "Class-weighted L2 Logistic Regression",
            "features": "word TF-IDF (1,3) + character TF-IDF (3,5)",
            "score_interpretation": "risk score, not post-hoc calibrated probability",
            **dict(additional_metadata or {}),
        },
    )

    metadata = {
        "exp0_version": EXP0_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config.experiment_name,
        "config": asdict(config),
        "n_oof_predictions": int(len(results["oof_predictions"])),
        "total_runtime_seconds": float(results["total_runtime_seconds"]),
        "total_vectorization_seconds": float(fold_training["vectorization_seconds"].sum()),
        "total_model_fit_seconds": float(fold_training["model_fit_seconds"].sum()),
        "total_prediction_seconds": float(fold_training["prediction_seconds"].sum()),
        "folds_with_convergence_warning": int((fold_training["convergence_warning_count"] > 0).sum()),
        "additional_metadata": dict(additional_metadata or {}),
    }
    with run_metadata_json.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return Exp0Artifacts(
        config_json=config_json,
        fold_training_csv=fold_training_csv,
        top_features_csv=top_features_csv,
        run_metadata_json=run_metadata_json,
        evaluation_paths=evaluation_paths,
    )


def run_exp0(
    normalized_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    config: Exp0Config = Exp0Config(),
    output_dir: Optional[Path | str] = None,
    additional_metadata: Optional[Mapping[str, object]] = None,
) -> dict:
    """Run all project-aware OOF folds for CS1-EXP0-LR."""
    _validate_config(config)

    split_config = SplitConfig(
        n_splits=config.n_splits,
        random_state=config.random_state,
        shuffle=True,
    )
    assert_manifest_integrity(manifest, config=split_config)

    frame = apply_manifest(
        frame=normalized_frame,
        manifest=manifest,
        source_id_column=config.source_id_column,
    )
    frame = _validate_dataset(frame, config)

    # Central grouped-CV guarantee, checked again right before model fitting.
    for fold_id in range(config.n_splits):
        test_projects = set(frame.loc[frame[config.fold_column] == fold_id, config.project_column])
        train_projects = set(frame.loc[frame[config.fold_column] != fold_id, config.project_column])
        if test_projects.intersection(train_projects):
            raise ValueError(f"Project leakage detected in fold {fold_id}.")

    all_predictions: list[pd.DataFrame] = []
    all_top_features: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    started = time.perf_counter()

    for fold_id in range(config.n_splits):
        train_frame = frame.loc[frame[config.fold_column] != fold_id].reset_index(drop=True)
        test_frame = frame.loc[frame[config.fold_column] == fold_id].reset_index(drop=True)
        if train_frame.empty or test_frame.empty:
            raise RuntimeError(f"Fold {fold_id} has an empty partition.")
        if set(test_frame[config.label_column].unique().tolist()) != {0, 1}:
            raise RuntimeError(f"Test fold {fold_id} does not contain both classes.")

        predictions, top_features, diagnostics = _run_one_fold(
            train_frame=train_frame,
            test_frame=test_frame,
            fold_id=fold_id,
            config=config,
        )
        all_predictions.append(predictions)
        all_top_features.append(top_features)
        fold_rows.append(diagnostics)

        del train_frame, test_frame
        gc.collect()

    raw_oof = pd.concat(all_predictions, ignore_index=True).sort_values("source_row_id").reset_index(drop=True)
    if len(raw_oof) != len(frame) or raw_oof["source_row_id"].duplicated().any():
        raise RuntimeError("OOF predictions must contain exactly one row per input sample.")

    evaluation = evaluate_oof_predictions(
        raw_oof,
        config=EvaluationConfig(
            threshold=config.decision_threshold,
            expected_n_folds=config.n_splits,
        ),
    )

    results = {
        "oof_predictions": evaluation["predictions"],
        "fold_training": pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True),
        "top_features": pd.concat(all_top_features, ignore_index=True).sort_values(
            ["fold", "direction", "rank"]
        ).reset_index(drop=True),
        "evaluation": evaluation,
        "total_runtime_seconds": float(time.perf_counter() - started),
    }

    if output_dir is not None:
        results["artifacts"] = _save_results(
            results=results,
            output_dir=output_dir,
            config=config,
            additional_metadata=additional_metadata,
        )

    return results


__all__ = ["EXP0_VERSION", "Exp0Artifacts", "Exp0Config", "run_exp0"]
