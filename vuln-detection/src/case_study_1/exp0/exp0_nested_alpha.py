"""
CS1 EXP-0 nested project-grouped alpha-tuning study (development partition only).

Purpose
-------
This module extends the frozen CS1 EXP-0 baseline with a *separate* nested
cross-validation sensitivity study for the SGD Logistic Regression L2
regularisation parameter ``alpha``.

It must not replace the completed fixed-configuration EXP-0 baseline and it
must never access the frozen 20% global outer holdout. The intended claim is:

    "How robust is EXP-0 to a small, predeclared alpha grid when alpha is
     selected using only project-disjoint inner folds?"

Nested procedure
----------------
The existing frozen 5-fold development manifest is reused as the outer
*development-evaluation* partition. For each outer-development fold:

1. The fold's projects remain untouched for final scoring.
2. The remaining projects are split using 3-fold StratifiedGroupKFold.
3. Each alpha is evaluated with pooled inner out-of-fold PR-AUC.
4. The chosen alpha is refit on all outer-training projects.
5. The refit model scores the untouched outer-development test projects.

No score from an outer-development test fold is used to choose alpha.

Efficiency
----------
TF-IDF matrices are fitted once per inner fold and reused across all alpha
values. Only the inexpensive SGD model fit is repeated per alpha. This keeps
nested CV practical while preserving train-only feature fitting.

Important scope boundary
------------------------
This is a post-selection robustness / tuning study. The global 20% outer
holdout has already been consumed by the selected EXP-2 MLP in the main study,
so this module must not be used to reopen final model selection or rerun that
holdout. It writes to a dedicated output directory and records this scope in
metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
import warnings
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from ..evaluation import EvaluationConfig, evaluate_oof_predictions, save_evaluation_artifacts
from ..split_manifest import SplitConfig
from .exp0_lr import (
    EXP0_VERSION,
    Exp0Config,
    _build_model,
    _fit_and_transform_fold,
    _format_seconds,
    _log,
    _prepare_dataset_with_folds,
)


NESTED_EXP0_VERSION = "cs1-exp0-nested-alpha-v1-development-only"


@dataclass(frozen=True)
class NestedAlphaConfig:
    """Configuration for the separate nested EXP-0 alpha sensitivity study."""

    experiment_name: str = "cs1_exp0_nested_alpha_dev_grouped"
    alpha_grid: tuple[float, ...] = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
    inner_n_splits: int = 3
    inner_random_state: int = 20260707
    selection_metric: str = "average_precision_pr_auc"
    decision_threshold: float = 0.50
    tie_break_rule: str = "higher_alpha_then_grid_order"
    top_features_per_direction: int = 30
    verbose: bool = True


@dataclass(frozen=True)
class NestedAlphaArtifacts:
    """Paths saved by a completed nested EXP-0 alpha study."""

    nested_config_json: Path
    inner_alpha_scores_csv: Path
    selected_alpha_csv: Path
    inner_split_audit_csv: Path
    outer_fold_training_csv: Path
    run_metadata_json: Path
    run_state_json: Path
    evaluation_paths: object


def _json_default(value: object) -> object:
    """Serialize NumPy and pathlib values in transparent artifact metadata."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _stable_float(value: float) -> float:
    """Normalize floating-point values used in deterministic configuration checks."""
    return float(f"{float(value):.16g}")


def _validate_nested_config(base_config: Exp0Config, nested_config: NestedAlphaConfig) -> None:
    """Fail early on invalid grids or incompatible fixed EXP-0 assumptions."""
    if base_config.decision_threshold != nested_config.decision_threshold:
        raise ValueError(
            "Nested and base thresholds must match. This study keeps the EXP-0 "
            "fixed threshold policy unchanged."
        )
    if nested_config.selection_metric != "average_precision_pr_auc":
        raise ValueError(
            "This implementation intentionally selects alpha by pooled inner PR-AUC."
        )
    if nested_config.inner_n_splits < 2:
        raise ValueError("inner_n_splits must be at least 2.")
    if not nested_config.alpha_grid:
        raise ValueError("alpha_grid cannot be empty.")
    alpha_values = tuple(_stable_float(alpha) for alpha in nested_config.alpha_grid)
    if len(set(alpha_values)) != len(alpha_values):
        raise ValueError("alpha_grid contains duplicate values.")
    if any(alpha <= 0.0 or not np.isfinite(alpha) for alpha in alpha_values):
        raise ValueError("Every alpha must be a finite value > 0.")
    if nested_config.tie_break_rule != "higher_alpha_then_grid_order":
        raise ValueError(
            "Only the declared tie-break rule 'higher_alpha_then_grid_order' is supported."
        )


def _base_config_for_alpha(base_config: Exp0Config, alpha: float, *, verbose: bool) -> Exp0Config:
    """Return a fixed EXP-0 configuration differing only in alpha and verbosity."""
    return replace(
        base_config,
        sgd_alpha=float(alpha),
        top_features_per_direction=0,
        verbose=verbose,
    )


def _assert_two_classes(frame: pd.DataFrame, label_column: str, context: str) -> None:
    values = set(pd.to_numeric(frame[label_column], errors="raise").astype(int).unique().tolist())
    if values != {0, 1}:
        raise RuntimeError(
            f"{context} must contain both classes {{0, 1}}; observed {sorted(values)}."
        )


def _assert_project_disjoint(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    project_column: str,
    context: str,
) -> None:
    overlap = set(train_frame[project_column].astype(str)).intersection(
        set(test_frame[project_column].astype(str))
    )
    if overlap:
        raise RuntimeError(
            f"Project leakage in {context}. Examples: {sorted(overlap)[:10]}"
        )


def _fit_scores_for_alpha(
    x_train,
    y_train: np.ndarray,
    x_test,
    config: Exp0Config,
) -> tuple[np.ndarray, dict]:
    """Fit one SGD Logistic Regression candidate and produce positive-class scores."""
    model = _build_model(config)
    fit_start = time.perf_counter()
    convergence_messages: list[str] = []

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)

    for warning_record in caught_warnings:
        if issubclass(warning_record.category, ConvergenceWarning):
            convergence_messages.append(str(warning_record.message))

    fit_seconds = float(time.perf_counter() - fit_start)
    positive_class_indices = np.where(model.classes_ == 1)[0]
    if len(positive_class_indices) != 1:
        raise RuntimeError("The fitted SGD model does not expose class label 1.")

    prediction_start = time.perf_counter()
    y_score = model.predict_proba(x_test)[:, int(positive_class_indices[0])]
    prediction_seconds = float(time.perf_counter() - prediction_start)

    metadata = {
        "model_fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "model_n_iter": int(model.n_iter_),
        "convergence_warning_count": int(len(convergence_messages)),
        "convergence_warning_messages": " | ".join(convergence_messages),
    }

    return y_score.astype(np.float64), metadata


def _inner_splitter(n_splits: int, random_state: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def _select_alpha_from_scores(alpha_summary: pd.DataFrame) -> pd.Series:
    """
    Select by highest pooled inner PR-AUC.

    Exact ties select the higher alpha (stronger regularisation / simpler fit),
    then retain original declared-grid order. This rule is fixed before results.
    """
    if alpha_summary.empty:
        raise RuntimeError("Cannot select alpha from an empty inner-score table.")

    ordered = alpha_summary.sort_values(
        ["inner_pooled_pr_auc", "alpha", "alpha_grid_order"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return ordered.iloc[0]


def _run_inner_alpha_selection(
    outer_train_frame: pd.DataFrame,
    outer_fold_id: int,
    base_config: Exp0Config,
    nested_config: NestedAlphaConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Tune alpha on project-disjoint inner CV within one outer-development train split.

    Vectorization is fitted once per inner fold, then reused across all alpha values.
    This is valid because vectorizer settings do not vary in the grid and each
    vectorizer is fit only on that inner fold's training projects.
    """
    outer_label = f"Outer development fold {outer_fold_id + 1}/{base_config.n_splits}"
    _log(
        f"{outer_label} | inner tuning started "
        f"({nested_config.inner_n_splits}-fold project-grouped CV; "
        f"{len(nested_config.alpha_grid)} alpha values).",
        nested_config.verbose,
    )

    _assert_two_classes(outer_train_frame, base_config.label_column, f"{outer_label} training partition")

    alpha_values = tuple(_stable_float(alpha) for alpha in nested_config.alpha_grid)
    splitter = _inner_splitter(
        nested_config.inner_n_splits,
        nested_config.inner_random_state + int(outer_fold_id),
    )

    labels = outer_train_frame[base_config.label_column].to_numpy(dtype=np.int8)
    groups = outer_train_frame[base_config.project_column].astype(str).to_numpy()
    dummy_x = np.zeros((len(outer_train_frame), 1), dtype=np.uint8)

    prediction_rows_by_alpha: dict[float, list[pd.DataFrame]] = {
        alpha: [] for alpha in alpha_values
    }
    score_rows: list[dict] = []
    audit_rows: list[dict] = []

    for inner_fold_id, (inner_train_pos, inner_valid_pos) in enumerate(
        splitter.split(dummy_x, labels, groups)
    ):
        inner_train = outer_train_frame.iloc[inner_train_pos].reset_index(drop=True)
        inner_valid = outer_train_frame.iloc[inner_valid_pos].reset_index(drop=True)

        context = f"{outer_label}, inner fold {inner_fold_id + 1}/{nested_config.inner_n_splits}"
        _assert_project_disjoint(inner_train, inner_valid, base_config.project_column, context)
        _assert_two_classes(inner_train, base_config.label_column, f"{context} train")
        _assert_two_classes(inner_valid, base_config.label_column, f"{context} validation")

        _log(
            f"{context} | vectorizing once for all alpha candidates...",
            nested_config.verbose,
        )

        vector_config = _base_config_for_alpha(
            base_config,
            alpha=base_config.sgd_alpha,
            verbose=False,
        )
        x_train, x_valid, word_vectorizer, char_vectorizer, vector_metadata = _fit_and_transform_fold(
            train_code=inner_train[base_config.code_column],
            test_code=inner_valid[base_config.code_column],
            config=vector_config,
            fold_id=inner_fold_id,
        )

        _log(
            f"{context} | vectorization done in "
            f"{_format_seconds(vector_metadata['vectorization_seconds'])} "
            f"({vector_metadata['total_features']:,} features).",
            nested_config.verbose,
        )

        y_train = inner_train[base_config.label_column].to_numpy(dtype=np.int8)
        y_valid = inner_valid[base_config.label_column].to_numpy(dtype=np.int8)

        audit_rows.append(
            {
                "outer_fold": int(outer_fold_id),
                "inner_fold": int(inner_fold_id),
                "inner_train_rows": int(len(inner_train)),
                "inner_validation_rows": int(len(inner_valid)),
                "inner_train_projects": int(inner_train[base_config.project_column].nunique()),
                "inner_validation_projects": int(inner_valid[base_config.project_column].nunique()),
                "inner_train_positive_rate": float(inner_train[base_config.label_column].mean()),
                "inner_validation_positive_rate": float(inner_valid[base_config.label_column].mean()),
                "inner_project_overlap": 0,
                **vector_metadata,
            }
        )

        for grid_order, alpha in enumerate(alpha_values):
            candidate_config = _base_config_for_alpha(base_config, alpha=alpha, verbose=False)
            y_score, fit_metadata = _fit_scores_for_alpha(
                x_train=x_train,
                y_train=y_train,
                x_test=x_valid,
                config=candidate_config,
            )
            fold_pr_auc = float(average_precision_score(y_valid, y_score))

            score_rows.append(
                {
                    "outer_fold": int(outer_fold_id),
                    "inner_fold": int(inner_fold_id),
                    "alpha": float(alpha),
                    "alpha_grid_order": int(grid_order),
                    "inner_fold_pr_auc": fold_pr_auc,
                    "inner_train_rows": int(len(inner_train)),
                    "inner_validation_rows": int(len(inner_valid)),
                    "inner_train_projects": int(inner_train[base_config.project_column].nunique()),
                    "inner_validation_projects": int(inner_valid[base_config.project_column].nunique()),
                    **fit_metadata,
                }
            )

            inner_predictions = inner_valid[
                [base_config.source_id_column, base_config.label_column, base_config.project_column]
            ].copy()
            inner_predictions = inner_predictions.rename(
                columns={
                    base_config.source_id_column: "source_row_id",
                    base_config.label_column: "label",
                    base_config.project_column: "project",
                }
            )
            inner_predictions["outer_fold"] = int(outer_fold_id)
            inner_predictions["inner_fold"] = int(inner_fold_id)
            inner_predictions["alpha"] = float(alpha)
            inner_predictions["y_score"] = y_score
            prediction_rows_by_alpha[alpha].append(inner_predictions)

        del x_train, x_valid, word_vectorizer, char_vectorizer
        del inner_train, inner_valid
        gc.collect()

    raw_score_df = pd.DataFrame(score_rows).sort_values(
        ["outer_fold", "alpha_grid_order", "inner_fold"], kind="stable"
    ).reset_index(drop=True)
    audit_df = pd.DataFrame(audit_rows).sort_values(
        ["outer_fold", "inner_fold"], kind="stable"
    ).reset_index(drop=True)

    alpha_summary_rows: list[dict] = []
    for grid_order, alpha in enumerate(alpha_values):
        alpha_predictions = pd.concat(prediction_rows_by_alpha[alpha], ignore_index=True)
        if alpha_predictions["source_row_id"].duplicated().any():
            raise RuntimeError(
                f"{outer_label}: alpha={alpha:g} produced duplicate inner OOF IDs."
            )
        if len(alpha_predictions) != len(outer_train_frame):
            raise RuntimeError(
                f"{outer_label}: alpha={alpha:g} inner OOF coverage mismatch: "
                f"{len(alpha_predictions):,} vs {len(outer_train_frame):,}."
            )

        alpha_rows = raw_score_df.loc[raw_score_df["alpha"] == alpha]
        alpha_summary_rows.append(
            {
                "outer_fold": int(outer_fold_id),
                "alpha": float(alpha),
                "alpha_grid_order": int(grid_order),
                "inner_pooled_pr_auc": float(
                    average_precision_score(alpha_predictions["label"], alpha_predictions["y_score"])
                ),
                "inner_mean_fold_pr_auc": float(alpha_rows["inner_fold_pr_auc"].mean()),
                "inner_std_fold_pr_auc": float(alpha_rows["inner_fold_pr_auc"].std(ddof=0)),
                "inner_oof_rows": int(len(alpha_predictions)),
                "inner_oof_positive_rate": float(alpha_predictions["label"].mean()),
                "total_inner_model_fit_seconds": float(alpha_rows["model_fit_seconds"].sum()),
                "total_inner_prediction_seconds": float(alpha_rows["prediction_seconds"].sum()),
                "inner_convergence_warning_count": int(alpha_rows["convergence_warning_count"].sum()),
            }
        )

    alpha_summary_df = pd.DataFrame(alpha_summary_rows).sort_values(
        ["outer_fold", "alpha_grid_order"], kind="stable"
    ).reset_index(drop=True)
    selected = _select_alpha_from_scores(alpha_summary_df)

    selection_metadata = {
        "selected_alpha": float(selected["alpha"]),
        "selected_inner_pooled_pr_auc": float(selected["inner_pooled_pr_auc"]),
        "selected_inner_mean_fold_pr_auc": float(selected["inner_mean_fold_pr_auc"]),
        "selection_metric": nested_config.selection_metric,
        "tie_break_rule": nested_config.tie_break_rule,
        "total_inner_vectorization_seconds": float(audit_df["vectorization_seconds"].sum()),
        "outer_train_rows": int(len(outer_train_frame)),
        "outer_train_projects": int(outer_train_frame[base_config.project_column].nunique()),
    }

    _log(
        f"{outer_label} | selected alpha={selection_metadata['selected_alpha']:g} "
        f"from inner pooled PR-AUC={selection_metadata['selected_inner_pooled_pr_auc']:.6f}.",
        nested_config.verbose,
    )

    return raw_score_df, alpha_summary_df, audit_df, selection_metadata


def _fit_outer_selected_model(
    outer_train_frame: pd.DataFrame,
    outer_test_frame: pd.DataFrame,
    outer_fold_id: int,
    selected_alpha: float,
    base_config: Exp0Config,
    nested_config: NestedAlphaConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Refit one selected-alpha model and score the untouched outer-development fold."""
    label = f"Outer development fold {outer_fold_id + 1}/{base_config.n_splits}"
    _assert_project_disjoint(outer_train_frame, outer_test_frame, base_config.project_column, label)
    _assert_two_classes(outer_train_frame, base_config.label_column, f"{label} train")
    _assert_two_classes(outer_test_frame, base_config.label_column, f"{label} test")

    _log(
        f"{label} | final refit with selected alpha={selected_alpha:g} "
        f"and scoring untouched outer-development projects...",
        nested_config.verbose,
    )
    fold_start = time.perf_counter()

    selected_config = _base_config_for_alpha(base_config, selected_alpha, verbose=False)
    x_train, x_test, word_vectorizer, char_vectorizer, vector_metadata = _fit_and_transform_fold(
        train_code=outer_train_frame[base_config.code_column],
        test_code=outer_test_frame[base_config.code_column],
        config=selected_config,
        fold_id=outer_fold_id,
    )

    y_train = outer_train_frame[base_config.label_column].to_numpy(dtype=np.int8)
    y_score, model_metadata = _fit_scores_for_alpha(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        config=selected_config,
    )

    outer_predictions = outer_test_frame[
        [
            base_config.source_id_column,
            base_config.fold_column,
            base_config.label_column,
            base_config.project_column,
        ]
    ].copy().rename(
        columns={
            base_config.source_id_column: "source_row_id",
            base_config.fold_column: "fold",
            base_config.label_column: "label",
            base_config.project_column: "project",
        }
    )
    outer_predictions["selected_alpha"] = float(selected_alpha)
    outer_predictions["y_score"] = y_score

    # This nested sensitivity study intentionally does not export linear
    # coefficients. The frozen main EXP-0 baseline remains the canonical source
    # for coefficient-based descriptive analysis, and avoiding a second fit here
    # prevents duplicated computation solely for an auxiliary artifact.

    total_seconds = float(time.perf_counter() - fold_start)
    training_metadata = {
        "outer_fold": int(outer_fold_id),
        "selected_alpha": float(selected_alpha),
        "outer_train_rows": int(len(outer_train_frame)),
        "outer_test_rows": int(len(outer_test_frame)),
        "outer_train_vulnerable": int((outer_train_frame[base_config.label_column] == 1).sum()),
        "outer_test_vulnerable": int((outer_test_frame[base_config.label_column] == 1).sum()),
        "outer_train_positive_rate": float(outer_train_frame[base_config.label_column].mean()),
        "outer_test_positive_rate": float(outer_test_frame[base_config.label_column].mean()),
        "outer_train_projects": int(outer_train_frame[base_config.project_column].nunique()),
        "outer_test_projects": int(outer_test_frame[base_config.project_column].nunique()),
        "outer_project_overlap": 0,
        **vector_metadata,
        **model_metadata,
        "outer_total_fold_seconds": total_seconds,
        "outer_score_min": float(np.min(y_score)),
        "outer_score_max": float(np.max(y_score)),
        "outer_score_mean": float(np.mean(y_score)),
    }

    _log(
        f"{label} | completed in {_format_seconds(total_seconds)}.",
        nested_config.verbose,
    )

    del x_train, x_test, word_vectorizer, char_vectorizer
    gc.collect()

    # No top-feature export is intentional: this is an alpha robustness study;
    # interpretability remains attached to the frozen main EXP-0 baseline.
    return outer_predictions, pd.DataFrame(), training_metadata


def _nested_config_payload(base_config: Exp0Config, nested_config: NestedAlphaConfig) -> dict:
    return {
        "nested_exp0_version": NESTED_EXP0_VERSION,
        "base_exp0_version": EXP0_VERSION,
        "base_config": asdict(base_config),
        "nested_config": asdict(nested_config),
        "scope": {
            "run_kind": "nested_alpha_tuning_development_only",
            "global_outer_holdout_used": False,
            "main_study_status": "Does not replace frozen EXP-0 baseline or locked EXP-2 holdout result.",
            "selection_unit": "pooled inner out-of-fold PR-AUC within each outer-development training partition",
        },
    }


def _prepare_output_dir(
    output_dir: Path | str,
    payload: dict,
    *,
    resume: bool,
) -> tuple[Path, Path]:
    """Create or safely resume a dedicated artifact directory."""
    output_dir = Path(output_dir)
    config_path = output_dir / "cs1_exp0_nested_alpha_config.json"
    state_path = output_dir / "cs1_exp0_nested_alpha_run_state.json"

    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise RuntimeError(
                f"Nested EXP-0 output directory is not empty:\n{output_dir}\n\n"
                "Use a new directory, or resume only an interrupted run with the "
                "identical configuration. Never overwrite a completed result."
            )
        if not config_path.is_file():
            raise RuntimeError(
                f"Cannot resume because configuration artifact is missing:\n{config_path}"
            )
        with config_path.open("r", encoding="utf-8") as file:
            saved_payload = json.load(file)
        if saved_payload != payload:
            raise RuntimeError(
                "Cannot resume: nested configuration differs from the existing "
                "artifact directory. Use a separate output directory."
            )
        if state_path.is_file():
            with state_path.open("r", encoding="utf-8") as file:
                state = json.load(file)
            if state.get("status") == "completed":
                raise RuntimeError(
                    "This nested EXP-0 run is already completed. Do not rerun or overwrite it."
                )
        return output_dir, state_path

    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=_json_default)

    initial_state = {
        "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed_outer_folds": [],
        "note": "Global 20% outer holdout is intentionally not available to this study.",
    }
    with state_path.open("w", encoding="utf-8") as file:
        json.dump(initial_state, file, indent=2)
    return output_dir, state_path


def _checkpoint_dir(output_dir: Path) -> Path:
    path = output_dir / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_paths(output_dir: Path, outer_fold_id: int) -> dict[str, Path]:
    root = _checkpoint_dir(output_dir)
    prefix = f"outer_fold_{outer_fold_id}"
    return {
        "predictions": root / f"{prefix}_predictions.parquet",
        "inner_scores": root / f"{prefix}_inner_scores.csv",
        "selected": root / f"{prefix}_selected_alpha.json",
        "inner_audit": root / f"{prefix}_inner_audit.csv",
        "outer_training": root / f"{prefix}_outer_training.json",
    }


def _write_outer_checkpoint(
    output_dir: Path,
    outer_fold_id: int,
    predictions: pd.DataFrame,
    inner_scores: pd.DataFrame,
    selected: dict,
    inner_audit: pd.DataFrame,
    outer_training: dict,
) -> None:
    paths = _checkpoint_paths(output_dir, outer_fold_id)
    predictions.to_parquet(paths["predictions"], index=False)
    inner_scores.to_csv(paths["inner_scores"], index=False)
    inner_audit.to_csv(paths["inner_audit"], index=False)
    with paths["selected"].open("w", encoding="utf-8") as file:
        json.dump(selected, file, indent=2, default=_json_default)
    with paths["outer_training"].open("w", encoding="utf-8") as file:
        json.dump(outer_training, file, indent=2, default=_json_default)


def _load_outer_checkpoint(output_dir: Path, outer_fold_id: int) -> Optional[dict]:
    paths = _checkpoint_paths(output_dir, outer_fold_id)
    if not any(path.exists() for path in paths.values()):
        return None
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError(
            f"Outer fold {outer_fold_id} checkpoint is incomplete. Remove this "
            "partial checkpoint directory manually only after inspection, then resume."
        )

    with paths["selected"].open("r", encoding="utf-8") as file:
        selected = json.load(file)
    with paths["outer_training"].open("r", encoding="utf-8") as file:
        outer_training = json.load(file)

    predictions = pd.read_parquet(paths["predictions"])
    if predictions["fold"].astype(int).nunique() != 1 or int(predictions["fold"].iloc[0]) != outer_fold_id:
        raise RuntimeError(f"Checkpoint fold identity mismatch for outer fold {outer_fold_id}.")
    if predictions["source_row_id"].duplicated().any():
        raise RuntimeError(f"Checkpoint contains duplicate IDs for outer fold {outer_fold_id}.")

    return {
        "predictions": predictions,
        "inner_scores": pd.read_csv(paths["inner_scores"]),
        "selected": selected,
        "inner_audit": pd.read_csv(paths["inner_audit"]),
        "outer_training": outer_training,
    }


def _update_run_state(state_path: Path, completed_outer_folds: Sequence[int], status: str) -> None:
    state = {
        "status": status,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_outer_folds": sorted(int(value) for value in completed_outer_folds),
        "note": "Global 20% outer holdout is intentionally not available to this study.",
    }
    with state_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


def run_exp0_nested_inner_profile(
    development_frame: pd.DataFrame,
    development_manifest: pd.DataFrame,
    outer_fold_id: int = 4,
    base_config: Exp0Config = Exp0Config(verbose=False),
    nested_config: NestedAlphaConfig = NestedAlphaConfig(),
) -> dict:
    """
    Run only the *inner* tuning stage for one outer-development training split.

    This is a feasibility profile. It deliberately does not score the selected
    model on the outer-development test fold, so the official outer-fold
    evaluation remains unseen until the full nested run.
    """
    _validate_nested_config(base_config, nested_config)
    if outer_fold_id not in range(base_config.n_splits):
        raise ValueError(
            f"outer_fold_id must be in [0, {base_config.n_splits - 1}]."
        )

    prepared = _prepare_dataset_with_folds(
        normalized_frame=development_frame,
        manifest=development_manifest,
        config=base_config,
    )
    outer_train = prepared.loc[prepared[base_config.fold_column] != outer_fold_id].reset_index(drop=True)

    start = time.perf_counter()
    raw_scores, alpha_summary, audit, selected = _run_inner_alpha_selection(
        outer_train_frame=outer_train,
        outer_fold_id=outer_fold_id,
        base_config=base_config,
        nested_config=nested_config,
    )
    total_seconds = float(time.perf_counter() - start)

    _log(
        "Nested EXP-0 inner profile complete. No outer-development test fold and "
        "no global outer holdout was scored.",
        nested_config.verbose,
    )

    return {
        "outer_fold_id": int(outer_fold_id),
        "inner_scores": raw_scores,
        "alpha_summary": alpha_summary,
        "inner_split_audit": audit,
        "selected_alpha": selected,
        "total_profile_seconds": total_seconds,
    }


def run_exp0_nested_alpha(
    development_frame: pd.DataFrame,
    development_manifest: pd.DataFrame,
    base_config: Exp0Config = Exp0Config(verbose=False),
    nested_config: NestedAlphaConfig = NestedAlphaConfig(),
    output_dir: Optional[Path | str] = None,
    resume: bool = False,
    additional_metadata: Optional[Mapping[str, object]] = None,
) -> dict:
    """
    Run nested project-grouped alpha tuning on the frozen development partition only.

    Parameters
    ----------
    development_frame:
        Exactly the 80% development rows. Passing global holdout rows is rejected
        by the manifest coverage checks in the calling notebook.
    development_manifest:
        The existing frozen 5-fold project-grouped manifest for development rows.
    output_dir:
        Dedicated new directory. Existing completed directories cannot be rerun.
    resume:
        Resume a previously interrupted run with identical stored configuration.

    Returns
    -------
    dict
        Outer OOF predictions, inner-selection summaries, evaluation, and paths.
    """
    _validate_nested_config(base_config, nested_config)

    prepared = _prepare_dataset_with_folds(
        normalized_frame=development_frame,
        manifest=development_manifest,
        config=base_config,
    )

    if len(prepared) != len(development_frame):
        raise RuntimeError(
            "Development frame and frozen development manifest differ in row coverage."
        )

    payload = _nested_config_payload(base_config, nested_config)
    output_path: Optional[Path] = None
    state_path: Optional[Path] = None
    if output_dir is not None:
        output_path, state_path = _prepare_output_dir(output_dir, payload, resume=resume)

    _log(
        f"Nested EXP-0 alpha study started: {base_config.n_splits} outer-development folds, "
        f"{nested_config.inner_n_splits} inner project-grouped folds, "
        f"alpha grid={list(nested_config.alpha_grid)}.",
        nested_config.verbose,
    )
    _log(
        "Scope: development partition only. The global 20% outer holdout is not loaded or scored.",
        nested_config.verbose,
    )

    all_outer_predictions: list[pd.DataFrame] = []
    all_inner_scores: list[pd.DataFrame] = []
    all_selected_rows: list[dict] = []
    all_inner_audits: list[pd.DataFrame] = []
    all_outer_training_rows: list[dict] = []
    completed_folds: list[int] = []

    run_start = time.perf_counter()

    try:
        for outer_fold_id in range(base_config.n_splits):
            loaded_checkpoint = _load_outer_checkpoint(output_path, outer_fold_id) if output_path is not None and resume else None
            if loaded_checkpoint is not None:
                _log(
                    f"Outer development fold {outer_fold_id + 1}/{base_config.n_splits} "
                    "loaded from validated checkpoint.",
                    nested_config.verbose,
                )
                all_outer_predictions.append(loaded_checkpoint["predictions"])
                all_inner_scores.append(loaded_checkpoint["inner_scores"])
                all_selected_rows.append(loaded_checkpoint["selected"])
                all_inner_audits.append(loaded_checkpoint["inner_audit"])
                all_outer_training_rows.append(loaded_checkpoint["outer_training"])
                completed_folds.append(outer_fold_id)
                continue

            outer_train = prepared.loc[
                prepared[base_config.fold_column] != outer_fold_id
            ].reset_index(drop=True)
            outer_test = prepared.loc[
                prepared[base_config.fold_column] == outer_fold_id
            ].reset_index(drop=True)
            _assert_project_disjoint(outer_train, outer_test, base_config.project_column, f"outer development fold {outer_fold_id}")

            inner_scores, alpha_summary, inner_audit, selection = _run_inner_alpha_selection(
                outer_train_frame=outer_train,
                outer_fold_id=outer_fold_id,
                base_config=base_config,
                nested_config=nested_config,
            )
            selected_alpha = float(selection["selected_alpha"])

            outer_predictions, _, outer_training = _fit_outer_selected_model(
                outer_train_frame=outer_train,
                outer_test_frame=outer_test,
                outer_fold_id=outer_fold_id,
                selected_alpha=selected_alpha,
                base_config=base_config,
                nested_config=nested_config,
            )

            selected_row = {
                "outer_fold": int(outer_fold_id),
                **selection,
                "alpha_grid": ";".join(f"{alpha:g}" for alpha in nested_config.alpha_grid),
                "inner_n_splits": int(nested_config.inner_n_splits),
            }

            if output_path is not None:
                _write_outer_checkpoint(
                    output_dir=output_path,
                    outer_fold_id=outer_fold_id,
                    predictions=outer_predictions,
                    inner_scores=inner_scores,
                    selected=selected_row,
                    inner_audit=inner_audit,
                    outer_training=outer_training,
                )

            all_outer_predictions.append(outer_predictions)
            all_inner_scores.append(inner_scores)
            all_selected_rows.append(selected_row)
            all_inner_audits.append(inner_audit)
            all_outer_training_rows.append(outer_training)
            completed_folds.append(outer_fold_id)

            if state_path is not None:
                _update_run_state(state_path, completed_folds, status="running")

            del outer_train, outer_test
            gc.collect()

        oof_predictions = pd.concat(all_outer_predictions, ignore_index=True).sort_values(
            "source_row_id", kind="stable"
        ).reset_index(drop=True)
        if len(oof_predictions) != len(prepared):
            raise RuntimeError(
                f"Nested OOF coverage mismatch: {len(oof_predictions):,} vs {len(prepared):,}."
            )
        if oof_predictions["source_row_id"].duplicated().any():
            raise RuntimeError("Nested OOF predictions contain duplicate source_row_id values.")
        if set(oof_predictions["fold"].astype(int).unique().tolist()) != set(range(base_config.n_splits)):
            raise RuntimeError("Nested OOF predictions do not cover all frozen development folds.")

        inner_scores_df = pd.concat(all_inner_scores, ignore_index=True).sort_values(
            ["outer_fold", "alpha_grid_order", "inner_fold"], kind="stable"
        ).reset_index(drop=True)
        selected_alpha_df = pd.DataFrame(all_selected_rows).sort_values("outer_fold").reset_index(drop=True)
        inner_audit_df = pd.concat(all_inner_audits, ignore_index=True).sort_values(
            ["outer_fold", "inner_fold"], kind="stable"
        ).reset_index(drop=True)
        outer_training_df = pd.DataFrame(all_outer_training_rows).sort_values("outer_fold").reset_index(drop=True)

        evaluation = evaluate_oof_predictions(
            oof_predictions,
            config=EvaluationConfig(
                threshold=base_config.decision_threshold,
                expected_n_folds=base_config.n_splits,
            ),
        )
        total_runtime_seconds = float(time.perf_counter() - run_start)

        results = {
            "oof_predictions": evaluation["predictions"],
            "inner_scores": inner_scores_df,
            "selected_alpha": selected_alpha_df,
            "inner_split_audit": inner_audit_df,
            "outer_fold_training": outer_training_df,
            "evaluation": evaluation,
            "total_runtime_seconds": total_runtime_seconds,
        }

        if output_path is not None:
            artifacts = _save_nested_artifacts(
                results=results,
                output_dir=output_path,
                base_config=base_config,
                nested_config=nested_config,
                additional_metadata=additional_metadata,
            )
            results["artifacts"] = artifacts
            if state_path is not None:
                _update_run_state(state_path, completed_folds, status="completed")

        _log(
            f"Nested EXP-0 alpha study completed in {_format_seconds(total_runtime_seconds)}.",
            nested_config.verbose,
        )
        return results

    except Exception:
        if state_path is not None:
            _update_run_state(state_path, completed_folds, status="interrupted")
        raise


def _save_nested_artifacts(
    results: Mapping[str, object],
    output_dir: Path,
    base_config: Exp0Config,
    nested_config: NestedAlphaConfig,
    additional_metadata: Optional[Mapping[str, object]],
) -> NestedAlphaArtifacts:
    """Persist report-ready nested-study artifacts after all outer folds complete."""
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_predictions = results["oof_predictions"]
    inner_scores = results["inner_scores"]
    selected_alpha = results["selected_alpha"]
    inner_split_audit = results["inner_split_audit"]
    outer_fold_training = results["outer_fold_training"]
    evaluation = results["evaluation"]

    for name, value in {
        "oof_predictions": oof_predictions,
        "inner_scores": inner_scores,
        "selected_alpha": selected_alpha,
        "inner_split_audit": inner_split_audit,
        "outer_fold_training": outer_fold_training,
    }.items():
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")

    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping.")

    safe_name = nested_config.experiment_name
    nested_config_json = output_dir / f"{safe_name}_nested_config.json"
    inner_alpha_scores_csv = output_dir / f"{safe_name}_inner_alpha_scores.csv"
    selected_alpha_csv = output_dir / f"{safe_name}_selected_alpha_per_outer_fold.csv"
    inner_split_audit_csv = output_dir / f"{safe_name}_inner_split_audit.csv"
    outer_fold_training_csv = output_dir / f"{safe_name}_outer_fold_training.csv"
    run_metadata_json = output_dir / f"{safe_name}_run_metadata.json"
    run_state_json = output_dir / "cs1_exp0_nested_alpha_run_state.json"

    # config file already exists before execution to enable safe resume; write the same payload again
    # only after successful completion to make intended configuration explicit in final artifacts.
    with nested_config_json.open("w", encoding="utf-8") as file:
        json.dump(_nested_config_payload(base_config, nested_config), file, indent=2, default=_json_default)

    inner_scores.to_csv(inner_alpha_scores_csv, index=False)
    selected_alpha.to_csv(selected_alpha_csv, index=False)
    inner_split_audit.to_csv(inner_split_audit_csv, index=False)
    outer_fold_training.to_csv(outer_fold_training_csv, index=False)

    evaluation_paths = save_evaluation_artifacts(
        evaluation=evaluation,
        output_dir=output_dir,
        experiment_name=nested_config.experiment_name,
        config=EvaluationConfig(
            threshold=base_config.decision_threshold,
            expected_n_folds=base_config.n_splits,
        ),
        additional_metadata={
            "nested_exp0_version": NESTED_EXP0_VERSION,
            "base_exp0_version": EXP0_VERSION,
            "run_kind": "nested_alpha_tuning_development_only",
            "global_outer_holdout_used": False,
            "alpha_selection": (
                "For each frozen outer-development fold, alpha was selected from "
                "pooled project-grouped inner OOF PR-AUC on outer-training projects only."
            ),
            "tie_break_rule": nested_config.tie_break_rule,
            "score_interpretation": "Risk score, not calibrated real-world probability.",
            **dict(additional_metadata or {}),
        },
    )

    run_metadata = {
        "nested_exp0_version": NESTED_EXP0_VERSION,
        "base_exp0_version": EXP0_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": nested_config.experiment_name,
        "base_config": asdict(base_config),
        "nested_config": asdict(nested_config),
        "n_outer_oof_predictions": int(len(oof_predictions)),
        "outer_folds_completed": sorted(oof_predictions["fold"].astype(int).unique().tolist()),
        "selected_alpha_by_outer_fold": {
            str(int(row.outer_fold)): float(row.selected_alpha)
            for row in selected_alpha.itertuples(index=False)
        },
        "total_inner_model_fit_seconds": float(inner_scores["model_fit_seconds"].sum()),
        "total_inner_prediction_seconds": float(inner_scores["prediction_seconds"].sum()),
        "total_inner_vectorization_seconds": float(inner_split_audit["vectorization_seconds"].sum()),
        "total_outer_model_fit_seconds": float(outer_fold_training["model_fit_seconds"].sum()),
        "total_outer_vectorization_seconds": float(outer_fold_training["vectorization_seconds"].sum()),
        "total_runtime_seconds": float(results["total_runtime_seconds"]),
        "global_outer_holdout_used": False,
        "scope_limit": (
            "Post-selection nested sensitivity study. It does not replace the frozen "
            "baseline comparison or re-open final model selection after the EXP-2 holdout."
        ),
        "additional_metadata": dict(additional_metadata or {}),
    }
    with run_metadata_json.open("w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2, default=_json_default)

    return NestedAlphaArtifacts(
        nested_config_json=nested_config_json,
        inner_alpha_scores_csv=inner_alpha_scores_csv,
        selected_alpha_csv=selected_alpha_csv,
        inner_split_audit_csv=inner_split_audit_csv,
        outer_fold_training_csv=outer_fold_training_csv,
        run_metadata_json=run_metadata_json,
        run_state_json=run_state_json,
        evaluation_paths=evaluation_paths,
    )


__all__ = [
    "NESTED_EXP0_VERSION",
    "NestedAlphaArtifacts",
    "NestedAlphaConfig",
    "run_exp0_nested_alpha",
    "run_exp0_nested_inner_profile",
]
