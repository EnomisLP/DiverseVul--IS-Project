"""
CS1 EXP-0 nested project-grouped random-oversampling study (development only).

This is a separate supplementary robustness study. It evaluates a small,
predeclared set of imbalance-handling policies inside a strict nested
project-grouped CV procedure:

* balanced_no_sampling: the frozen EXP-0 class-weighted baseline;
* ros_1_to_8_unweighted: duplicate only real vulnerable training functions
  until minority / majority = 1 / 8, then train without class weights;
* ros_1_to_4_unweighted: duplicate only real vulnerable training functions
  until minority / majority = 1 / 4, then train without class weights.

For each frozen outer-development fold, the complete candidate configuration
(policy + alpha) is selected by pooled inner project-grouped OOF PR-AUC. The
outer-development projects are never used during selection. The global 20%
holdout must not be passed to this module and is never scored.

Oversampling is deliberately performed only *after* TF-IDF is fitted on the
original inner/outer training rows. Thus duplicated examples do not alter the
learned vocabulary or IDF weights; they only alter how often real vulnerable
training rows contribute to the SGD objective.
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
from .exp0_lr import (
    EXP0_VERSION,
    Exp0Config,
    _build_model,
    _fit_and_transform_fold,
    _format_seconds,
    _log,
    _prepare_dataset_with_folds,
)


NESTED_ROS_EXP0_VERSION = "cs1-exp0-nested-ros-v1-development-only"


@dataclass(frozen=True)
class SamplingPolicy:
    """One predeclared imbalance-handling candidate."""

    name: str
    target_minority_to_majority_ratio: Optional[float]
    class_weight: Optional[str]
    policy_complexity_rank: int
    description: str


DEFAULT_SAMPLING_POLICIES: tuple[SamplingPolicy, ...] = (
    SamplingPolicy(
        name="balanced_no_sampling",
        target_minority_to_majority_ratio=None,
        class_weight="balanced",
        policy_complexity_rank=0,
        description="No resampling; preserve the frozen EXP-0 class_weight='balanced' policy.",
    ),
    SamplingPolicy(
        name="ros_1_to_8_unweighted",
        target_minority_to_majority_ratio=0.125,
        class_weight=None,
        policy_complexity_rank=1,
        description=(
            "Randomly duplicate real vulnerable training functions after TF-IDF "
            "until minority/majority=1/8; use class_weight=None."
        ),
    ),
    SamplingPolicy(
        name="ros_1_to_4_unweighted",
        target_minority_to_majority_ratio=0.25,
        class_weight=None,
        policy_complexity_rank=2,
        description=(
            "Randomly duplicate real vulnerable training functions after TF-IDF "
            "until minority/majority=1/4; use class_weight=None."
        ),
    ),
)


@dataclass(frozen=True)
class NestedRosConfig:
    """Configuration for the development-only nested ROS study."""

    experiment_name: str = "cs1_exp0_nested_ros_dev_grouped"
    alpha_grid: tuple[float, ...] = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
    sampling_policies: tuple[SamplingPolicy, ...] = DEFAULT_SAMPLING_POLICIES
    inner_n_splits: int = 3
    inner_random_state: int = 20260707
    oversampling_random_state: int = 20260708
    selection_metric: str = "average_precision_pr_auc"
    decision_threshold: float = 0.50
    tie_break_rule: str = "higher_pr_auc_then_simpler_policy_then_higher_alpha_then_declared_order"
    verbose: bool = True


@dataclass(frozen=True)
class NestedRosArtifacts:
    """Saved paths returned by a completed nested ROS study."""

    nested_config_json: Path
    inner_candidate_scores_csv: Path
    selected_candidate_csv: Path
    inner_split_audit_csv: Path
    resampling_audit_csv: Path
    outer_fold_training_csv: Path
    run_metadata_json: Path
    run_state_json: Path
    evaluation_paths: object


def _json_default(value: object) -> object:
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
    return float(f"{float(value):.16g}")


def _assert_two_classes(frame: pd.DataFrame, label_column: str, context: str) -> None:
    values = set(pd.to_numeric(frame[label_column], errors="raise").astype(int).unique().tolist())
    if values != {0, 1}:
        raise RuntimeError(f"{context} must contain both classes {{0, 1}}; observed {sorted(values)}.")


def _assert_project_disjoint(train_frame: pd.DataFrame, test_frame: pd.DataFrame, project_column: str, context: str) -> None:
    overlap = set(train_frame[project_column].astype(str)).intersection(
        set(test_frame[project_column].astype(str))
    )
    if overlap:
        raise RuntimeError(f"Project leakage in {context}. Examples: {sorted(overlap)[:10]}")


def _validate_nested_config(base_config: Exp0Config, nested_config: NestedRosConfig) -> None:
    if base_config.decision_threshold != nested_config.decision_threshold:
        raise ValueError("Nested ROS and base EXP-0 thresholds must match.")
    if nested_config.selection_metric != "average_precision_pr_auc":
        raise ValueError("This study selects complete candidates only by pooled inner PR-AUC.")
    if nested_config.inner_n_splits < 2:
        raise ValueError("inner_n_splits must be at least 2.")
    alpha_values = tuple(_stable_float(alpha) for alpha in nested_config.alpha_grid)
    if not alpha_values or len(alpha_values) != len(set(alpha_values)):
        raise ValueError("alpha_grid must contain unique positive values.")
    if any(alpha <= 0 or not np.isfinite(alpha) for alpha in alpha_values):
        raise ValueError("Every alpha must be finite and > 0.")
    if not nested_config.sampling_policies:
        raise ValueError("At least one sampling policy is required.")

    names = [policy.name for policy in nested_config.sampling_policies]
    if len(names) != len(set(names)):
        raise ValueError("Sampling-policy names must be unique.")
    ranks = [policy.policy_complexity_rank for policy in nested_config.sampling_policies]
    if len(ranks) != len(set(ranks)):
        raise ValueError("Sampling-policy complexity ranks must be unique.")

    for policy in nested_config.sampling_policies:
        ratio = policy.target_minority_to_majority_ratio
        if ratio is None:
            if policy.class_weight != "balanced":
                raise ValueError(
                    "The no-resampling control must preserve class_weight='balanced'."
                )
        else:
            if not (0.0 < float(ratio) < 1.0):
                raise ValueError(f"Invalid target ratio for {policy.name}: {ratio}")
            if policy.class_weight is not None:
                raise ValueError(
                    f"ROS policy {policy.name} must use class_weight=None to isolate "
                    "oversampling from explicit class weighting."
                )


def _candidate_config(base_config: Exp0Config, *, alpha: float, class_weight: Optional[str]) -> Exp0Config:
    return replace(
        base_config,
        sgd_alpha=float(alpha),
        sgd_class_weight=class_weight,  # sklearn accepts None for unweighted training.
        top_features_per_direction=0,
        verbose=False,
    )


def _inner_splitter(n_splits: int, random_state: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _fit_scores(x_train, y_train: np.ndarray, x_test, config: Exp0Config) -> tuple[np.ndarray, dict]:
    model = _build_model(config)
    start = time.perf_counter()
    convergence_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    for warning_record in caught:
        if issubclass(warning_record.category, ConvergenceWarning):
            convergence_messages.append(str(warning_record.message))
    fit_seconds = float(time.perf_counter() - start)

    positive_idx = np.where(model.classes_ == 1)[0]
    if len(positive_idx) != 1:
        raise RuntimeError("The fitted SGD model does not expose class label 1.")
    prediction_start = time.perf_counter()
    y_score = model.predict_proba(x_test)[:, int(positive_idx[0])]
    prediction_seconds = float(time.perf_counter() - prediction_start)
    return y_score.astype(np.float64), {
        "model_fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "model_n_iter": int(model.n_iter_),
        "convergence_warning_count": int(len(convergence_messages)),
        "convergence_warning_messages": " | ".join(convergence_messages),
    }


def _resampling_seed(
    *, outer_fold_id: int, inner_fold_id: int, policy_rank: int, base_seed: int
) -> int:
    # Fixed deterministic mixing; no model outcome is used here.
    return int(base_seed + 100_000 * (outer_fold_id + 1) + 1_000 * (inner_fold_id + 1) + policy_rank)


def _make_training_indices(
    y_train: np.ndarray,
    policy: SamplingPolicy,
    *,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Return indices of original rows, plus duplicated vulnerable row indices for ROS.

    The returned indices always refer to real inner/outer training rows only.
    No synthetic feature vector is created and no test-project row can enter here.
    """
    y_train = np.asarray(y_train, dtype=np.int8)
    majority_indices = np.flatnonzero(y_train == 0)
    minority_indices = np.flatnonzero(y_train == 1)
    if len(majority_indices) == 0 or len(minority_indices) == 0:
        raise RuntimeError("Resampling requires both classes in the training partition.")

    original_ratio = float(len(minority_indices) / len(majority_indices))
    base_indices = np.arange(len(y_train), dtype=np.int64)

    if policy.target_minority_to_majority_ratio is None:
        return base_indices, {
            "sampling_policy": policy.name,
            "class_weight": policy.class_weight,
            "target_minority_to_majority_ratio": None,
            "original_train_rows": int(len(y_train)),
            "original_majority_rows": int(len(majority_indices)),
            "original_minority_rows": int(len(minority_indices)),
            "original_minority_to_majority_ratio": original_ratio,
            "target_minority_rows": int(len(minority_indices)),
            "duplicated_minority_rows": 0,
            "resampled_train_rows": int(len(y_train)),
            "resampled_majority_rows": int(len(majority_indices)),
            "resampled_minority_rows": int(len(minority_indices)),
            "achieved_minority_to_majority_ratio": original_ratio,
            "oversampling_seed": None,
        }

    target_minority_rows = int(np.ceil(float(policy.target_minority_to_majority_ratio) * len(majority_indices)))
    target_minority_rows = max(target_minority_rows, len(minority_indices))
    n_duplicates = int(target_minority_rows - len(minority_indices))

    if n_duplicates:
        rng = np.random.default_rng(seed)
        duplicate_indices = rng.choice(minority_indices, size=n_duplicates, replace=True).astype(np.int64)
        all_indices = np.concatenate([base_indices, duplicate_indices])
        # Shuffle only training order. It does not create or delete observations.
        all_indices = rng.permutation(all_indices)
    else:
        all_indices = base_indices

    achieved_minority = int(len(minority_indices) + n_duplicates)
    return all_indices, {
        "sampling_policy": policy.name,
        "class_weight": policy.class_weight,
        "target_minority_to_majority_ratio": float(policy.target_minority_to_majority_ratio),
        "original_train_rows": int(len(y_train)),
        "original_majority_rows": int(len(majority_indices)),
        "original_minority_rows": int(len(minority_indices)),
        "original_minority_to_majority_ratio": original_ratio,
        "target_minority_rows": int(target_minority_rows),
        "duplicated_minority_rows": int(n_duplicates),
        "resampled_train_rows": int(len(all_indices)),
        "resampled_majority_rows": int(len(majority_indices)),
        "resampled_minority_rows": achieved_minority,
        "achieved_minority_to_majority_ratio": float(achieved_minority / len(majority_indices)),
        "oversampling_seed": int(seed),
    }


def _select_candidate(summary: pd.DataFrame) -> pd.Series:
    """Predeclared ranking: PR-AUC, simpler policy, higher alpha, declared order."""
    if summary.empty:
        raise RuntimeError("Cannot select from an empty candidate summary.")
    ordered = summary.sort_values(
        [
            "inner_pooled_pr_auc",
            "policy_complexity_rank",
            "alpha",
            "policy_grid_order",
            "alpha_grid_order",
        ],
        ascending=[False, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return ordered.iloc[0]


def _run_inner_selection(
    outer_train_frame: pd.DataFrame,
    outer_fold_id: int,
    base_config: Exp0Config,
    nested_config: NestedRosConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    label = f"Outer development fold {outer_fold_id + 1}/{base_config.n_splits}"
    _log(
        f"{label} | inner joint selection started "
        f"({nested_config.inner_n_splits}-fold project-grouped CV; "
        f"{len(nested_config.sampling_policies)} policies × {len(nested_config.alpha_grid)} alphas).",
        nested_config.verbose,
    )
    _assert_two_classes(outer_train_frame, base_config.label_column, f"{label} training partition")

    alpha_values = tuple(_stable_float(alpha) for alpha in nested_config.alpha_grid)
    policies = nested_config.sampling_policies
    splitter = _inner_splitter(nested_config.inner_n_splits, nested_config.inner_random_state + outer_fold_id)

    labels = outer_train_frame[base_config.label_column].to_numpy(dtype=np.int8)
    groups = outer_train_frame[base_config.project_column].astype(str).to_numpy()
    dummy_x = np.zeros((len(outer_train_frame), 1), dtype=np.uint8)

    key_to_predictions: dict[tuple[str, float], list[pd.DataFrame]] = {
        (policy.name, alpha): [] for policy in policies for alpha in alpha_values
    }
    score_rows: list[dict] = []
    split_audit_rows: list[dict] = []
    resampling_audit_rows: list[dict] = []

    for inner_fold_id, (train_pos, valid_pos) in enumerate(splitter.split(dummy_x, labels, groups)):
        inner_train = outer_train_frame.iloc[train_pos].reset_index(drop=True)
        inner_valid = outer_train_frame.iloc[valid_pos].reset_index(drop=True)
        context = f"{label}, inner fold {inner_fold_id + 1}/{nested_config.inner_n_splits}"
        _assert_project_disjoint(inner_train, inner_valid, base_config.project_column, context)
        _assert_two_classes(inner_train, base_config.label_column, f"{context} train")
        _assert_two_classes(inner_valid, base_config.label_column, f"{context} validation")

        _log(f"{context} | vectorizing once for all policy/alpha candidates...", nested_config.verbose)
        vector_config = _candidate_config(
            base_config,
            alpha=base_config.sgd_alpha,
            class_weight=base_config.sgd_class_weight,
        )
        x_train, x_valid, word_vectorizer, char_vectorizer, vector_metadata = _fit_and_transform_fold(
            train_code=inner_train[base_config.code_column],
            test_code=inner_valid[base_config.code_column],
            config=vector_config,
            fold_id=inner_fold_id,
        )
        _log(
            f"{context} | vectorization done in {_format_seconds(vector_metadata['vectorization_seconds'])} "
            f"({vector_metadata['total_features']:,} features).",
            nested_config.verbose,
        )

        y_train = inner_train[base_config.label_column].to_numpy(dtype=np.int8)
        y_valid = inner_valid[base_config.label_column].to_numpy(dtype=np.int8)
        split_audit_rows.append({
            "outer_fold": int(outer_fold_id),
            "inner_fold": int(inner_fold_id),
            "inner_train_rows": int(len(inner_train)),
            "inner_validation_rows": int(len(inner_valid)),
            "inner_train_projects": int(inner_train[base_config.project_column].nunique()),
            "inner_validation_projects": int(inner_valid[base_config.project_column].nunique()),
            "inner_train_positive_rate": float(y_train.mean()),
            "inner_validation_positive_rate": float(y_valid.mean()),
            "inner_project_overlap": 0,
            **vector_metadata,
        })

        for policy_grid_order, policy in enumerate(policies):
            seed = _resampling_seed(
                outer_fold_id=outer_fold_id,
                inner_fold_id=inner_fold_id,
                policy_rank=policy.policy_complexity_rank,
                base_seed=nested_config.oversampling_random_state,
            )
            train_indices, resampling_metadata = _make_training_indices(y_train, policy, seed=seed)
            x_fit = x_train[train_indices]
            y_fit = y_train[train_indices]
            resampling_audit_rows.append({
                "outer_fold": int(outer_fold_id),
                "inner_fold": int(inner_fold_id),
                "policy_grid_order": int(policy_grid_order),
                "policy_complexity_rank": int(policy.policy_complexity_rank),
                **resampling_metadata,
            })

            for alpha_grid_order, alpha in enumerate(alpha_values):
                candidate_config = _candidate_config(
                    base_config,
                    alpha=alpha,
                    class_weight=policy.class_weight,
                )
                y_score, fit_metadata = _fit_scores(x_fit, y_fit, x_valid, candidate_config)
                fold_pr_auc = float(average_precision_score(y_valid, y_score))
                score_rows.append({
                    "outer_fold": int(outer_fold_id),
                    "inner_fold": int(inner_fold_id),
                    "sampling_policy": policy.name,
                    "policy_grid_order": int(policy_grid_order),
                    "policy_complexity_rank": int(policy.policy_complexity_rank),
                    "target_minority_to_majority_ratio": policy.target_minority_to_majority_ratio,
                    "class_weight": policy.class_weight,
                    "alpha": float(alpha),
                    "alpha_grid_order": int(alpha_grid_order),
                    "inner_fold_pr_auc": fold_pr_auc,
                    **resampling_metadata,
                    **fit_metadata,
                })
                pred = inner_valid[[base_config.source_id_column, base_config.label_column, base_config.project_column]].copy()
                pred = pred.rename(columns={
                    base_config.source_id_column: "source_row_id",
                    base_config.label_column: "label",
                    base_config.project_column: "project",
                })
                pred["outer_fold"] = int(outer_fold_id)
                pred["inner_fold"] = int(inner_fold_id)
                pred["sampling_policy"] = policy.name
                pred["alpha"] = float(alpha)
                pred["y_score"] = y_score
                key_to_predictions[(policy.name, alpha)].append(pred)

            del x_fit, y_fit
            gc.collect()

        del x_train, x_valid, word_vectorizer, char_vectorizer, inner_train, inner_valid
        gc.collect()

    raw_scores = pd.DataFrame(score_rows).sort_values(
        ["outer_fold", "policy_grid_order", "alpha_grid_order", "inner_fold"], kind="stable"
    ).reset_index(drop=True)
    split_audit = pd.DataFrame(split_audit_rows).sort_values(["outer_fold", "inner_fold"], kind="stable").reset_index(drop=True)
    resampling_audit = pd.DataFrame(resampling_audit_rows).sort_values(
        ["outer_fold", "inner_fold", "policy_grid_order"], kind="stable"
    ).reset_index(drop=True)

    summary_rows: list[dict] = []
    for policy_grid_order, policy in enumerate(policies):
        for alpha_grid_order, alpha in enumerate(alpha_values):
            candidate_predictions = pd.concat(key_to_predictions[(policy.name, alpha)], ignore_index=True)
            if candidate_predictions["source_row_id"].duplicated().any():
                raise RuntimeError(f"{label}: duplicate inner OOF IDs for {policy.name}, alpha={alpha:g}.")
            if len(candidate_predictions) != len(outer_train_frame):
                raise RuntimeError(
                    f"{label}: inner OOF coverage mismatch for {policy.name}, alpha={alpha:g}: "
                    f"{len(candidate_predictions):,} vs {len(outer_train_frame):,}."
                )
            candidate_rows = raw_scores.loc[
                (raw_scores["sampling_policy"] == policy.name) & (raw_scores["alpha"] == alpha)
            ]
            summary_rows.append({
                "outer_fold": int(outer_fold_id),
                "sampling_policy": policy.name,
                "policy_grid_order": int(policy_grid_order),
                "policy_complexity_rank": int(policy.policy_complexity_rank),
                "target_minority_to_majority_ratio": policy.target_minority_to_majority_ratio,
                "class_weight": policy.class_weight,
                "alpha": float(alpha),
                "alpha_grid_order": int(alpha_grid_order),
                "inner_pooled_pr_auc": float(average_precision_score(candidate_predictions["label"], candidate_predictions["y_score"])),
                "inner_mean_fold_pr_auc": float(candidate_rows["inner_fold_pr_auc"].mean()),
                "inner_std_fold_pr_auc": float(candidate_rows["inner_fold_pr_auc"].std(ddof=0)),
                "inner_oof_rows": int(len(candidate_predictions)),
                "inner_oof_positive_rate": float(candidate_predictions["label"].mean()),
                "total_inner_model_fit_seconds": float(candidate_rows["model_fit_seconds"].sum()),
                "total_inner_prediction_seconds": float(candidate_rows["prediction_seconds"].sum()),
                "inner_convergence_warning_count": int(candidate_rows["convergence_warning_count"].sum()),
            })
    candidate_summary = pd.DataFrame(summary_rows).sort_values(
        ["outer_fold", "policy_grid_order", "alpha_grid_order"], kind="stable"
    ).reset_index(drop=True)
    selected = _select_candidate(candidate_summary)
    selection = {
        "selected_sampling_policy": str(selected["sampling_policy"]),
        "selected_policy_complexity_rank": int(selected["policy_complexity_rank"]),
        "selected_target_minority_to_majority_ratio": (
            None if pd.isna(selected["target_minority_to_majority_ratio"])
            else float(selected["target_minority_to_majority_ratio"])
        ),
        "selected_class_weight": None if pd.isna(selected["class_weight"]) else str(selected["class_weight"]),
        "selected_alpha": float(selected["alpha"]),
        "selected_inner_pooled_pr_auc": float(selected["inner_pooled_pr_auc"]),
        "selected_inner_mean_fold_pr_auc": float(selected["inner_mean_fold_pr_auc"]),
        "selection_metric": nested_config.selection_metric,
        "tie_break_rule": nested_config.tie_break_rule,
        "outer_train_rows": int(len(outer_train_frame)),
        "outer_train_projects": int(outer_train_frame[base_config.project_column].nunique()),
        "total_inner_vectorization_seconds": float(split_audit["vectorization_seconds"].sum()),
    }
    _log(
        f"{label} | selected policy={selection['selected_sampling_policy']} "
        f"alpha={selection['selected_alpha']:g} from inner pooled PR-AUC="
        f"{selection['selected_inner_pooled_pr_auc']:.6f}.",
        nested_config.verbose,
    )
    return raw_scores, candidate_summary, split_audit, resampling_audit, selection


def _policy_by_name(nested_config: NestedRosConfig, name: str) -> SamplingPolicy:
    for policy in nested_config.sampling_policies:
        if policy.name == name:
            return policy
    raise KeyError(f"Unknown sampling policy: {name}")


def _fit_outer_selected_model(
    outer_train_frame: pd.DataFrame,
    outer_test_frame: pd.DataFrame,
    outer_fold_id: int,
    selected: Mapping[str, object],
    base_config: Exp0Config,
    nested_config: NestedRosConfig,
) -> tuple[pd.DataFrame, dict, dict]:
    label = f"Outer development fold {outer_fold_id + 1}/{base_config.n_splits}"
    _assert_project_disjoint(outer_train_frame, outer_test_frame, base_config.project_column, label)
    _assert_two_classes(outer_train_frame, base_config.label_column, f"{label} train")
    _assert_two_classes(outer_test_frame, base_config.label_column, f"{label} test")

    policy = _policy_by_name(nested_config, str(selected["selected_sampling_policy"]))
    alpha = float(selected["selected_alpha"])
    _log(
        f"{label} | final refit with policy={policy.name}, alpha={alpha:g}; "
        "scoring untouched outer-development projects...",
        nested_config.verbose,
    )
    start = time.perf_counter()
    selected_config = _candidate_config(base_config, alpha=alpha, class_weight=policy.class_weight)
    x_train, x_test, word_vectorizer, char_vectorizer, vector_metadata = _fit_and_transform_fold(
        train_code=outer_train_frame[base_config.code_column],
        test_code=outer_test_frame[base_config.code_column],
        config=selected_config,
        fold_id=outer_fold_id,
    )
    y_train = outer_train_frame[base_config.label_column].to_numpy(dtype=np.int8)
    y_test = outer_test_frame[base_config.label_column].to_numpy(dtype=np.int8)
    seed = _resampling_seed(
        outer_fold_id=outer_fold_id,
        inner_fold_id=99,  # distinct from every inner split; fixed before execution.
        policy_rank=policy.policy_complexity_rank,
        base_seed=nested_config.oversampling_random_state,
    )
    fit_indices, resampling_metadata = _make_training_indices(y_train, policy, seed=seed)
    x_fit, y_fit = x_train[fit_indices], y_train[fit_indices]
    y_score, fit_metadata = _fit_scores(x_fit, y_fit, x_test, selected_config)
    total_seconds = float(time.perf_counter() - start)

    predictions = outer_test_frame[[base_config.source_id_column, base_config.label_column, base_config.project_column]].copy()
    predictions = predictions.rename(columns={
        base_config.source_id_column: "source_row_id",
        base_config.label_column: "label",
        base_config.project_column: "project",
    })
    predictions["fold"] = int(outer_fold_id)
    predictions["selected_sampling_policy"] = policy.name
    predictions["selected_alpha"] = alpha
    predictions["selected_target_minority_to_majority_ratio"] = policy.target_minority_to_majority_ratio
    predictions["selected_class_weight"] = policy.class_weight
    predictions["y_score"] = y_score

    training_metadata = {
        "outer_fold": int(outer_fold_id),
        "outer_train_rows": int(len(outer_train_frame)),
        "outer_test_rows": int(len(outer_test_frame)),
        "outer_train_projects": int(outer_train_frame[base_config.project_column].nunique()),
        "outer_test_projects": int(outer_test_frame[base_config.project_column].nunique()),
        "outer_project_overlap": 0,
        "selected_sampling_policy": policy.name,
        "selected_alpha": alpha,
        "selected_class_weight": policy.class_weight,
        **resampling_metadata,
        **vector_metadata,
        **fit_metadata,
        "outer_total_fold_seconds": total_seconds,
        "outer_score_min": float(np.min(y_score)),
        "outer_score_max": float(np.max(y_score)),
        "outer_score_mean": float(np.mean(y_score)),
    }
    _log(f"{label} | completed in {_format_seconds(total_seconds)}.", nested_config.verbose)
    del x_train, x_test, x_fit, y_fit, word_vectorizer, char_vectorizer
    gc.collect()
    return predictions, training_metadata, resampling_metadata


def _payload(base_config: Exp0Config, nested_config: NestedRosConfig) -> dict:
    return {
        "nested_ros_exp0_version": NESTED_ROS_EXP0_VERSION,
        "base_exp0_version": EXP0_VERSION,
        "base_config": asdict(base_config),
        "nested_config": asdict(nested_config),
        "scope": {
            "run_kind": "nested_ros_policy_tuning_development_only",
            "global_outer_holdout_used": False,
            "global_outer_holdout_scored": False,
            "selection_unit": "pooled inner project-grouped OOF PR-AUC for complete policy+alpha candidates",
            "main_study_status": "Supplementary development-only robustness study; does not replace the locked main MLP holdout result.",
        },
    }


def _prepare_output_dir(output_dir: Path | str, payload: dict, *, resume: bool) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    config_path = output_dir / "cs1_exp0_nested_ros_config.json"
    state_path = output_dir / "cs1_exp0_nested_ros_run_state.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise RuntimeError(
                f"Nested ROS output directory is not empty:\n{output_dir}\n\n"
                "Use a new directory, or resume only an interrupted run with identical configuration."
            )
        if not config_path.is_file():
            raise RuntimeError(f"Cannot resume: missing configuration artifact:\n{config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        if saved != payload:
            raise RuntimeError("Cannot resume: current configuration differs from saved configuration.")
        if state_path.is_file():
            with state_path.open("r", encoding="utf-8") as file:
                state = json.load(file)
            if state.get("status") == "completed":
                raise RuntimeError("This nested ROS run is complete. Do not overwrite it.")
        return output_dir, state_path

    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=_json_default)
    with state_path.open("w", encoding="utf-8") as file:
        json.dump({
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "completed_outer_folds": [],
            "note": "Only development rows may be passed to this study. The global outer holdout is never scored.",
        }, file, indent=2)
    return output_dir, state_path


def _checkpoint_dir(output_dir: Path) -> Path:
    root = output_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _checkpoint_paths(output_dir: Path, outer_fold_id: int) -> dict[str, Path]:
    root = _checkpoint_dir(output_dir)
    prefix = f"outer_fold_{outer_fold_id}"
    return {
        "predictions": root / f"{prefix}_predictions.parquet",
        "candidate_scores": root / f"{prefix}_inner_candidate_scores.csv",
        "selected": root / f"{prefix}_selected_candidate.json",
        "split_audit": root / f"{prefix}_inner_split_audit.csv",
        "resampling_audit": root / f"{prefix}_resampling_audit.csv",
        "outer_training": root / f"{prefix}_outer_training.json",
    }


def _write_checkpoint(
    output_dir: Path,
    outer_fold_id: int,
    predictions: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    selected: dict,
    split_audit: pd.DataFrame,
    resampling_audit: pd.DataFrame,
    outer_training: dict,
) -> None:
    paths = _checkpoint_paths(output_dir, outer_fold_id)
    predictions.to_parquet(paths["predictions"], index=False)
    candidate_scores.to_csv(paths["candidate_scores"], index=False)
    split_audit.to_csv(paths["split_audit"], index=False)
    resampling_audit.to_csv(paths["resampling_audit"], index=False)
    with paths["selected"].open("w", encoding="utf-8") as file:
        json.dump(selected, file, indent=2, default=_json_default)
    with paths["outer_training"].open("w", encoding="utf-8") as file:
        json.dump(outer_training, file, indent=2, default=_json_default)


def _load_checkpoint(output_dir: Path, outer_fold_id: int) -> Optional[dict]:
    paths = _checkpoint_paths(output_dir, outer_fold_id)
    if not any(path.exists() for path in paths.values()):
        return None
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError(f"Incomplete checkpoint for outer fold {outer_fold_id}; inspect before resume.")
    predictions = pd.read_parquet(paths["predictions"])
    if predictions["source_row_id"].duplicated().any():
        raise RuntimeError(f"Duplicate source IDs in checkpoint for fold {outer_fold_id}.")
    if predictions["fold"].astype(int).nunique() != 1 or int(predictions["fold"].iloc[0]) != outer_fold_id:
        raise RuntimeError(f"Checkpoint fold identity mismatch for fold {outer_fold_id}.")
    with paths["selected"].open("r", encoding="utf-8") as file:
        selected = json.load(file)
    with paths["outer_training"].open("r", encoding="utf-8") as file:
        outer_training = json.load(file)
    return {
        "predictions": predictions,
        "candidate_scores": pd.read_csv(paths["candidate_scores"]),
        "selected": selected,
        "split_audit": pd.read_csv(paths["split_audit"]),
        "resampling_audit": pd.read_csv(paths["resampling_audit"]),
        "outer_training": outer_training,
    }


def _update_state(state_path: Path, completed_folds: Sequence[int], status: str) -> None:
    with state_path.open("w", encoding="utf-8") as file:
        json.dump({
            "status": status,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed_outer_folds": sorted(int(fold) for fold in completed_folds),
            "note": "Only development rows may be passed to this study. The global outer holdout is never scored.",
        }, file, indent=2)


def run_exp0_nested_ros_inner_profile(
    development_frame: pd.DataFrame,
    development_manifest: pd.DataFrame,
    outer_fold_id: int = 4,
    base_config: Exp0Config = Exp0Config(verbose=False),
    nested_config: NestedRosConfig = NestedRosConfig(),
) -> dict:
    """Run inner candidate selection for one outer-development training split only.

    The profile never trains/scorers a selected model on the corresponding outer
    test fold. It is a computational and protocol check, not an official result.
    """
    _validate_nested_config(base_config, nested_config)
    if outer_fold_id not in range(base_config.n_splits):
        raise ValueError(f"outer_fold_id must be in [0, {base_config.n_splits - 1}].")
    prepared = _prepare_dataset_with_folds(development_frame, development_manifest, base_config)
    outer_train = prepared.loc[prepared[base_config.fold_column] != outer_fold_id].reset_index(drop=True)
    start = time.perf_counter()
    raw_scores, summary, split_audit, resampling_audit, selected = _run_inner_selection(
        outer_train, outer_fold_id, base_config, nested_config
    )
    duration = float(time.perf_counter() - start)
    _log(
        "Nested ROS inner profile complete. No outer-development test fold and no global outer holdout was scored.",
        nested_config.verbose,
    )
    return {
        "outer_fold_id": int(outer_fold_id),
        "inner_candidate_scores": raw_scores,
        "candidate_summary": summary,
        "inner_split_audit": split_audit,
        "resampling_audit": resampling_audit,
        "selected_candidate": selected,
        "total_profile_seconds": duration,
    }


def run_exp0_nested_ros(
    development_frame: pd.DataFrame,
    development_manifest: pd.DataFrame,
    base_config: Exp0Config = Exp0Config(verbose=False),
    nested_config: NestedRosConfig = NestedRosConfig(),
    output_dir: Optional[Path | str] = None,
    resume: bool = False,
    additional_metadata: Optional[Mapping[str, object]] = None,
) -> dict:
    """Run the complete development-only nested policy+alpha study."""
    _validate_nested_config(base_config, nested_config)
    prepared = _prepare_dataset_with_folds(development_frame, development_manifest, base_config)
    # Calling notebook must supply development rows only. Assert the manifest has no unexpected row coverage.
    if len(prepared) != len(development_frame):
        raise RuntimeError("Development-frame/manifest coverage mismatch.")

    payload = _payload(base_config, nested_config)
    output_path: Optional[Path] = None
    state_path: Optional[Path] = None
    if output_dir is not None:
        output_path, state_path = _prepare_output_dir(output_dir, payload, resume=resume)

    all_predictions: list[pd.DataFrame] = []
    all_candidate_scores: list[pd.DataFrame] = []
    all_selected: list[dict] = []
    all_split_audits: list[pd.DataFrame] = []
    all_resampling_audits: list[pd.DataFrame] = []
    all_outer_training: list[dict] = []
    completed: list[int] = []
    start = time.perf_counter()

    try:
        for outer_fold_id in range(base_config.n_splits):
            if output_path is not None and resume:
                checkpoint = _load_checkpoint(output_path, outer_fold_id)
                if checkpoint is not None:
                    _log(f"Outer development fold {outer_fold_id + 1}/{base_config.n_splits} | loading completed checkpoint.", nested_config.verbose)
                    all_predictions.append(checkpoint["predictions"])
                    all_candidate_scores.append(checkpoint["candidate_scores"])
                    all_selected.append(checkpoint["selected"])
                    all_split_audits.append(checkpoint["split_audit"])
                    all_resampling_audits.append(checkpoint["resampling_audit"])
                    all_outer_training.append(checkpoint["outer_training"])
                    completed.append(outer_fold_id)
                    continue

            outer_train = prepared.loc[prepared[base_config.fold_column] != outer_fold_id].reset_index(drop=True)
            outer_test = prepared.loc[prepared[base_config.fold_column] == outer_fold_id].reset_index(drop=True)
            raw_scores, summary, split_audit, resampling_audit, selection = _run_inner_selection(
                outer_train, outer_fold_id, base_config, nested_config
            )
            selected_policy = str(selection["selected_sampling_policy"])
            selected_alpha = float(selection["selected_alpha"])
            outer_predictions, outer_training, outer_resampling = _fit_outer_selected_model(
                outer_train, outer_test, outer_fold_id, selection, base_config, nested_config
            )
            selected_row = {
                "outer_fold": int(outer_fold_id),
                **selection,
                "alpha_grid": ";".join(f"{alpha:g}" for alpha in nested_config.alpha_grid),
                "sampling_policies": ";".join(policy.name for policy in nested_config.sampling_policies),
                "inner_n_splits": int(nested_config.inner_n_splits),
            }
            # Explicit consistency check: outer resampling policy must match selected inner policy.
            if outer_resampling["sampling_policy"] != selected_policy:
                raise RuntimeError("Outer resampling policy does not match the selected policy.")
            if float(outer_training["selected_alpha"]) != selected_alpha:
                raise RuntimeError("Outer alpha does not match selected alpha.")

            if output_path is not None:
                _write_checkpoint(
                    output_path, outer_fold_id, outer_predictions, raw_scores,
                    selected_row, split_audit, resampling_audit, outer_training
                )
            all_predictions.append(outer_predictions)
            all_candidate_scores.append(raw_scores)
            all_selected.append(selected_row)
            all_split_audits.append(split_audit)
            all_resampling_audits.append(resampling_audit)
            all_outer_training.append(outer_training)
            completed.append(outer_fold_id)
            if state_path is not None:
                _update_state(state_path, completed, status="running")
            del outer_train, outer_test
            gc.collect()

        oof_predictions = pd.concat(all_predictions, ignore_index=True).sort_values("source_row_id", kind="stable").reset_index(drop=True)
        if len(oof_predictions) != len(prepared):
            raise RuntimeError(f"Nested ROS OOF coverage mismatch: {len(oof_predictions):,} vs {len(prepared):,}.")
        if oof_predictions["source_row_id"].duplicated().any():
            raise RuntimeError("Nested ROS OOF predictions contain duplicate source IDs.")
        if set(oof_predictions["fold"].astype(int).unique()) != set(range(base_config.n_splits)):
            raise RuntimeError("Nested ROS OOF predictions do not cover all frozen development folds.")

        candidate_scores = pd.concat(all_candidate_scores, ignore_index=True).sort_values(
            ["outer_fold", "policy_grid_order", "alpha_grid_order", "inner_fold"], kind="stable"
        ).reset_index(drop=True)
        selected_candidates = pd.DataFrame(all_selected).sort_values("outer_fold", kind="stable").reset_index(drop=True)
        split_audit = pd.concat(all_split_audits, ignore_index=True).sort_values(["outer_fold", "inner_fold"], kind="stable").reset_index(drop=True)
        resampling_audit = pd.concat(all_resampling_audits, ignore_index=True).sort_values(
            ["outer_fold", "inner_fold", "policy_grid_order"], kind="stable"
        ).reset_index(drop=True)
        outer_training = pd.DataFrame(all_outer_training).sort_values("outer_fold", kind="stable").reset_index(drop=True)
        evaluation = evaluate_oof_predictions(
            oof_predictions,
            config=EvaluationConfig(threshold=base_config.decision_threshold, expected_n_folds=base_config.n_splits),
        )
        total_runtime = float(time.perf_counter() - start)
        results = {
            "oof_predictions": evaluation["predictions"],
            "inner_candidate_scores": candidate_scores,
            "selected_candidate": selected_candidates,
            "inner_split_audit": split_audit,
            "resampling_audit": resampling_audit,
            "outer_fold_training": outer_training,
            "evaluation": evaluation,
            "total_runtime_seconds": total_runtime,
        }
        if output_path is not None:
            artifacts = _save_artifacts(results, output_path, base_config, nested_config, additional_metadata)
            results["artifacts"] = artifacts
            if state_path is not None:
                _update_state(state_path, completed, status="completed")
        _log(f"Nested EXP-0 ROS study completed in {_format_seconds(total_runtime)}.", nested_config.verbose)
        return results
    except Exception:
        if state_path is not None:
            _update_state(state_path, completed, status="interrupted")
        raise


def _save_artifacts(
    results: Mapping[str, object],
    output_dir: Path,
    base_config: Exp0Config,
    nested_config: NestedRosConfig,
    additional_metadata: Optional[Mapping[str, object]],
) -> NestedRosArtifacts:
    oof = results["oof_predictions"]
    scores = results["inner_candidate_scores"]
    selected = results["selected_candidate"]
    split_audit = results["inner_split_audit"]
    resampling_audit = results["resampling_audit"]
    outer_training = results["outer_fold_training"]
    evaluation = results["evaluation"]
    for name, value in {
        "oof_predictions": oof,
        "inner_candidate_scores": scores,
        "selected_candidate": selected,
        "inner_split_audit": split_audit,
        "resampling_audit": resampling_audit,
        "outer_fold_training": outer_training,
    }.items():
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame.")
    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping.")

    safe = nested_config.experiment_name
    nested_config_json = output_dir / f"{safe}_nested_config.json"
    candidate_scores_csv = output_dir / f"{safe}_inner_candidate_scores.csv"
    selected_csv = output_dir / f"{safe}_selected_candidate_per_outer_fold.csv"
    split_audit_csv = output_dir / f"{safe}_inner_split_audit.csv"
    resampling_audit_csv = output_dir / f"{safe}_resampling_audit.csv"
    outer_training_csv = output_dir / f"{safe}_outer_fold_training.csv"
    run_metadata_json = output_dir / f"{safe}_run_metadata.json"
    run_state_json = output_dir / "cs1_exp0_nested_ros_run_state.json"

    with nested_config_json.open("w", encoding="utf-8") as file:
        json.dump(_payload(base_config, nested_config), file, indent=2, default=_json_default)
    scores.to_csv(candidate_scores_csv, index=False)
    selected.to_csv(selected_csv, index=False)
    split_audit.to_csv(split_audit_csv, index=False)
    resampling_audit.to_csv(resampling_audit_csv, index=False)
    outer_training.to_csv(outer_training_csv, index=False)

    evaluation_paths = save_evaluation_artifacts(
        evaluation=evaluation,
        output_dir=output_dir,
        experiment_name=nested_config.experiment_name,
        config=EvaluationConfig(threshold=base_config.decision_threshold, expected_n_folds=base_config.n_splits),
        additional_metadata={
            "nested_ros_exp0_version": NESTED_ROS_EXP0_VERSION,
            "base_exp0_version": EXP0_VERSION,
            "run_kind": "nested_ros_policy_tuning_development_only",
            "global_outer_holdout_used": False,
            "global_outer_holdout_scored": False,
            "selection_procedure": (
                "Within each frozen outer-development training partition, select the complete "
                "sampling-policy + alpha candidate by pooled project-grouped inner OOF PR-AUC."
            ),
            "oversampling_note": (
                "Only real vulnerable training rows are duplicated after TF-IDF fitting; "
                "no synthetic SMOTE feature vectors are created."
            ),
            "score_interpretation": "Risk scores are not calibrated real-world probabilities.",
            **dict(additional_metadata or {}),
        },
    )
    metadata = {
        "nested_ros_exp0_version": NESTED_ROS_EXP0_VERSION,
        "base_exp0_version": EXP0_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": nested_config.experiment_name,
        "base_config": asdict(base_config),
        "nested_config": asdict(nested_config),
        "n_outer_oof_predictions": int(len(oof)),
        "outer_folds_completed": sorted(oof["fold"].astype(int).unique().tolist()),
        "selected_candidate_by_outer_fold": {
            str(int(row.outer_fold)): {
                "policy": str(row.selected_sampling_policy),
                "alpha": float(row.selected_alpha),
            }
            for row in selected.itertuples(index=False)
        },
        "total_inner_model_fit_seconds": float(scores["model_fit_seconds"].sum()),
        "total_inner_prediction_seconds": float(scores["prediction_seconds"].sum()),
        "total_inner_vectorization_seconds": float(split_audit["vectorization_seconds"].sum()),
        "total_outer_model_fit_seconds": float(outer_training["model_fit_seconds"].sum()),
        "total_outer_vectorization_seconds": float(outer_training["vectorization_seconds"].sum()),
        "total_runtime_seconds": float(results["total_runtime_seconds"]),
        "global_outer_holdout_used": False,
        "scope_limit": (
            "Supplementary post-selection development-only study. It cannot replace the locked "
            "main EXP-2 holdout result or reopen final model selection."
        ),
        "additional_metadata": dict(additional_metadata or {}),
    }
    with run_metadata_json.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=_json_default)
    return NestedRosArtifacts(
        nested_config_json=nested_config_json,
        inner_candidate_scores_csv=candidate_scores_csv,
        selected_candidate_csv=selected_csv,
        inner_split_audit_csv=split_audit_csv,
        resampling_audit_csv=resampling_audit_csv,
        outer_fold_training_csv=outer_training_csv,
        run_metadata_json=run_metadata_json,
        run_state_json=run_state_json,
        evaluation_paths=evaluation_paths,
    )


__all__ = [
    "NESTED_ROS_EXP0_VERSION",
    "SamplingPolicy",
    "DEFAULT_SAMPLING_POLICIES",
    "NestedRosConfig",
    "NestedRosArtifacts",
    "run_exp0_nested_ros_inner_profile",
    "run_exp0_nested_ros",
]
