"""Project-block bootstrap confidence intervals for pooled predictions.

Rows in this project are not independent: they are grouped by `project`.
A row-level bootstrap would violate the same non-independence assumption
that the grouped CV / holdout split was built to protect, so every
interval here resamples whole projects with replacement, never rows.

Design decisions (see project discussion, do not relax without re-reading it):
- Threshold-independent metrics (PR-AUC, ROC-AUC) are computed directly on
  y_score. Threshold-dependent metrics (precision, recall, F1) are also
  supported, but ONLY at a single, explicitly fixed decision threshold
  (`threshold`, default 0.5 -- the same fixed threshold used elsewhere in
  this project's reporting, see PDD Section 7.5). Reporting these without a
  CI is misleading given the project-block resampling variance already
  observed for PR-AUC; unlike PR-AUC/ROC-AUC, their CI additionally reflects
  the arbitrariness of that fixed threshold choice, not sampling variance
  alone -- callers should not treat a wide precision/recall/F1 interval as
  equivalent evidence to a PR-AUC interval of the same width.
- Percentile method, not normal-approximation: the bootstrap distribution
  of PR-AUC under a rare positive class can be skewed.
- Degenerate resamples (no positive class, or all-positive) are dropped
  and counted rather than silently ignored.
- These intervals quantify sampling variability *within this dataset
  only*. They are not an estimate of generalization uncertainty to C
  functions outside this collection, and for nested-CV pooled
  predictions they do not capture the extra variability from re-running
  the outer split with a different seed (Bengio & Grandvalet, 2004).
- Comparing two experiments by eye-balling overlap between two separately
  computed marginal CIs is not a valid test. Use paired_bootstrap_metric_ci
  instead, which resamples the same projects for both experiments and
  builds the interval on the difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CI_MODULE_VERSION = "cs1-confidence-intervals-v2"

REQUIRED_CI_COLUMNS = ("project", "label", "y_score")

# Threshold-independent: computed directly on the continuous y_score.
SUPPORTED_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "average_precision_pr_auc": average_precision_score,
    "roc_auc": roc_auc_score,
}

# Threshold-dependent: y_score is first binarized at `threshold` (see
# _resolve_metric_fn). zero_division=0 avoids sklearn warnings/NaNs on
# degenerate bootstrap resamples with zero predicted positives.
THRESHOLD_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
}

ALL_METRICS = tuple(sorted(set(SUPPORTED_METRICS) | set(THRESHOLD_METRICS)))


def _resolve_metric_fn(metric: str, threshold: float) -> Callable[[np.ndarray, np.ndarray], float]:
    """Returns a (y_true, y_score) -> float callable for `metric`.

    Threshold-dependent metrics are wrapped to binarize y_score at
    `threshold` first, so every entry in SUPPORTED_METRICS/THRESHOLD_METRICS
    can be called uniformly as metric_fn(y_true, y_score) everywhere below.
    """
    if metric in SUPPORTED_METRICS:
        return SUPPORTED_METRICS[metric]
    if metric in THRESHOLD_METRICS:
        thresholded_fn = THRESHOLD_METRICS[metric]
        return lambda y_true, y_score: thresholded_fn(y_true, (np.asarray(y_score) >= threshold).astype(int))
    raise ValueError(f"Unsupported metric '{metric}'. Supported: {list(ALL_METRICS)}.")


def _require_ci_columns(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CI_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s) for confidence intervals: {missing}. "
            f"Available columns: {sorted(frame.columns.tolist())}"
        )


def _resample_rows_for_groups(
    predictions: pd.DataFrame,
    sampled_groups: np.ndarray,
    group_column: str,
) -> pd.DataFrame:
    grouped = predictions.groupby(group_column, sort=False)
    parts = [grouped.get_group(g) for g in sampled_groups]
    return pd.concat(parts, ignore_index=True)


@dataclass(frozen=True)
class BootstrapCIResult:
    metric_name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence: float
    n_bootstrap_requested: int
    n_bootstrap_valid: int
    n_degenerate_dropped: int
    n_groups: int
    random_state: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def bootstrap_metric_ci(
    predictions: pd.DataFrame,
    metric: str = "average_precision_pr_auc",
    group_column: str = "project",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
    min_positive_count: int = 1,
    threshold: float = 0.5,
) -> BootstrapCIResult:
    """Project-block bootstrap CI for one metric (threshold-independent or not).

    `threshold` only applies to threshold-dependent metrics (precision,
    recall, f1); ignored for average_precision_pr_auc / roc_auc.
    """
    _require_ci_columns(predictions)
    metric_fn = _resolve_metric_fn(metric, threshold)

    point_estimate = float(metric_fn(predictions["label"], predictions["y_score"]))

    groups = predictions[group_column].unique()
    n_groups = len(groups)

    rng = np.random.default_rng(random_state)
    scores: list[float] = []
    n_degenerate = 0
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(groups, size=n_groups, replace=True)
        resample = _resample_rows_for_groups(predictions, sampled_groups, group_column)
        y_true = resample["label"].to_numpy()
        if y_true.sum() < min_positive_count or y_true.sum() == len(y_true):
            n_degenerate += 1
            continue
        scores.append(float(metric_fn(y_true, resample["y_score"].to_numpy())))

    if not scores:
        raise RuntimeError(
            "All bootstrap resamples were degenerate (no valid positive/negative split). "
            "Increase n_bootstrap or check class balance per project."
        )

    lo_pct = (1 - confidence) / 2 * 100
    hi_pct = (1 + confidence) / 2 * 100

    return BootstrapCIResult(
        metric_name=metric,
        point_estimate=point_estimate,
        ci_low=float(np.percentile(scores, lo_pct)),
        ci_high=float(np.percentile(scores, hi_pct)),
        confidence=confidence,
        n_bootstrap_requested=n_bootstrap,
        n_bootstrap_valid=len(scores),
        n_degenerate_dropped=n_degenerate,
        n_groups=n_groups,
        random_state=random_state,
    )


@dataclass(frozen=True)
class PairedBootstrapCIResult:
    metric_name: str
    experiment_a: str
    experiment_b: str
    point_estimate_a: float
    point_estimate_b: float
    point_estimate_diff: float
    ci_low_diff: float
    ci_high_diff: float
    confidence: float
    n_bootstrap_requested: int
    n_bootstrap_valid: int
    n_degenerate_dropped: int
    n_groups: int
    random_state: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def paired_bootstrap_metric_ci(
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    experiment_name_a: str,
    experiment_name_b: str,
    metric: str = "average_precision_pr_auc",
    group_column: str = "project",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
    min_positive_count: int = 1,
    threshold: float = 0.5,
) -> PairedBootstrapCIResult:
    """Paired project-block bootstrap CI on the difference between two experiments.

    Both prediction frames must cover the same partition (identical set of
    projects), since the same resampled groups are used for both sides.
    `threshold` only applies to threshold-dependent metrics (precision,
    recall, f1); ignored for average_precision_pr_auc / roc_auc.
    """
    _require_ci_columns(predictions_a)
    _require_ci_columns(predictions_b)
    metric_fn = _resolve_metric_fn(metric, threshold)

    groups_a = set(predictions_a["project"].unique())
    groups_b = set(predictions_b["project"].unique())
    if groups_a != groups_b:
        raise ValueError(
            "Paired bootstrap requires both experiments to share exactly the same "
            "set of projects (same partition). "
            f"Only in A: {sorted(groups_a - groups_b)[:5]}, "
            f"only in B: {sorted(groups_b - groups_a)[:5]}."
        )

    point_a = float(metric_fn(predictions_a["label"], predictions_a["y_score"]))
    point_b = float(metric_fn(predictions_b["label"], predictions_b["y_score"]))

    groups = np.array(sorted(groups_a))
    n_groups = len(groups)

    rng = np.random.default_rng(random_state)
    diffs: list[float] = []
    n_degenerate = 0
    grouped_a = predictions_a.groupby(group_column, sort=False)
    grouped_b = predictions_b.groupby(group_column, sort=False)
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(groups, size=n_groups, replace=True)
        resample_a = pd.concat([grouped_a.get_group(g) for g in sampled_groups], ignore_index=True)
        resample_b = pd.concat([grouped_b.get_group(g) for g in sampled_groups], ignore_index=True)

        y_true_a = resample_a["label"].to_numpy()
        y_true_b = resample_b["label"].to_numpy()
        if (
            y_true_a.sum() < min_positive_count
            or y_true_a.sum() == len(y_true_a)
            or y_true_b.sum() < min_positive_count
            or y_true_b.sum() == len(y_true_b)
        ):
            n_degenerate += 1
            continue

        score_a = float(metric_fn(y_true_a, resample_a["y_score"].to_numpy()))
        score_b = float(metric_fn(y_true_b, resample_b["y_score"].to_numpy()))
        diffs.append(score_a - score_b)

    if not diffs:
        raise RuntimeError(
            "All paired bootstrap resamples were degenerate. Increase n_bootstrap "
            "or check class balance per project."
        )

    lo_pct = (1 - confidence) / 2 * 100
    hi_pct = (1 + confidence) / 2 * 100

    return PairedBootstrapCIResult(
        metric_name=metric,
        experiment_a=experiment_name_a,
        experiment_b=experiment_name_b,
        point_estimate_a=point_a,
        point_estimate_b=point_b,
        point_estimate_diff=point_a - point_b,
        ci_low_diff=float(np.percentile(diffs, lo_pct)),
        ci_high_diff=float(np.percentile(diffs, hi_pct)),
        confidence=confidence,
        n_bootstrap_requested=n_bootstrap,
        n_bootstrap_valid=len(diffs),
        n_degenerate_dropped=n_degenerate,
        n_groups=n_groups,
        random_state=random_state,
    )


def format_ci_report(result: BootstrapCIResult) -> str:
    lines = [
        f"{result.metric_name}: point estimate = {result.point_estimate:.4f}",
        f"  {int(result.confidence * 100)}% CI (project-block bootstrap): "
        f"[{result.ci_low:.4f}, {result.ci_high:.4f}]",
        f"  valid resamples: {result.n_bootstrap_valid}/{result.n_bootstrap_requested} "
        f"({result.n_degenerate_dropped} degenerate, dropped)",
        f"  n_projects: {result.n_groups}, random_state={result.random_state}",
        "  Reflects sampling variability within this dataset only; not an "
        "estimate of generalization to C functions outside this collection.",
    ]
    return "\n".join(lines)


def format_paired_ci_report(result: PairedBootstrapCIResult) -> str:
    lines = [
        f"{result.metric_name}: {result.experiment_a} = {result.point_estimate_a:.4f}, "
        f"{result.experiment_b} = {result.point_estimate_b:.4f}",
        f"  Difference ({result.experiment_a} - {result.experiment_b}) = "
        f"{result.point_estimate_diff:.4f}",
        f"  {int(result.confidence * 100)}% CI on the difference (paired project-block "
        f"bootstrap): [{result.ci_low_diff:.4f}, {result.ci_high_diff:.4f}]",
        f"  valid resamples: {result.n_bootstrap_valid}/{result.n_bootstrap_requested} "
        f"({result.n_degenerate_dropped} degenerate, dropped)",
        "  An interval excluding zero means the difference is unlikely to be "
        "bootstrap noise within this dataset; it says nothing about generalization "
        "beyond it.",
    ]
    return "\n".join(lines)