"""
Reusable evaluation utilities shared by every experiment (CS1-EXP1, CS1-EXP2, CS2-EXP3, CS2-EXP4).

Evaluates out-of-fold (OOF) binary vulnerability predictions collected across
the 5 rotating outer folds. Expected prediction schema:

    source_row_id : immutable ID from the canonical dataset
    fold          : held-out test fold ID
    label         : ground-truth label, 0=non-vulnerable and 1=vulnerable
    y_score       : predicted probability/risk score for label 1 in [0, 1]

Optional columns such as ``project`` and ``experiment`` are preserved in
exports. The evaluator creates ``y_pred`` using the configured threshold.

Important methodological rule
-----------------------------
Outer test predictions must NOT be used to choose a threshold; every
experiment selects its decision threshold from inner-CV validation scores
only (see ``select_f1_threshold``), never from the outer test fold.

Headline result: mean and standard deviation of each metric across the 5
outer folds (``summarize_fold_metrics``). The pooled-OOF numbers are kept as
a secondary, single-number cross-check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)


EVALUATION_VERSION = "cs1-evaluation-v1"

REQUIRED_PREDICTION_COLUMNS = (
    "source_row_id",
    "fold",
    "label",
    "y_score",
)

DEFAULT_THRESHOLD_GRID = tuple(np.round(np.linspace(0.01, 0.99, 99), 2).tolist())


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration shared by Case Study 1 evaluation calls."""

    threshold: float = 0.50
    positive_label: int = 1
    expected_n_folds: int = 5
    threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID


@dataclass(frozen=True)
class EvaluationPaths:
    """Paths returned after saving one experiment's evaluation artifacts."""

    predictions_parquet: Path
    predictions_csv: Path
    pooled_metrics_json: Path
    fold_metrics_csv: Path
    fold_summary_csv: Path
    threshold_metrics_csv: Path
    pr_curve_png: Path
    pooled_confusion_matrix_png: Path
    metadata_json: Path


def _json_default(value: object) -> object:
    """Convert NumPy/Pandas scalar values into JSON-serializable Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a clear error listing any missing required column."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {sorted(frame.columns.tolist())}"
        )


def _validate_threshold(threshold: float) -> float:
    """Validate that a threshold lies strictly between 0 and 1."""
    threshold = float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"threshold must lie strictly between 0 and 1; received {threshold}."
        )
    return threshold


def _validate_prediction_frame(
    predictions: pd.DataFrame,
    config: EvaluationConfig,
) -> pd.DataFrame:
    """
    Validate and standardize OOF predictions before scoring.

    This intentionally fails loudly on duplicate IDs, missing values, invalid
    probabilities, labels outside {0,1}, or unexpected outer-fold IDs.
    """
    _require_columns(predictions, REQUIRED_PREDICTION_COLUMNS)

    frame = predictions.copy()

    if frame.empty:
        raise ValueError("Prediction dataframe is empty.")

    if frame["source_row_id"].isna().any():
        raise ValueError("source_row_id contains missing values.")

    if frame["source_row_id"].duplicated().any():
        duplicate_count = int(frame["source_row_id"].duplicated().sum())
        raise ValueError(
            "Each OOF sample must appear exactly once. "
            f"Found {duplicate_count:,} duplicate source_row_id value(s)."
        )

    for column in ("fold", "label", "y_score"):
        if frame[column].isna().any():
            missing_count = int(frame[column].isna().sum())
            raise ValueError(f"{column} contains {missing_count:,} missing value(s).")

    labels = pd.to_numeric(frame["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        observed = sorted(pd.unique(labels).tolist())
        raise ValueError(
            "Binary vulnerability labels must be 0 or 1. "
            f"Observed values: {observed[:20]}"
        )
    frame["label"] = labels.astype("int8")

    if frame["label"].nunique() != 2:
        raise ValueError("Predictions must contain both label classes.")

    folds = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    expected_folds = set(range(config.expected_n_folds))
    observed_folds = set(folds.unique().tolist())
    if observed_folds != expected_folds:
        raise ValueError(
            f"Expected outer fold IDs {sorted(expected_folds)}; "
            f"observed {sorted(observed_folds)}."
        )
    frame["fold"] = folds.astype("int8")

    scores = pd.to_numeric(frame["y_score"], errors="raise").astype(float)
    if not np.isfinite(scores).all():
        raise ValueError("y_score contains NaN or infinite values.")
    if (scores < 0.0).any() or (scores > 1.0).any():
        score_min = float(scores.min())
        score_max = float(scores.max())
        raise ValueError(
            "y_score must be a probability/risk score in [0, 1]. "
            f"Observed range: [{score_min:.6f}, {score_max:.6f}]"
        )
    frame["y_score"] = scores

    _validate_threshold(config.threshold)

    return frame


def _binary_predictions(scores: Sequence[float], threshold: float) -> np.ndarray:
    """Convert positive-class probabilities into binary decisions."""
    threshold = _validate_threshold(threshold)
    return (np.asarray(scores, dtype=float) >= threshold).astype(np.int8)


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Divide, returning 0.0 instead of raising on a zero denominator."""
    return float(numerator / denominator) if denominator else 0.0


def compute_binary_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.50,
) -> dict:
    """
    Compute classification metrics for one binary prediction collection.

    ``average_precision`` is the standard Average Precision summary of the
    precision-recall curve and is reported as PR-AUC / AP in project outputs.
    """
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=float)

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty label/score collection.")
    if len(y_true) != len(y_score):
        raise ValueError("labels and scores must have equal length.")
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("Metrics require both classes 0 and 1.")
    if not np.isfinite(y_score).all():
        raise ValueError("Scores contain NaN or infinite values.")
    if (y_score < 0.0).any() or (y_score > 1.0).any():
        raise ValueError("Scores must lie in [0, 1].")

    threshold = _validate_threshold(threshold)
    y_pred = _binary_predictions(y_score, threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    specificity = _safe_divide(tn, tn + fp)
    npv = _safe_divide(tn, tn + fn)
    false_positive_rate = _safe_divide(fp, fp + tn)
    false_negative_rate = _safe_divide(fn, fn + tp)
    accuracy = _safe_divide(tp + tn, len(y_true))
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "n_samples": int(len(y_true)),
        "vulnerable_1": int((y_true == 1).sum()),
        "non_vulnerable_0": int((y_true == 0).sum()),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "average_precision_pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "specificity": float(specificity),
        "negative_predictive_value": float(npv),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_positive": int(y_pred.sum()),
        "predicted_positive_rate": float(y_pred.mean()),
    }


DEFAULT_VALIDATION_THRESHOLD_GRID = tuple(np.round(np.arange(0.01, 0.99 + 1e-9, 0.01), 2).tolist())


def select_f1_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold_grid: Sequence[float] = DEFAULT_VALIDATION_THRESHOLD_GRID,
) -> tuple[float, dict]:
    """Select the F1-maximizing threshold from validation-only scores; ties broken by precision, then fewer alerts."""
    best: Optional[tuple[tuple[float, float, int], dict]] = None
    for threshold in threshold_grid:
        metrics = compute_binary_metrics(labels=labels, scores=scores, threshold=float(threshold))
        key = (metrics["f1"], metrics["precision"], -metrics["predicted_positive"])
        if best is None or key > best[0]:
            best = (key, metrics)

    assert best is not None
    selected = dict(best[1])
    selected["selection_objective"] = "f1"
    selected["candidate_count"] = int(len(threshold_grid))
    selected["candidate_min"] = float(min(threshold_grid))
    selected["candidate_max"] = float(max(threshold_grid))
    selected["selected_at_grid_boundary"] = bool(
        np.isclose(selected["threshold"], selected["candidate_min"])
        or np.isclose(selected["threshold"], selected["candidate_max"])
    )
    return float(selected["threshold"]), selected


def build_threshold_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """
    Calculate decision metrics at a predefined list of thresholds.

    Use the resulting table for diagnostics and triage discussion. Do not choose
    a threshold based on held-out outer-test predictions.
    """
    rows: list[dict] = []
    for threshold in thresholds:
        metric_row = compute_binary_metrics(labels, scores, threshold=float(threshold))
        rows.append(metric_row)

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def _fold_metric_rows(
    prediction_frame: pd.DataFrame,
    config: EvaluationConfig,
) -> pd.DataFrame:
    """Compute metrics separately for each held-out outer fold."""
    rows: list[dict] = []

    for fold_id in range(config.expected_n_folds):
        fold_frame = prediction_frame.loc[prediction_frame["fold"] == fold_id]
        fold_metrics = compute_binary_metrics(
            labels=fold_frame["label"],
            scores=fold_frame["y_score"],
            threshold=config.threshold,
        )
        fold_metrics["fold"] = int(fold_id)

        if "project" in fold_frame.columns:
            fold_metrics["test_unique_projects"] = int(
                fold_frame["project"].astype(str).nunique()
            )

        rows.append(fold_metrics)

    columns = ["fold"] + [column for column in rows[0] if column != "fold"]
    return pd.DataFrame(rows)[columns].sort_values("fold").reset_index(drop=True)


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Produce unweighted mean, standard deviation, minimum, and maximum across folds.

    The values are intentionally unweighted: this describes fold-to-fold
    stability. The pooled OOF metrics remain the primary aggregate result.
    """
    if fold_metrics.empty:
        raise ValueError("fold_metrics is empty.")

    numeric_columns = fold_metrics.select_dtypes(include=[np.number]).columns.tolist()
    excluded_columns = {
        "fold",
        "n_samples",
        "vulnerable_1",
        "non_vulnerable_0",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "predicted_positive",
        "test_unique_projects",
    }
    metric_columns = [column for column in numeric_columns if column not in excluded_columns]

    rows: list[dict] = []
    for metric in metric_columns:
        values = fold_metrics[metric].astype(float)
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def evaluate_oof_predictions(
    predictions: pd.DataFrame,
    config: EvaluationConfig = EvaluationConfig(),
) -> dict:
    """
    Evaluate a complete collection of 5-fold out-of-fold predictions.

    Returns a dictionary containing:
    - predictions: validated OOF predictions plus y_pred
    - pooled_metrics: one pooled result over all held-out samples
    - fold_metrics: one row per outer fold
    - fold_summary: mean/std/min/max per metric across folds
    - threshold_metrics: diagnostic table across predefined thresholds
    """
    prediction_frame = _validate_prediction_frame(predictions, config=config)
    prediction_frame["y_pred"] = _binary_predictions(
        prediction_frame["y_score"],
        config.threshold,
    )

    pooled_metrics = compute_binary_metrics(
        labels=prediction_frame["label"],
        scores=prediction_frame["y_score"],
        threshold=config.threshold,
    )

    fold_metrics = _fold_metric_rows(prediction_frame, config=config)
    fold_summary = summarize_fold_metrics(fold_metrics)
    threshold_metrics = build_threshold_metrics(
        labels=prediction_frame["label"],
        scores=prediction_frame["y_score"],
        thresholds=config.threshold_grid,
    )

    return {
        "predictions": prediction_frame,
        "pooled_metrics": pooled_metrics,
        "fold_metrics": fold_metrics,
        "fold_summary": fold_summary,
        "threshold_metrics": threshold_metrics,
    }


def plot_precision_recall_curve(
    labels: Sequence[int],
    scores: Sequence[float],
    output_path: Path | str,
    title: str = "Precision-Recall Curve",
) -> Path:
    """Save a precision-recall curve using Matplotlib defaults."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=float)
    ap = average_precision_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"Average Precision = {ap:.4f}")
    plt.axhline(
        y=float(y_true.mean()),
        linestyle="--",
        label=f"Random baseline = {float(y_true.mean()):.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    return output_path


def plot_confusion_matrix(
    labels: Sequence[int],
    predictions: Sequence[int],
    output_path: Path | str,
    title: str = "Confusion Matrix",
) -> Path:
    """Save a pooled binary confusion-matrix figure using Matplotlib defaults."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(labels, dtype=np.int8)
    y_pred = np.asarray(predictions, dtype=np.int8)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    image = plt.imshow(matrix)
    plt.colorbar(image)
    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    plt.xlabel("Prediction")
    plt.ylabel("Ground truth")
    plt.title(title)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            plt.text(
                column_index,
                row_index,
                str(int(matrix[row_index, column_index])),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    return output_path


def save_evaluation_artifacts(
    evaluation: Mapping[str, object],
    output_dir: Path | str,
    experiment_name: str,
    config: EvaluationConfig = EvaluationConfig(),
    additional_metadata: Optional[Mapping[str, object]] = None,
) -> EvaluationPaths:
    """
    Persist all report-ready outputs from ``evaluate_oof_predictions``.

    Saved files:
    - oof_predictions.parquet and .csv
    - pooled_metrics.json
    - fold_metrics.csv
    - fold_summary.csv
    - threshold_metrics.csv
    - precision_recall_curve.png
    - pooled_confusion_matrix.png
    - evaluation_metadata.json
    """
    required_keys = {
        "predictions",
        "pooled_metrics",
        "fold_metrics",
        "fold_summary",
        "threshold_metrics",
    }
    missing_keys = required_keys - set(evaluation.keys())
    if missing_keys:
        raise KeyError(f"Evaluation object is missing keys: {sorted(missing_keys)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = evaluation["predictions"]
    pooled_metrics = evaluation["pooled_metrics"]
    fold_metrics = evaluation["fold_metrics"]
    fold_summary = evaluation["fold_summary"]
    threshold_metrics = evaluation["threshold_metrics"]

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("evaluation['predictions'] must be a pandas DataFrame.")
    if not isinstance(fold_metrics, pd.DataFrame):
        raise TypeError("evaluation['fold_metrics'] must be a pandas DataFrame.")
    if not isinstance(fold_summary, pd.DataFrame):
        raise TypeError("evaluation['fold_summary'] must be a pandas DataFrame.")
    if not isinstance(threshold_metrics, pd.DataFrame):
        raise TypeError("evaluation['threshold_metrics'] must be a pandas DataFrame.")
    if not isinstance(pooled_metrics, Mapping):
        raise TypeError("evaluation['pooled_metrics'] must be a mapping.")

    safe_name = experiment_name.strip().lower().replace(" ", "_")
    safe_name = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in safe_name
    )

    predictions_parquet = output_dir / f"{safe_name}_oof_predictions.parquet"
    predictions_csv = output_dir / f"{safe_name}_oof_predictions.csv"
    pooled_metrics_json = output_dir / f"{safe_name}_pooled_metrics.json"
    fold_metrics_csv = output_dir / f"{safe_name}_fold_metrics.csv"
    fold_summary_csv = output_dir / f"{safe_name}_fold_summary.csv"
    threshold_metrics_csv = output_dir / f"{safe_name}_threshold_metrics.csv"
    pr_curve_png = output_dir / f"{safe_name}_precision_recall_curve.png"
    pooled_confusion_matrix_png = output_dir / f"{safe_name}_pooled_confusion_matrix.png"
    metadata_json = output_dir / f"{safe_name}_evaluation_metadata.json"

    predictions.to_parquet(predictions_parquet, index=False)
    predictions.to_csv(predictions_csv, index=False)
    fold_metrics.to_csv(fold_metrics_csv, index=False)
    fold_summary.to_csv(fold_summary_csv, index=False)
    threshold_metrics.to_csv(threshold_metrics_csv, index=False)

    with pooled_metrics_json.open("w", encoding="utf-8") as file:
        json.dump(dict(pooled_metrics), file, indent=2, default=_json_default)

    plot_precision_recall_curve(
        labels=predictions["label"],
        scores=predictions["y_score"],
        output_path=pr_curve_png,
        title=f"{experiment_name}: Pooled OOF Precision-Recall Curve",
    )
    plot_confusion_matrix(
        labels=predictions["label"],
        predictions=predictions["y_pred"],
        output_path=pooled_confusion_matrix_png,
        title=f"{experiment_name}: Pooled OOF Confusion Matrix",
    )

    metadata = {
        "evaluation_version": EVALUATION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "config": asdict(config),
        "primary_aggregation": (
            "Pooled out-of-fold predictions: every sample is evaluated once "
            "by the model trained without that sample's held-out project fold."
        ),
        "fold_summary_aggregation": (
            "Unweighted mean and standard deviation across outer folds; "
            "used as a stability diagnostic, not as the primary score."
        ),
        "threshold_note": (
            "Threshold table is descriptive. Do not select a deployment "
            "threshold using outer test predictions."
        ),
        "n_oof_predictions": int(len(predictions)),
        "additional_metadata": dict(additional_metadata or {}),
    }

    with metadata_json.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=_json_default)

    return EvaluationPaths(
        predictions_parquet=predictions_parquet,
        predictions_csv=predictions_csv,
        pooled_metrics_json=pooled_metrics_json,
        fold_metrics_csv=fold_metrics_csv,
        fold_summary_csv=fold_summary_csv,
        threshold_metrics_csv=threshold_metrics_csv,
        pr_curve_png=pr_curve_png,
        pooled_confusion_matrix_png=pooled_confusion_matrix_png,
        metadata_json=metadata_json,
    )


def format_metric_report(pooled_metrics: Mapping[str, object]) -> str:
    """Format the core pooled OOF metrics for notebook output."""
    keys = [
        "n_samples",
        "vulnerable_1",
        "non_vulnerable_0",
        "positive_rate",
        "threshold",
        "average_precision_pr_auc",
        "precision",
        "recall",
        "f1",
        "mcc",
        "specificity",
        "false_positive_rate",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]

    lines = ["Pooled Out-of-Fold Evaluation"]
    lines.append("=" * 44)

    for key in keys:
        if key not in pooled_metrics:
            continue

        value = pooled_metrics[key]
        if key in {
            "positive_rate",
            "threshold",
            "average_precision_pr_auc",
            "precision",
            "recall",
            "f1",
            "mcc",
            "specificity",
            "false_positive_rate",
        }:
            lines.append(f"{key:>28}: {float(value):.6f}")
        else:
            lines.append(f"{key:>28}: {value}")

    return "\n".join(lines)


__all__ = [
    "DEFAULT_THRESHOLD_GRID",
    "DEFAULT_VALIDATION_THRESHOLD_GRID",
    "EVALUATION_VERSION",
    "EvaluationConfig",
    "EvaluationPaths",
    "build_threshold_metrics",
    "compute_binary_metrics",
    "evaluate_oof_predictions",
    "format_metric_report",
    "plot_confusion_matrix",
    "plot_precision_recall_curve",
    "save_evaluation_artifacts",
    "select_f1_threshold",
    "summarize_fold_metrics",
]
