"""
Shared evaluation utilities for the Vulnerability Detection System.

Provides:
  - compute_metrics      : precision / recall / F1 / PR-AUC for one fold
  - aggregate_fold_metrics : mean ± std across all CV folds
  - EvaluationResult     : lightweight dataclass wrapping a fold's scores
  - plot_pr_curve        : save a publication-quality PR curve to disk
  - evaluate_model       : convenience wrapper used by both Track A and Track B
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.utils import get_logger, save_metrics

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Holds all scalar evaluation metrics for a single fold or run.

    Attributes:
        fold:       Fold index (0-based).  ``-1`` for holdout / aggregate.
        exp_id:     Experiment identifier string (e.g. ``"EXP_3"``).
        precision:  Precision at the default decision threshold (0.5).
        recall:     Recall at the default decision threshold.
        f1:         Binary F1-score at the default threshold.
        pr_auc:     Area under the precision-recall curve (primary metric).
        threshold:  Decision threshold used.  Default 0.5.
        extra:      Dict for any additional task-specific scalars.
    """

    fold: int
    exp_id: str
    precision: float
    recall: float
    f1: float
    pr_auc: float
    threshold: float = 0.5
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[{self.exp_id} | fold={self.fold}] "
            f"P={self.precision:.4f}  R={self.recall:.4f}  "
            f"F1={self.f1:.4f}  PR-AUC={self.pr_auc:.4f}"
        )


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    *,
    threshold: float = 0.5,
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> Dict[str, float]:
    """Compute precision, recall, F1, and PR-AUC from predicted probabilities.

    Accuracy is intentionally excluded because the dataset is class-imbalanced;
    PR-AUC is the primary optimisation target throughout this project.

    Args:
        y_true:       Ground-truth binary labels (0 or 1).
        y_scores:     Predicted probabilities for the positive class.
        threshold:    Decision threshold applied to *y_scores* to obtain hard
                      predictions.  Default 0.5.
        pos_label:    Which class is considered "positive".  Default 1.
        zero_division: Value to use for ill-defined precision / recall / F1
                      (e.g. when a class has no predicted samples).

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``, ``pr_auc``,
        ``threshold``.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_scores_arr = np.asarray(y_scores, dtype=np.float64)

    if y_true_arr.shape != y_scores_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true_arr.shape}, y_scores={y_scores_arr.shape}"
        )
    if y_true_arr.ndim != 1:
        raise ValueError("y_true and y_scores must be 1-D arrays.")

    y_pred = (y_scores_arr >= threshold).astype(np.int64)

    precision = float(
        precision_score(
            y_true_arr, y_pred, pos_label=pos_label,
            zero_division=zero_division,
        )
    )
    recall = float(
        recall_score(
            y_true_arr, y_pred, pos_label=pos_label,
            zero_division=zero_division,
        )
    )
    f1 = float(
        f1_score(
            y_true_arr, y_pred, pos_label=pos_label,
            zero_division=zero_division,
        )
    )
    pr_auc = float(
        average_precision_score(y_true_arr, y_scores_arr, pos_label=pos_label)
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "threshold": threshold,
    }


def evaluate_fold(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    *,
    fold: int,
    exp_id: str,
    threshold: float = 0.5,
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> EvaluationResult:
    """Evaluate one CV fold and return an :class:`EvaluationResult`.

    This is the canonical entry point for both Track A and Track B training
    loops.

    Args:
        y_true:   Ground-truth labels for this fold's validation set.
        y_scores: Predicted positive-class probabilities.
        fold:     Fold index (0-based).
        exp_id:   Experiment identifier.
        threshold:    Decision threshold (default 0.5).
        pos_label:    Positive class label (default 1).
        zero_division: Fill value for ill-defined metrics.

    Returns:
        :class:`EvaluationResult` populated with all scalar metrics.
    """
    scores = compute_metrics(
        y_true, y_scores,
        threshold=threshold,
        pos_label=pos_label,
        zero_division=zero_division,
    )
    result = EvaluationResult(fold=fold, exp_id=exp_id, **scores)
    _log.info("%s", result)
    return result


# ---------------------------------------------------------------------------
# Fold aggregation
# ---------------------------------------------------------------------------

def aggregate_fold_metrics(
    fold_results: List[EvaluationResult],
) -> Dict[str, Dict[str, float]]:
    """Compute mean and standard deviation across all CV folds.

    Args:
        fold_results: List of :class:`EvaluationResult` objects, one per fold.

    Returns:
        Nested dict::

            {
                "mean": {"precision": ..., "recall": ..., "f1": ..., "pr_auc": ...},
                "std":  {"precision": ..., "recall": ..., "f1": ..., "pr_auc": ...},
                "per_fold": [{"fold": 0, "precision": ..., ...}, ...],
            }

    Raises:
        ValueError: If *fold_results* is empty.
    """
    if not fold_results:
        raise ValueError("fold_results must not be empty.")

    metric_keys = ["precision", "recall", "f1", "pr_auc"]
    arrays: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for r in fold_results:
        for k in metric_keys:
            arrays[k].append(getattr(r, k))

    mean = {k: float(np.mean(arrays[k])) for k in metric_keys}
    std  = {k: float(np.std(arrays[k], ddof=1)) for k in metric_keys}  # sample std

    _log.info(
        "[%s | %d folds] PR-AUC %.4f ± %.4f | F1 %.4f ± %.4f",
        fold_results[0].exp_id,
        len(fold_results),
        mean["pr_auc"], std["pr_auc"],
        mean["f1"],     std["f1"],
    )

    return {
        "mean":     mean,
        "std":      std,
        "per_fold": [r.to_dict() for r in fold_results],
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    *,
    fold: int,
    exp_id: str,
    output_dir: Optional[Union[str, Path]] = None,
    threshold: float = 0.5,
    overwrite: bool = False,
) -> EvaluationResult:
    """Evaluate one fold and optionally persist metrics to ``output_dir``.

    Designed to be called at the end of each fold iteration in both Track A
    and Track B training loops.

    Args:
        y_true:      Ground-truth labels.
        y_scores:    Predicted positive-class probabilities.
        fold:        Fold index.
        exp_id:      Experiment identifier.
        output_dir:  If provided, metrics JSON is written here as
                     ``<exp_id>_fold<fold>.json``.
        threshold:   Decision threshold (default 0.5).
        overwrite:   Whether to overwrite existing metrics file.

    Returns:
        :class:`EvaluationResult`.
    """
    result = evaluate_fold(
        y_true, y_scores,
        fold=fold,
        exp_id=exp_id,
        threshold=threshold,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        fname = f"{exp_id}_fold{fold}.json"
        save_metrics(result.to_dict(), output_dir / fname, overwrite=overwrite)

    return result


# ---------------------------------------------------------------------------
# PR curve plotting
# ---------------------------------------------------------------------------

def plot_pr_curve(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    *,
    exp_id: str,
    fold: Optional[int] = None,
    output_path: Union[str, Path],
    title: Optional[str] = None,
    pos_label: int = 1,
    dpi: int = 150,
) -> Path:
    """Save a precision-recall curve plot as a PNG file.

    Uses matplotlib with a clean style.  No matplotlib window is shown
    (backend set to ``Agg``) so this is safe to run in Colab / headless
    environments.

    Args:
        y_true:      Ground-truth labels.
        y_scores:    Predicted positive-class probabilities.
        exp_id:      Experiment identifier (used in the plot title).
        fold:        Fold index for the subtitle.  Pass ``None`` for holdout.
        output_path: Destination PNG file path.  Parent directories created
                     automatically.
        title:       Custom plot title.  If ``None``, auto-generated.
        pos_label:   Positive class label.
        dpi:         Output resolution (default 150).

    Returns:
        Resolved :class:`~pathlib.Path` of the saved PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_pr_curve. "
            "Install it with: pip install matplotlib"
        ) from exc

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_scores_arr = np.asarray(y_scores, dtype=np.float64)

    precision_vals, recall_vals, _ = precision_recall_curve(
        y_true_arr, y_scores_arr, pos_label=pos_label
    )
    pr_auc = average_precision_score(y_true_arr, y_scores_arr, pos_label=pos_label)

    # Baseline = class-imbalance ratio (fraction of positives)
    baseline = float(y_true_arr.mean())

    fold_label = f"Fold {fold}" if fold is not None else "Holdout"
    if title is None:
        title = f"Precision-Recall Curve — {exp_id} ({fold_label})"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_vals, precision_vals, lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.axhline(y=baseline, color="grey", linestyle="--", lw=1, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    _log.info("PR curve saved → %s", output_path)
    return output_path


def plot_all_folds_pr_curves(
    fold_data: List[Dict],
    *,
    exp_id: str,
    output_path: Union[str, Path],
    pos_label: int = 1,
    dpi: int = 150,
) -> Path:
    """Overlay PR curves from all CV folds on a single figure.

    Args:
        fold_data:   List of dicts, each with keys ``"y_true"``, ``"y_scores"``,
                     ``"fold"`` (int).
        exp_id:      Experiment identifier.
        output_path: Destination PNG file path.
        pos_label:   Positive class label.
        dpi:         Output resolution.

    Returns:
        Resolved :class:`~pathlib.Path` of the saved PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting.") from exc

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = cm.tab10.colors  # up to 10 distinct colours

    aucs: List[float] = []
    for entry in fold_data:
        yt = np.asarray(entry["y_true"], dtype=np.int64)
        ys = np.asarray(entry["y_scores"], dtype=np.float64)
        fold_idx = int(entry["fold"])

        precision_vals, recall_vals, _ = precision_recall_curve(yt, ys, pos_label=pos_label)
        auc_val = average_precision_score(yt, ys, pos_label=pos_label)
        aucs.append(auc_val)

        color = colors[fold_idx % len(colors)]
        ax.plot(recall_vals, precision_vals, lw=1.5, color=color,
                label=f"Fold {fold_idx} (AUC={auc_val:.4f})", alpha=0.85)

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs, ddof=1))

    baseline = float(np.concatenate([np.asarray(e["y_true"]) for e in fold_data]).mean())
    ax.axhline(y=baseline, color="grey", linestyle="--", lw=1, label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        f"PR Curves (all folds) — {exp_id}\n"
        f"Mean PR-AUC = {mean_auc:.4f} ± {std_auc:.4f}",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    _log.info("All-folds PR curve saved → %s", output_path)
    return output_path