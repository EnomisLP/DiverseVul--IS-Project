"""
5-fold stratified cross-validation training loop for Track A.

Wires together:
  src.data          → load dataset, load split indices, generate CV folds
  src.track_a.features → TFIDFPipeline (fitted inside each fold)
  src.track_a.models   → build_model dispatcher (LR or RF)
  src.evaluate         → evaluate_model, aggregate_fold_metrics
  src.utils            → set_seed, load_config, generate_exp_id, save_metrics

Enforced constraints:
  - Holdout indices are loaded but NEVER used during training or validation.
    They are passed through only so the caller can run holdout evaluation
    after all CV folds complete.
  - TF-IDF is fitted fresh on each fold's training split — never on val data.
  - Predicted probabilities (not hard labels) are passed to evaluate_model
    so PR-AUC can be computed over the full score range.

Public API:
  - run_experiment(exp_id, cfg, df, train_idx) -> (fold_results, agg)
  - run_all_track_a(cfg, df, train_idx)        -> dict of exp_id → agg metrics
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.data import COL_FUNC, COL_TARGET, get_cv_folds, get_texts_and_labels
from src.evaluate import (
    EvaluationResult,
    aggregate_fold_metrics,
    evaluate_model,
    plot_all_folds_pr_curves,
)
from src.track_a.features import TFIDFPipeline
from src.track_a.models import build_model
from src.utils import (
    generate_exp_id,
    get_logger,
    load_config,
    save_metrics,
    set_seed,
)

_log = get_logger(__name__)

# Where per-fold JSON files land (committed to git)
_METRICS_DIR = Path("results/metrics")
# Where PR curve PNGs land (committed to git)
_FIGURES_DIR = Path("results/figures")


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    exp_id: str,
    cfg,
    df,
    train_idx: np.ndarray,
    *,
    metrics_dir: Union[str, Path] = _METRICS_DIR,
    figures_dir: Union[str, Path] = _FIGURES_DIR,
    threshold: float = 0.5,
    save_pr_curves: bool = True,
    overwrite: bool = False,
) -> Tuple[List[EvaluationResult], Dict]:
    """Run one Track A experiment (EXP_0 or EXP_1) with 5-fold CV.

    For each fold:
      1. Extract train / val texts and labels using *train_idx* only.
      2. Fit TF-IDF on fold training split.
      3. Transform both splits (no leakage — vectorizer never sees val data).
      4. Fit the sklearn estimator on training features.
      5. Predict probabilities on validation features.
      6. Evaluate and persist per-fold metrics JSON.

    After all folds, aggregate results and optionally save PR curves.

    Args:
        exp_id:       ``"EXP_0"`` (Logistic Regression) or ``"EXP_1"``
                      (Random Forest).
        cfg:          Config namespace from
                      ``load_config("configs/track_a.yaml")``.
        df:           Full dataset DataFrame (all rows, both splits).
        train_idx:    Integer index array for the training partition only
                      (from :func:`src.data.load_holdout_split`).
        metrics_dir:  Directory for per-fold JSON files.
        figures_dir:  Directory for PR curve PNGs.
        threshold:    Decision threshold for precision/recall/F1 (default 0.5).
        save_pr_curves: Whether to save a combined PR curve PNG (default True).
        overwrite:    Whether to overwrite existing metrics files.

    Returns:
        Tuple of:
          - ``fold_results`` — list of :class:`~src.evaluate.EvaluationResult`
          - ``agg``          — aggregated metrics dict
            ``{"mean": {...}, "std": {...}, "per_fold": [...]}``
    """
    _log.info("=" * 60)
    _log.info("Starting %s  (%s)", exp_id, _model_label(exp_id))
    _log.info("=" * 60)

    set_seed(cfg.cv.random_state)

    labels = df[COL_TARGET].to_numpy(dtype=np.int64)
    fold_results: List[EvaluationResult] = []
    fold_data_for_plot: List[Dict] = []       # accumulates (y_true, y_scores) per fold

    for fold, (fold_train_idx, fold_val_idx) in enumerate(
        get_cv_folds(
            train_idx,
            labels,
            n_splits=cfg.cv.n_splits,
            random_state=cfg.cv.random_state,
        )
    ):
        _log.info("--- Fold %d / %d ---", fold + 1, cfg.cv.n_splits)

        # 1. Extract raw texts and labels
        X_train_raw, y_train = get_texts_and_labels(df, fold_train_idx)
        X_val_raw,   y_val   = get_texts_and_labels(df, fold_val_idx)

        # 2+3. TF-IDF: fit on train only, transform both
        pipeline = TFIDFPipeline.from_config(cfg)
        X_train_feat = pipeline.fit_transform(X_train_raw)
        X_val_feat   = pipeline.transform(X_val_raw)

        # 4. Build and fit model (fresh instance each fold)
        model = build_model(exp_id, cfg)
        model.fit(X_train_feat, y_train)

        # 5. Predict probabilities for positive class
        y_scores = _predict_proba(model, X_val_feat)

        # 6. Evaluate + save per-fold JSON
        result = evaluate_model(
            y_val,
            y_scores,
            fold=fold,
            exp_id=exp_id,
            output_dir=metrics_dir,
            threshold=threshold,
            overwrite=overwrite,
        )
        fold_results.append(result)

        fold_data_for_plot.append({
            "fold":     fold,
            "y_true":   y_val.tolist(),
            "y_scores": y_scores.tolist(),
        })

    # Aggregate across folds
    agg = aggregate_fold_metrics(fold_results)

    # Save aggregate metrics JSON
    agg_path = Path(metrics_dir) / f"{exp_id}_aggregate.json"
    save_metrics(agg, agg_path, overwrite=overwrite)

    # PR curves
    if save_pr_curves:
        curve_path = Path(figures_dir) / f"{exp_id}_pr_curves.png"
        try:
            plot_all_folds_pr_curves(
                fold_data_for_plot,
                exp_id=exp_id,
                output_path=curve_path,
            )
        except ImportError:
            _log.warning("matplotlib not available — skipping PR curve plot.")

    _log.info(
        "%s complete — PR-AUC %.4f ± %.4f | F1 %.4f ± %.4f",
        exp_id,
        agg["mean"]["pr_auc"], agg["std"]["pr_auc"],
        agg["mean"]["f1"],     agg["std"]["f1"],
    )

    return fold_results, agg


# ---------------------------------------------------------------------------
# Run all Track A experiments
# ---------------------------------------------------------------------------

def run_all_track_a(
    cfg,
    df,
    train_idx: np.ndarray,
    *,
    metrics_dir: Union[str, Path] = _METRICS_DIR,
    figures_dir: Union[str, Path] = _FIGURES_DIR,
    threshold: float = 0.5,
    save_pr_curves: bool = True,
    overwrite: bool = False,
) -> Dict[str, Dict]:
    """Run EXP_0 and EXP_1 sequentially and return all aggregated metrics.

    Convenience wrapper for ``01_track_a.ipynb`` — calls
    :func:`run_experiment` for each Track A experiment ID.

    Args:
        cfg:          Config namespace from
                      ``load_config("configs/track_a.yaml")``.
        df:           Full dataset DataFrame.
        train_idx:    Training partition indices from
                      :func:`src.data.load_holdout_split`.
        metrics_dir:  Directory for JSON output.
        figures_dir:  Directory for PNG output.
        threshold:    Decision threshold (default 0.5).
        save_pr_curves: Save combined PR curve PNGs (default True).
        overwrite:    Overwrite existing output files.

    Returns:
        Dict mapping experiment ID → aggregated metrics dict::

            {
                "EXP_0": {"mean": {...}, "std": {...}, "per_fold": [...]},
                "EXP_1": {"mean": {...}, "std": {...}, "per_fold": [...]},
            }
    """
    results: Dict[str, Dict] = {}

    for exp_id in ("EXP_0", "EXP_1"):
        _, agg = run_experiment(
            exp_id,
            cfg,
            df,
            train_idx,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
            threshold=threshold,
            save_pr_curves=save_pr_curves,
            overwrite=overwrite,
        )
        results[exp_id] = agg

    _log.info("Track A complete.")
    _log.info(
        "EXP_0 PR-AUC: %.4f ± %.4f",
        results["EXP_0"]["mean"]["pr_auc"],
        results["EXP_0"]["std"]["pr_auc"],
    )
    _log.info(
        "EXP_1 PR-AUC: %.4f ± %.4f",
        results["EXP_1"]["mean"]["pr_auc"],
        results["EXP_1"]["std"]["pr_auc"],
    )

    return results


# ---------------------------------------------------------------------------
# Holdout evaluation (called from 05_results.ipynb, not during CV)
# ---------------------------------------------------------------------------

def evaluate_on_holdout(
    exp_id: str,
    cfg,
    df,
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    *,
    metrics_dir: Union[str, Path] = _METRICS_DIR,
    figures_dir: Union[str, Path] = _FIGURES_DIR,
    threshold: float = 0.5,
    overwrite: bool = False,
) -> EvaluationResult:
    """Retrain on the full training partition and evaluate on the holdout set.

    This is called **once**, after all CV experiments are complete and the
    best configuration is selected.  The holdout set is never seen during CV.

    Retraining uses the full ``train_idx`` partition (all 5 folds combined),
    with TF-IDF fitted on that entire training set.

    Args:
        exp_id:       Experiment to evaluate (``"EXP_0"`` or ``"EXP_1"``).
        cfg:          Track A config namespace.
        df:           Full dataset DataFrame.
        train_idx:    Training partition indices.
        holdout_idx:  Holdout partition indices (used here for the first time).
        metrics_dir:  Directory for JSON output.
        figures_dir:  Directory for PNG output.
        threshold:    Decision threshold.
        overwrite:    Overwrite existing output files.

    Returns:
        :class:`~src.evaluate.EvaluationResult` for the holdout set.
    """
    _log.info("Holdout evaluation — %s", exp_id)

    set_seed(cfg.cv.random_state)

    X_train_raw, y_train = get_texts_and_labels(df, train_idx)
    X_hold_raw,  y_hold  = get_texts_and_labels(df, holdout_idx)

    # TF-IDF fitted on full training partition
    pipeline = TFIDFPipeline.from_config(cfg)
    X_train_feat = pipeline.fit_transform(X_train_raw)
    X_hold_feat  = pipeline.transform(X_hold_raw)

    model = build_model(exp_id, cfg)
    model.fit(X_train_feat, y_train)

    y_scores = _predict_proba(model, X_hold_feat)

    result = evaluate_model(
        y_hold,
        y_scores,
        fold=-1,       # -1 signals holdout
        exp_id=exp_id,
        output_dir=metrics_dir,
        threshold=threshold,
        overwrite=overwrite,
    )

    # Save holdout PR curve
    curve_path = Path(figures_dir) / f"{exp_id}_holdout_pr_curve.png"
    try:
        from src.evaluate import plot_pr_curve
        plot_pr_curve(
            y_hold,
            y_scores,
            exp_id=exp_id,
            fold=None,
            output_path=curve_path,
        )
    except ImportError:
        _log.warning("matplotlib not available — skipping holdout PR curve.")

    _log.info("Holdout %s — PR-AUC %.4f | F1 %.4f", exp_id, result.pr_auc, result.f1)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _predict_proba(model, X) -> np.ndarray:
    """Return positive-class probabilities from any sklearn estimator.

    Uses ``predict_proba`` when available (LR, RF).  Falls back to
    ``decision_function`` with min-max normalisation for estimators that
    only expose a score (not used in Track A, but kept for robustness).

    Args:
        model: Fitted sklearn estimator.
        X:     Feature matrix (sparse or dense).

    Returns:
        1-D float64 array of shape ``(n_samples,)``.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # proba shape: (n_samples, n_classes); column 1 = positive class
        return proba[:, 1].astype(np.float64)

    # fallback — should not be reached for LR or RF
    scores = model.decision_function(X).astype(np.float64)
    min_s, max_s = scores.min(), scores.max()
    if max_s > min_s:
        scores = (scores - min_s) / (max_s - min_s)
    return scores


def _model_label(exp_id: str) -> str:
    """Human-readable model name for logging."""
    return {
        "EXP_0": "Logistic Regression + TF-IDF",
        "EXP_1": "Random Forest + TF-IDF",
    }.get(exp_id, exp_id)