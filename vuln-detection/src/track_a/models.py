"""
Scikit-learn model definitions for Track A.

EXP_0 — Logistic Regression on TF-IDF features
EXP_1 — Random Forest on TF-IDF features

Both estimators read their hyperparameters directly from the config namespace
returned by src.utils.load_config("configs/track_a.yaml").

Class imbalance is handled at the estimator level:
  - LogisticRegression : class_weight="balanced"
  - RandomForest       : class_weight="balanced_subsample"

Public API:
  - build_logistic_regression(cfg) -> LogisticRegression
  - build_random_forest(cfg)       -> RandomForestClassifier
  - build_model(exp_id, cfg)       -> dispatches by experiment ID string
"""

from __future__ import annotations

from typing import Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# EXP_0 — Logistic Regression
# ---------------------------------------------------------------------------

def build_logistic_regression(cfg) -> LogisticRegression:
    """Build a LogisticRegression estimator from the Track A config.

    Reads from ``cfg.logistic_regression.*``.  All hyperparameters are passed
    directly to sklearn — no defaults are overridden silently.

    Args:
        cfg: Config namespace from ``load_config("configs/track_a.yaml")``.

    Returns:
        Un-fitted :class:`sklearn.linear_model.LogisticRegression`.
    """
    lr_cfg = cfg.logistic_regression

    model = LogisticRegression(
        penalty=lr_cfg.penalty,
        C=lr_cfg.C,
        solver=lr_cfg.solver,
        max_iter=lr_cfg.max_iter,
        class_weight=lr_cfg.class_weight,
        tol=lr_cfg.tol,
        random_state=lr_cfg.random_state,
        n_jobs=-1,
    )

    _log.info(
        "Built LogisticRegression — penalty=%s  C=%.4f  solver=%s  "
        "class_weight=%s  max_iter=%d",
        lr_cfg.penalty, lr_cfg.C, lr_cfg.solver,
        lr_cfg.class_weight, lr_cfg.max_iter,
    )
    return model


# ---------------------------------------------------------------------------
# EXP_1 — Random Forest
# ---------------------------------------------------------------------------

def build_random_forest(cfg) -> RandomForestClassifier:
    """Build a RandomForestClassifier estimator from the Track A config.

    Reads from ``cfg.random_forest.*``.  ``max_depth: null`` in YAML is
    loaded as Python ``None`` by PyYAML, which sklearn interprets as
    "expand until all leaves are pure" — this is the intended behaviour.

    Args:
        cfg: Config namespace from ``load_config("configs/track_a.yaml")``.

    Returns:
        Un-fitted :class:`sklearn.ensemble.RandomForestClassifier`.
    """
    rf_cfg = cfg.random_forest

    model = RandomForestClassifier(
        n_estimators=rf_cfg.n_estimators,
        max_features=rf_cfg.max_features,
        max_depth=rf_cfg.max_depth,       # None → fully grown trees
        class_weight=rf_cfg.class_weight,
        criterion=rf_cfg.criterion,
        random_state=rf_cfg.random_state,
        n_jobs=-1,
    )

    _log.info(
        "Built RandomForestClassifier — n_estimators=%d  max_features=%s  "
        "max_depth=%s  class_weight=%s  criterion=%s",
        rf_cfg.n_estimators, rf_cfg.max_features,
        rf_cfg.max_depth, rf_cfg.class_weight, rf_cfg.criterion,
    )
    return model


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_BUILDERS = {
    "EXP_0": build_logistic_regression,
    "EXP_1": build_random_forest,
}


def build_model(
    exp_id: str,
    cfg,
) -> Union[LogisticRegression, RandomForestClassifier]:
    """Dispatch to the correct model builder by experiment ID.

    Args:
        exp_id: One of ``"EXP_0"`` or ``"EXP_1"``.
        cfg:    Config namespace from ``load_config("configs/track_a.yaml")``.

    Returns:
        Un-fitted sklearn estimator for the requested experiment.

    Raises:
        ValueError: If *exp_id* is not a recognised Track A experiment.

    Example::

        cfg   = load_config("configs/track_a.yaml")
        model = build_model("EXP_0", cfg)   # LogisticRegression
        model = build_model("EXP_1", cfg)   # RandomForestClassifier
    """
    if exp_id not in _BUILDERS:
        raise ValueError(
            f"Unknown experiment ID '{exp_id}' for Track A. "
            f"Valid options: {sorted(_BUILDERS.keys())}"
        )
    return _BUILDERS[exp_id](cfg)