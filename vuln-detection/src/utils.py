"""
Shared utilities for the Vulnerability Detection System.

Provides:
  - set_seed           : reproducible seeding across random / numpy / torch
  - load_config        : YAML config loader returning a SimpleNamespace tree
  - compute_class_weights : inverse-frequency weights for imbalanced labels
  - save_metrics       : persist per-fold / aggregate dicts to JSON
  - generate_exp_id    : deterministic experiment identifier string
  - get_logger         : pre-configured stdlib logger
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout with a standard format.

    Args:
        name:  Logger name (typically __name__ of the calling module).
        level: Logging level (default INFO).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed all relevant random-number generators for reproducibility.

    Covers Python ``random``, NumPy, and PyTorch (CPU + CUDA) when torch is
    available.  Also sets ``PYTHONHASHSEED`` and enables cuDNN determinism.

    Args:
        seed: Integer seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        _log.debug("PyTorch seeded (seed=%d, CUDA available=%s).", seed, torch.cuda.is_available())
    except ImportError:
        _log.debug("PyTorch not available — skipping torch seed.")

    _log.info("Global seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a nested dict to a :class:`~types.SimpleNamespace`."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def load_config(path: Union[str, Path]) -> SimpleNamespace:
    """Load a YAML config file and return it as a dot-accessible namespace.

    Example::

        cfg = load_config("configs/track_a.yaml")
        print(cfg.tfidf.max_features)   # 50000
        print(cfg.cv.n_splits)          # 5

    Args:
        path: Path to the YAML file.

    Returns:
        Nested :class:`~types.SimpleNamespace` mirroring the YAML structure.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError:    If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(raw).__name__}.")

    cfg = _dict_to_namespace(raw)
    _log.info("Loaded config: %s", path)
    return cfg


# ---------------------------------------------------------------------------
# Class weight calculation
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: Sequence[int],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Compute inverse-frequency class weights for imbalanced classification.

    Uses the same formula as scikit-learn's ``class_weight='balanced'``::

        weight_c = n_samples / (n_classes * count_c)

    Args:
        labels:      1-D sequence of integer class labels (0-indexed).
        num_classes: Total number of classes.  If ``None``, inferred as
                     ``max(labels) + 1``.

    Returns:
        Float64 array of shape ``(num_classes,)`` where entry *c* is the
        weight for class *c*.

    Raises:
        ValueError: If ``labels`` is empty or contains negative values.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.ndim != 1 or len(labels_arr) == 0:
        raise ValueError("`labels` must be a non-empty 1-D sequence.")
    if labels_arr.min() < 0:
        raise ValueError("Labels must be non-negative integers.")

    if num_classes is None:
        num_classes = int(labels_arr.max()) + 1

    n_samples = len(labels_arr)
    weights = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        count_c = np.sum(labels_arr == c)
        if count_c == 0:
            _log.warning("Class %d has zero samples — weight set to 0.", c)
            weights[c] = 0.0
        else:
            weights[c] = n_samples / (num_classes * count_c)

    _log.info(
        "Class weights (n=%d, classes=%d): %s",
        n_samples,
        num_classes,
        {c: round(float(weights[c]), 4) for c in range(num_classes)},
    )
    return weights


def class_weights_as_tensor(
    labels: Sequence[int],
    num_classes: Optional[int] = None,
    device: str = "cpu",
):
    """Return class weights as a ``torch.Tensor``.

    Convenience wrapper around :func:`compute_class_weights` for use with
    ``torch.nn.CrossEntropyLoss(weight=...)``.

    Args:
        labels:      1-D sequence of integer class labels.
        num_classes: Total number of classes (inferred if ``None``).
        device:      Torch device string (``"cpu"``, ``"cuda"``).

    Returns:
        ``torch.FloatTensor`` of shape ``(num_classes,)`` on *device*.
    """
    import torch  # deferred — torch not required for Track A

    weights = compute_class_weights(labels, num_classes=num_classes)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Metrics persistence
# ---------------------------------------------------------------------------

def save_metrics(
    metrics: Dict,
    output_path: Union[str, Path],
    *,
    indent: int = 2,
    overwrite: bool = False,
) -> Path:
    """Serialise a metrics dict to a JSON file.

    Handles numpy scalars / arrays by converting them to plain Python types
    before serialisation so ``json.dump`` never raises ``TypeError``.

    Args:
        metrics:     Dict (possibly nested) of metric names → values.
        output_path: Destination file path.  Parent directories are created
                     automatically.
        indent:      JSON indentation width (default 2).
        overwrite:   If ``False`` (default) and the file already exists, a
                     ``FileExistsError`` is raised.

    Returns:
        Resolved :class:`~pathlib.Path` of the written file.

    Raises:
        FileExistsError: If the file exists and *overwrite* is ``False``.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Metrics file already exists: {output_path}. "
            "Pass overwrite=True to replace it."
        )

    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    serialisable = _convert(metrics)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=indent)

    _log.info("Metrics saved → %s", output_path)
    return output_path


def load_metrics(path: Union[str, Path]) -> Dict:
    """Load a previously saved metrics JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Dict of metrics.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Experiment ID generation
# ---------------------------------------------------------------------------

def generate_exp_id(
    exp_name: str,
    *,
    include_timestamp: bool = True,
    extra: Optional[Dict] = None,
) -> str:
    """Generate a human-readable, sortable experiment identifier.

    Format::

        <exp_name>_<YYYYMMDD-HHMMSS>[_<key>-<val>...]

    Examples::

        generate_exp_id("EXP_3")
        # → "EXP_3_20260628-153012"

        generate_exp_id("EXP_3", extra={"lora_r": 16, "fold": 2})
        # → "EXP_3_20260628-153012_lora_r-16_fold-2"

    Args:
        exp_name:          Short experiment name (e.g. ``"EXP_3"``).
        include_timestamp: Append a ``YYYYMMDD-HHMMSS`` timestamp (default ``True``).
        extra:             Optional dict of additional key→value tags.

    Returns:
        Experiment ID string safe for use as a filename.
    """
    parts: List[str] = [exp_name]

    if include_timestamp:
        ts = time.strftime("%Y%m%d-%H%M%S")
        parts.append(ts)

    if extra:
        for k, v in extra.items():
            parts.append(f"{k}-{v}")

    exp_id = "_".join(parts)
    _log.debug("Generated experiment ID: %s", exp_id)
    return exp_id