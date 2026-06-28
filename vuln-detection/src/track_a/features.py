"""
TF-IDF feature extraction for Track A.

Design constraints:
  - The vectorizer is ALWAYS fitted on the training fold only, never on the
    full dataset or the validation/holdout split. This is enforced by the
    TFIDFPipeline.fit_transform / transform API: callers must call
    fit_transform on X_train, then transform on X_val / X_holdout.
  - Config is loaded from configs/track_a.yaml via src.utils.load_config.
  - Sparse matrices are returned as scipy.sparse.csr_matrix for compatibility
    with scikit-learn classifiers and the MLP (which converts to dense).

Public API:
  - TFIDFPipeline        : stateful wrapper (fit / transform / fit_transform)
  - build_tfidf_pipeline : factory that reads hyperparams from a config namespace
  - token_pattern        : custom regex that keeps C/C++ identifiers intact
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from src.utils import get_logger, load_config

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# C/C++ aware token pattern
# ---------------------------------------------------------------------------

# Keeps: identifiers (including underscores), numeric literals (hex/decimal),
# operator digraphs, and punctuation that carries semantic weight in C/C++.
# scikit-learn's default pattern strips underscores, which mangles identifiers
# like `__builtin_expect` or `NULL` into useless fragments.
_CPP_TOKEN_PATTERN = r"(?u)\b[a-zA-Z_]\w*\b|\b0x[0-9a-fA-F]+\b|\b\d+\b"


# ---------------------------------------------------------------------------
# TFIDFPipeline
# ---------------------------------------------------------------------------

class TFIDFPipeline:
    """Stateful TF-IDF feature pipeline for C/C++ source code.

    Wraps a :class:`sklearn.feature_extraction.text.TfidfVectorizer` and an
    optional :class:`sklearn.preprocessing.MaxAbsScaler` (to normalise sparse
    matrices in-place without breaking sparsity).

    Usage inside a CV fold::

        pipeline = TFIDFPipeline.from_config(cfg)
        X_train_feat = pipeline.fit_transform(X_train_raw)
        X_val_feat   = pipeline.transform(X_val_raw)

    Attributes:
        vectorizer: The underlying :class:`TfidfVectorizer`.
        scaler:     :class:`MaxAbsScaler` applied after vectorization (optional).
        fitted:     Whether the pipeline has been fitted.
    """

    def __init__(
        self,
        *,
        ngram_range: Tuple[int, int] = (1, 3),
        analyzer: str = "word",
        max_features: int = 50_000,
        sublinear_tf: bool = True,
        token_pattern: Optional[str] = _CPP_TOKEN_PATTERN,
        scale: bool = False,
    ) -> None:
        """Initialise the pipeline with vectorizer hyperparameters.

        Args:
            ngram_range:   Min and max n-gram sizes (default ``(1, 3)``).
            analyzer:      Feature type: ``"word"`` or ``"char"``
                           (default ``"word"``).
            max_features:  Vocabulary size cap (default 50 000).
            sublinear_tf:  Apply log(1 + tf) term-frequency scaling
                           (default ``True``).
            token_pattern: Regex for token splitting.  Defaults to the
                           C/C++-aware pattern that preserves underscored
                           identifiers.  Pass ``None`` to use sklearn default.
            scale:         If ``True``, apply :class:`MaxAbsScaler` after
                           vectorization (rarely needed; TF-IDF already
                           produces L2-normalised rows).
        """
        vec_kwargs = dict(
            ngram_range=ngram_range,
            analyzer=analyzer,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            strip_accents="unicode",
            norm="l2",
        )
        if analyzer == "word" and token_pattern is not None:
            vec_kwargs["token_pattern"] = token_pattern

        self.vectorizer = TfidfVectorizer(**vec_kwargs)
        self.scaler: Optional[MaxAbsScaler] = MaxAbsScaler() if scale else None
        self.fitted: bool = False

        _log.debug(
            "TFIDFPipeline created: ngram=%s max_feat=%d sublinear_tf=%s scale=%s",
            ngram_range, max_features, sublinear_tf, scale,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg,
        *,
        scale: bool = False,
    ) -> "TFIDFPipeline":
        """Build a :class:`TFIDFPipeline` from a loaded config namespace.

        Reads ``cfg.tfidf.ngram_range``, ``cfg.tfidf.analyzer``,
        ``cfg.tfidf.max_features``, and ``cfg.tfidf.sublinear_tf``.

        Args:
            cfg:   Config namespace returned by :func:`src.utils.load_config`
                   (typically ``configs/track_a.yaml``).
            scale: Whether to include :class:`MaxAbsScaler`.

        Returns:
            Configured :class:`TFIDFPipeline` instance.
        """
        tfidf_cfg = cfg.tfidf
        ngram_range = tuple(tfidf_cfg.ngram_range)  # YAML list → tuple

        return cls(
            ngram_range=ngram_range,
            analyzer=tfidf_cfg.analyzer,
            max_features=tfidf_cfg.max_features,
            sublinear_tf=tfidf_cfg.sublinear_tf,
            scale=scale,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: Sequence[str],
        y=None,
    ) -> sp.csr_matrix:
        """Fit the vectorizer on *X* and return the transformed features.

        Must be called on the **training fold only**.  Calling this on the
        full dataset before the CV split would introduce data leakage.

        Args:
            X: Iterable of raw C/C++ function strings (training fold).
            y: Ignored — present for sklearn Pipeline compatibility.

        Returns:
            Sparse feature matrix of shape ``(n_samples, n_features)``.
        """
        _log.info(
            "Fitting TF-IDF on %d samples (max_features=%d) …",
            len(X),
            self.vectorizer.max_features,
        )
        X_tfidf: sp.csr_matrix = self.vectorizer.fit_transform(X)

        vocab_size = len(self.vectorizer.vocabulary_)
        _log.info(
            "Vocabulary built: %d unique tokens → matrix %s",
            vocab_size,
            X_tfidf.shape,
        )

        if self.scaler is not None:
            X_tfidf = self.scaler.fit_transform(X_tfidf)

        self.fitted = True
        return X_tfidf

    def transform(
        self,
        X: Sequence[str],
    ) -> sp.csr_matrix:
        """Transform *X* using the already-fitted vectorizer.

        Args:
            X: Iterable of raw C/C++ function strings (validation or holdout).

        Returns:
            Sparse feature matrix of shape ``(n_samples, n_features)``.

        Raises:
            RuntimeError: If called before :meth:`fit_transform`.
        """
        if not self.fitted:
            raise RuntimeError(
                "TFIDFPipeline must be fitted before calling transform(). "
                "Call fit_transform(X_train) first."
            )
        X_tfidf: sp.csr_matrix = self.vectorizer.transform(X)

        if self.scaler is not None:
            X_tfidf = self.scaler.transform(X_tfidf)

        _log.debug("Transformed %d samples → shape %s", len(X), X_tfidf.shape)
        return X_tfidf

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def vocabulary_size(self) -> int:
        """Number of tokens in the fitted vocabulary."""
        if not self.fitted:
            raise RuntimeError("Pipeline not yet fitted.")
        return len(self.vectorizer.vocabulary_)

    def top_tokens(self, n: int = 30) -> List[str]:
        """Return the *n* tokens with the highest average IDF weight.

        Useful for a quick sanity-check: the top tokens should be
        semantically meaningful C/C++ identifiers or n-grams.

        Args:
            n: Number of top tokens to return.

        Returns:
            List of token strings sorted by descending IDF.
        """
        if not self.fitted:
            raise RuntimeError("Pipeline not yet fitted.")
        idf = self.vectorizer.idf_
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        top_idx = np.argsort(idf)[::-1][:n]
        return feature_names[top_idx].tolist()


# ---------------------------------------------------------------------------
# Convenience factory (module-level)
# ---------------------------------------------------------------------------

def build_tfidf_pipeline(
    config_path: Union[str, Path] = "configs/track_a.yaml",
    *,
    scale: bool = False,
) -> TFIDFPipeline:
    """Load config from *config_path* and return a :class:`TFIDFPipeline`.

    Shorthand for notebooks / scripts that don't want to manage the config
    object themselves::

        pipeline = build_tfidf_pipeline()
        X_train_feat = pipeline.fit_transform(X_train)
        X_val_feat   = pipeline.transform(X_val)

    Args:
        config_path: Path to ``track_a.yaml`` (default ``"configs/track_a.yaml"``).
        scale:       Whether to apply :class:`MaxAbsScaler`.

    Returns:
        Un-fitted :class:`TFIDFPipeline`.
    """
    cfg = load_config(config_path)
    return TFIDFPipeline.from_config(cfg, scale=scale)