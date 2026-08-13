from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
import time
import warnings
from typing import Iterable, List, NamedTuple, Optional, Sequence, Type, Union

import pandas as pd

logger = logging.getLogger(__name__)

SCOPE2_ABSTRACTION_VERSION = "cs1-scope2-abstraction-v2"

DEFAULT_SOURCE_COLUMN = "normalized_code"
DEFAULT_TARGET_COLUMN = "abstracted_code_v1"
DEFAULT_FAILURE_FLAG_COLUMN = "abstracted_code_v1_scope2_failed"

MAX_INPUT_SIZE = 10_000_000

_INSTALL_HINT = (
    "SCoPE2 is not installed (it is not on PyPI, must be installed from source). Run:\n"
    "  pip install 'tree-sitter==0.23.0' tree-sitter-cpp 'ruamel.yaml>=0.2.7'\n"
    "  pip install git+https://github.com/jp2425/SCoPE2.git\n"
    "tree-sitter-cpp is a separate grammar package SCoPE2 does not declare as a "
    "dependency in its own pyproject.toml; it must be installed explicitly."
)

_SCOPE2_NOT_FOUND_MARKER = "TreeSitter didn't returned any "


def _import_scope2():
    try:
        import tree_sitter_cpp
        from tree_sitter import Language
        from SCoPE2.SCoPE import SCoPE
        from SCoPE2.representations.CodeRepresentation import CodeRepresentation
        from SCoPE2.transformation.implementations import (
            RemoveCommentsTr,
            ReplaceFunctionNamesTr,
            ReplaceVariableNamesTr,
        )
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    transformations = (RemoveCommentsTr, ReplaceVariableNamesTr, ReplaceFunctionNamesTr)
    return SCoPE, Language, tree_sitter_cpp, CodeRepresentation, transformations


def build_scope2_processor(query_yaml_text: str):
    SCoPE, Language, tree_sitter_cpp, _, _ = _import_scope2()
    return SCoPE(query_yaml_text, Language(tree_sitter_cpp.language()))


def default_transformations() -> List[Type]:
    _, _, _, _, transformations = _import_scope2()
    return list(transformations)


def default_representation_cls():
    _, _, _, code_representation_cls, _ = _import_scope2()
    return code_representation_cls


def _coerce_code(value: object, max_size: int = MAX_INPUT_SIZE) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, str):
        text = value
    else:
        raise TypeError(f"Expected str or bytes source code, got {type(value).__name__}")
    if len(text) > max_size:
        raise ValueError(f"Code sample exceeds maximum size {max_size}: {len(text)}")
    return text


def _extract_skip_query_id(warning_message: str) -> str:
    idx = warning_message.find(_SCOPE2_NOT_FOUND_MARKER)
    if idx == -1:
        return warning_message
    return warning_message[idx + len(_SCOPE2_NOT_FOUND_MARKER):].strip()


class Scope2RowResult(NamedTuple):
    text: str
    failed: bool
    failure_reason: Optional[str]
    input_was_empty: bool
    skipped_transformations: tuple[str, ...]


def apply_scope2_to_code(
    code: object,
    scope,
    transformations: Sequence[Type],
    representation_cls: Type,
) -> Scope2RowResult:
    try:
        text = _coerce_code(code)
    except (TypeError, ValueError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning("SCoPE2 input coercion failed; row flagged as failed. Reason: %s", reason)
        return Scope2RowResult(text="", failed=True, failure_reason=reason, input_was_empty=False, skipped_transformations=())

    if not text.strip():
        return Scope2RowResult(text="", failed=False, failure_reason=None, input_was_empty=True, skipped_transformations=())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            transformed = scope.process(text, list(transformations), representation_cls)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("SCoPE2 failed on a row; falling back to normalized_code. Reason: %s", reason)
            return Scope2RowResult(text=text, failed=True, failure_reason=reason, input_was_empty=False, skipped_transformations=())

    skipped = tuple(_extract_skip_query_id(str(w.message)) for w in caught)
    return Scope2RowResult(
        text=str(transformed), failed=False, failure_reason=None, input_was_empty=False, skipped_transformations=skipped
    )


@dataclass(frozen=True)
class Scope2BatchReport:
    n_rows: int
    n_empty_input: int
    n_failed: int
    failure_reason_counts: Counter
    skip_reason_counts: Counter
    elapsed_seconds: float
    failed_row_positions: tuple[int, ...]

    def summary_lines(self) -> List[str]:
        lines = [
            f"SCoPE2 batch: {self.n_rows} rows, {self.elapsed_seconds:.1f}s "
            f"({self.n_rows / self.elapsed_seconds if self.elapsed_seconds > 0 else float('inf'):.1f} rows/sec)",
            f"  empty input rows (passed through as empty string): {self.n_empty_input}",
            f"  rows where SCoPE2 itself failed (fell back to normalized_code): {self.n_failed}",
        ]
        if self.failure_reason_counts:
            lines.append(f"  failure reasons: {dict(self.failure_reason_counts)}")
        if self.skip_reason_counts:
            lines.append(
                "  benign per-transformation skips (e.g. 'comment' = rows with no comments to remove, "
                f"NOT row failures): {dict(self.skip_reason_counts)}"
            )
        return lines


def apply_scope2_to_series(
    codes: Union[pd.Series, Iterable[object]],
    scope,
    transformations: Optional[Sequence[Type]] = None,
    representation_cls: Optional[Type] = None,
    *,
    log_every: int = 25_000,
) -> tuple[pd.Series, pd.Series, Scope2BatchReport]:
    if transformations is None:
        transformations = default_transformations()
    if representation_cls is None:
        representation_cls = default_representation_cls()

    if not isinstance(codes, pd.Series):
        codes = pd.Series(list(codes))

    n_rows = len(codes)
    texts: List[str] = []
    failed_flags: List[bool] = []
    failure_reason_counts: Counter = Counter()
    skip_reason_counts: Counter = Counter()
    failed_positions: List[int] = []
    n_empty_input = 0

    start = time.monotonic()
    for pos, value in enumerate(codes):
        result = apply_scope2_to_code(value, scope, transformations, representation_cls)
        texts.append(result.text)
        failed_flags.append(result.failed)
        if result.input_was_empty:
            n_empty_input += 1
        if result.failed:
            failed_positions.append(pos)
            reason_key = result.failure_reason.split(":", 1)[0] if result.failure_reason else "Unknown"
            failure_reason_counts[reason_key] += 1
        for query_id in result.skipped_transformations:
            skip_reason_counts[query_id] += 1

        if log_every and (pos + 1) % log_every == 0:
            elapsed = time.monotonic() - start
            rate = (pos + 1) / elapsed if elapsed > 0 else float("inf")
            logger.info(
                "SCoPE2 progress: %d/%d rows (%.1f rows/sec, %d failed so far)",
                pos + 1, n_rows, rate, len(failed_positions),
            )

    elapsed_seconds = time.monotonic() - start

    report = Scope2BatchReport(
        n_rows=n_rows,
        n_empty_input=n_empty_input,
        n_failed=len(failed_positions),
        failure_reason_counts=failure_reason_counts,
        skip_reason_counts=skip_reason_counts,
        elapsed_seconds=elapsed_seconds,
        failed_row_positions=tuple(failed_positions[:100]),
    )

    abstracted_series = pd.Series(texts, index=codes.index, name=DEFAULT_TARGET_COLUMN)
    failed_series = pd.Series(failed_flags, index=codes.index, name=DEFAULT_FAILURE_FLAG_COLUMN)
    return abstracted_series, failed_series, report


def add_scope2_abstracted_code_column(
    frame: pd.DataFrame,
    scope,
    source_column: str = DEFAULT_SOURCE_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    failure_flag_column: str = DEFAULT_FAILURE_FLAG_COLUMN,
    transformations: Optional[Sequence[Type]] = None,
    representation_cls: Optional[Type] = None,
    *,
    log_every: int = 25_000,
) -> tuple[pd.DataFrame, Scope2BatchReport]:
    if source_column not in frame.columns:
        raise KeyError(f"Missing source column: {source_column}")

    abstracted, failed, report = apply_scope2_to_series(
        frame[source_column],
        scope,
        transformations=transformations,
        representation_cls=representation_cls,
        log_every=log_every,
    )

    output = frame.copy()
    output[target_column] = abstracted.to_numpy()
    output[failure_flag_column] = failed.to_numpy()
    return output, report
