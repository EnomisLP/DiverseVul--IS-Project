"""
Conservative normalization utilities for Case Study 1.

Purpose
-------
Make C/C++ source code text consistent for TF-IDF without removing potentially
security-relevant lexical signals. In particular, the EXP-0 default preserves:
- identifiers and API names (e.g., strcpy, memcpy)
- operators and punctuation (e.g., ->, [], *, &, casts)
- string and character literals
- comments
- line structure

It only removes obvious formatting noise:
- byte-order marks and NUL bytes
- inconsistent line endings
- trailing whitespace
- excessive horizontal whitespace outside literals/comments
- excessive blank lines

This module does NOT parse C/C++ or claim semantic equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union
import re
import unicodedata

import pandas as pd


NORMALIZATION_VERSION = "cs1-conservative-v1"


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration for conservative source-code normalization."""

    collapse_horizontal_whitespace: bool = True
    max_consecutive_blank_lines: int = 1
    preserve_comments: bool = True
    preserve_line_breaks: bool = True


DEFAULT_CONFIG = NormalizationConfig()


def _coerce_code(value: object) -> str:
    """Convert a possible dataset cell value to a source-code string."""
    if value is None:
        return ""

    # pandas.NA / numpy.nan safely become empty strings.
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def _normalize_line_endings_and_unicode(code: str) -> str:
    """Normalize Unicode and line-ending representation without changing tokens."""
    code = unicodedata.normalize("NFKC", code)
    code = code.lstrip("\ufeff")  # UTF-8 byte-order mark, if present.
    code = code.replace("\x00", "")
    return code.replace("\r\n", "\n").replace("\r", "\n")


def _collapse_whitespace_outside_literals_and_comments(
    code: str,
    preserve_comments: bool,
) -> str:
    """
    Replace repeated spaces/tabs with one space only in normal code regions.

    The scanner preserves exact contents of strings and characters. With the
    EXP-0 default, it also preserves comments. It is intentionally not a full
    C/C++ lexer; it provides conservative text cleanup for feature extraction.
    """
    output: list[str] = []
    i = 0
    n = len(code)

    state = "normal"
    escaped = False
    pending_space = False

    def flush_pending_space() -> None:
        nonlocal pending_space
        if pending_space and output:
            previous = output[-1]
            if previous not in {" ", "\n"}:
                output.append(" ")
        pending_space = False

    while i < n:
        char = code[i]
        next_char = code[i + 1] if i + 1 < n else ""

        if state == "normal":
            if char == "/" and next_char == "/":
                flush_pending_space()
                if preserve_comments:
                    output.extend(["/", "/"])
                else:
                    # Keep token boundaries when a comment is removed.
                    pending_space = True
                i += 2
                state = "line_comment"
                continue

            if char == "/" and next_char == "*":
                flush_pending_space()
                if preserve_comments:
                    output.extend(["/", "*"])
                else:
                    pending_space = True
                i += 2
                state = "block_comment"
                continue

            if char == '"':
                flush_pending_space()
                output.append(char)
                i += 1
                state = "string"
                escaped = False
                continue

            if char == "'":
                flush_pending_space()
                output.append(char)
                i += 1
                state = "char"
                escaped = False
                continue

            if char in {" ", "\t", "\v", "\f"}:
                pending_space = True
                i += 1
                continue

            if char == "\n":
                pending_space = False
                # Prevent whitespace before a newline.
                if output and output[-1] == " ":
                    output.pop()
                if not output or output[-1] != "\n":
                    output.append("\n")
                i += 1
                continue

            flush_pending_space()
            output.append(char)
            i += 1
            continue

        if state == "line_comment":
            if preserve_comments:
                output.append(char)
            if char == "\n":
                if not preserve_comments:
                    pending_space = False
                    if not output or output[-1] != "\n":
                        output.append("\n")
                state = "normal"
            i += 1
            continue

        if state == "block_comment":
            if char == "*" and next_char == "/":
                if preserve_comments:
                    output.extend(["*", "/"])
                i += 2
                state = "normal"
                continue

            if preserve_comments:
                output.append(char)
            elif char == "\n":
                pending_space = False
                if not output or output[-1] != "\n":
                    output.append("\n")
            i += 1
            continue

        if state == "string":
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                state = "normal"
            i += 1
            continue

        if state == "char":
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'":
                state = "normal"
            i += 1
            continue

    if state == "normal":
        flush_pending_space()

    return "".join(output)


def _limit_blank_lines(code: str, max_consecutive_blank_lines: int) -> str:
    """Limit only blank lines; preserve non-empty source-code lines verbatim."""
    if max_consecutive_blank_lines < 0:
        raise ValueError("max_consecutive_blank_lines must be >= 0.")

    lines = [line.rstrip() for line in code.split("\n")]
    output: list[str] = []
    blank_count = 0

    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= max_consecutive_blank_lines:
                output.append("")
        else:
            blank_count = 0
            output.append(line)

    return "\n".join(output).strip()


def normalize_code(
    code: object,
    config: NormalizationConfig = DEFAULT_CONFIG,
) -> str:
    """
    Normalize one C/C++ source-code sample conservatively.

    The default configuration preserves comments, literals, tokens, punctuation,
    and line structure. It does not perform identifier anonymization, comment
    stripping, or literal stripping.
    """
    normalized = _coerce_code(code)
    if not normalized:
        return ""

    normalized = _normalize_line_endings_and_unicode(normalized)

    if config.collapse_horizontal_whitespace:
        normalized = _collapse_whitespace_outside_literals_and_comments(
            normalized,
            preserve_comments=config.preserve_comments,
        )
    else:
        normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))

    if not config.preserve_line_breaks:
        normalized = re.sub(r"\s*\n\s*", " ", normalized)

    normalized = _limit_blank_lines(
        normalized,
        max_consecutive_blank_lines=config.max_consecutive_blank_lines,
    )

    return normalized


def normalize_code_series(
    codes: Union[pd.Series, Iterable[object]],
    config: NormalizationConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Normalize a collection of code samples while preserving its index."""
    if not isinstance(codes, pd.Series):
        codes = pd.Series(list(codes))

    return codes.map(lambda value: normalize_code(value, config=config))


def add_normalized_code_column(
    frame: pd.DataFrame,
    source_column: str = "code",
    target_column: str = "normalized_code",
    config: NormalizationConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Return a copy of a DataFrame with a normalized-code column added.

    The raw source column remains intact for traceability and later auditing.
    """
    if source_column not in frame.columns:
        raise KeyError(
            f"Expected source column '{source_column}' was not found. "
            f"Available columns: {sorted(frame.columns.tolist())}"
        )

    result = frame.copy()
    result[target_column] = normalize_code_series(
        result[source_column],
        config=config,
    )
    return result


def normalization_summary(
    raw_codes: pd.Series,
    normalized_codes: pd.Series,
) -> dict:
    """
    Build compact diagnostics for the normalization stage.

    It helps verify that normalization does not unexpectedly delete large
    amounts of source text.
    """
    if len(raw_codes) != len(normalized_codes):
        raise ValueError("raw_codes and normalized_codes must have equal length.")

    raw_text = raw_codes.fillna("").astype(str)
    normalized_text = normalized_codes.fillna("").astype(str)

    raw_lengths = raw_text.str.len()
    normalized_lengths = normalized_text.str.len()
    changed = raw_text != normalized_text

    return {
        "normalization_version": NORMALIZATION_VERSION,
        "rows": int(len(raw_text)),
        "rows_changed": int(changed.sum()),
        "rows_changed_rate": float(changed.mean()) if len(raw_text) else 0.0,
        "raw_empty_rows": int((raw_text.str.strip() == "").sum()),
        "normalized_empty_rows": int((normalized_text.str.strip() == "").sum()),
        "raw_characters_total": int(raw_lengths.sum()),
        "normalized_characters_total": int(normalized_lengths.sum()),
        "median_raw_characters": float(raw_lengths.median()) if len(raw_text) else 0.0,
        "median_normalized_characters": (
            float(normalized_lengths.median()) if len(normalized_text) else 0.0
        ),
    }


__all__ = [
    "DEFAULT_CONFIG",
    "NORMALIZATION_VERSION",
    "NormalizationConfig",
    "add_normalized_code_column",
    "normalization_summary",
    "normalize_code",
    "normalize_code_series",
]
