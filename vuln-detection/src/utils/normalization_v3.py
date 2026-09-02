from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, NamedTuple, Optional, Union
import hashlib
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

NORMALIZATION_VERSION = "cs1-conservative-v3-no-nfkc"

MAX_INPUT_SIZE = 10_000_000
MAX_OUTPUT_SIZE = 10_000_000


@dataclass(frozen=True)
class NormalizationConfig:
    collapse_horizontal_whitespace: bool = True
    max_consecutive_blank_lines: int = 1
    preserve_comments: bool = True
    preserve_line_breaks: bool = True
    max_input_size: int = MAX_INPUT_SIZE
    max_output_size: int = MAX_OUTPUT_SIZE

    def __post_init__(self) -> None:
        """Validate field ranges."""
        if self.max_consecutive_blank_lines < 0:
            raise ValueError("max_consecutive_blank_lines must be >= 0")
        if self.max_input_size <= 0 or self.max_output_size <= 0:
            raise ValueError("max_input_size and max_output_size must be positive")


DEFAULT_CONFIG = NormalizationConfig()


class Token(NamedTuple):
    kind: Literal[
        "identifier", "number", "string", "char", "raw_string", "comment",
        "whitespace", "newline", "operator", "punct", "other"
    ]
    text: str
    start: int
    end: int


def _coerce_code(value: object, max_size: int = MAX_INPUT_SIZE) -> str:
    """Coerce a raw value to a size-bounded string, treating null/NaN as empty."""
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


def _normalize_line_endings_and_unicode(code: str) -> str:
    """Strip a BOM/NUL bytes and normalize line endings to \\n."""
    code = code.lstrip("﻿")
    code = code.replace("\x00", "")
    return code.replace("\r\n", "\n").replace("\r", "\n")


def _is_ident_start(ch: str) -> bool:
    """Check whether a character can start a C/C++ identifier."""
    return ch == "_" or ch.isalpha()


def _is_ident_continue(ch: str) -> bool:
    """Check whether a character can continue a C/C++ identifier."""
    return ch == "_" or ch.isalpha() or ch.isdigit()


def _scan_identifier(code: str, i: int) -> tuple[str, int]:
    """Scan one identifier starting at index i."""
    j = i + 1
    while j < len(code) and _is_ident_continue(code[j]):
        j += 1
    return code[i:j], j


def _scan_line_comment(code: str, i: int) -> tuple[str, int]:
    """Scan a // line comment starting at index i."""
    j = i
    while j < len(code) and code[j] != "\n":
        j += 1
    if j < len(code):
        j += 1
    return code[i:j], j


def _scan_block_comment(code: str, i: int) -> tuple[str, int]:
    """Scan a /* ... */ block comment starting at index i."""
    j = i + 2
    while j + 1 < len(code):
        if code[j] == "*" and code[j + 1] == "/":
            return code[i:j + 2], j + 2
        j += 1
    logger.warning("Unterminated block comment at index %d", i)
    return code[i:], len(code)


def _scan_ordinary_quoted_literal(code: str, quote_index: int) -> tuple[str, int]:
    """Scan a quoted string/char literal, honoring backslash escapes."""
    quote = code[quote_index]
    assert quote in {"'", '"'}
    i = quote_index + 1
    escaped = False
    while i < len(code):
        ch = code[i]
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == quote:
            return code[quote_index:i + 1], i + 1
        i += 1
    logger.warning("Unterminated quoted literal at index %d", quote_index)
    return code[quote_index:], len(code)


def _scan_prefixed_ordinary_literal(code: str, i: int) -> Optional[tuple[str, int, str]]:
    """Scan a prefixed string/char literal (u8"...", L'...', etc.), if present at index i."""
    prefixes = ("u8", "u", "U", "L")
    for prefix in prefixes:
        if code.startswith(prefix, i):
            qpos = i + len(prefix)
            if qpos < len(code) and code[qpos] in {'"', "'"}:
                literal, end = _scan_ordinary_quoted_literal(code, qpos)
                kind = "string" if code[qpos] == '"' else "char"
                return prefix + literal, end, kind
    return None


def _scan_raw_string_literal(code: str, i: int) -> Optional[tuple[str, int]]:
    """Scan a C++11 raw string literal (R"delim(...)delim"), if present at index i."""
    prefixes = ("u8R", "uR", "UR", "LR", "R")
    for prefix in prefixes:
        if not code.startswith(prefix + '"', i):
            continue
        delim_start = i + len(prefix) + 1
        open_paren = code.find("(", delim_start, min(len(code), delim_start + 18))
        if open_paren == -1:
            continue
        delimiter = code[delim_start:open_paren]
        if any(ch.isspace() or ch in "()\\" for ch in delimiter):
            continue
        close = ")" + delimiter + '"'
        close_pos = code.find(close, open_paren + 1)
        if close_pos == -1:
            logger.warning("Unterminated raw string literal at index %d", i)
            return code[i:], len(code)
        end = close_pos + len(close)
        return code[i:end], end
    return None


def _scan_number_literal(code: str, i: int) -> tuple[str, int]:
    """Scan a numeric literal, including hex/exponent/digit-separator characters."""
    n = len(code)
    j = i

    while j < n:
        ch = code[j]
        if ch.isalnum() or ch in {"_", "."}:
            j += 1
            continue
        if ch == "'" and j + 1 < n and code[j + 1].isalnum():
            j += 1
            continue
        if ch in {"+", "-"} and j > i and code[j - 1] in {"e", "E", "p", "P"}:
            j += 1
            continue
        break
    return code[i:j], j


MULTI_CHAR_OPERATORS = (
    "...", "->*", "<<=", ">>=", "++", "--", "->", "<=", ">=", "==", "!=",
    "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>",
    "::", ".*", "##",
)


def iter_cpp_lexical_tokens(code: str) -> Iterator[Token]:
    """Tokenize C/C++ source into a stream of lexical Tokens."""
    i = 0
    n = len(code)
    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""

        if ch == "\n":
            yield Token("newline", ch, i, i + 1)
            i += 1
            continue

        if ch in " \t\v\f":
            j = i + 1
            while j < n and code[j] in " \t\v\f":
                j += 1
            yield Token("whitespace", code[i:j], i, j)
            i = j
            continue

        if ch == "/" and nxt == "/":
            text, end = _scan_line_comment(code, i)
            yield Token("comment", text, i, end)
            i = end
            continue

        if ch == "/" and nxt == "*":
            text, end = _scan_block_comment(code, i)
            yield Token("comment", text, i, end)
            i = end
            continue

        raw = _scan_raw_string_literal(code, i)
        if raw is not None:
            text, end = raw
            yield Token("raw_string", text, i, end)
            i = end
            continue

        prefixed = _scan_prefixed_ordinary_literal(code, i)
        if prefixed is not None:
            text, end, kind = prefixed
            yield Token(kind, text, i, end)  # type: ignore[arg-type]
            i = end
            continue

        if ch == '"':
            text, end = _scan_ordinary_quoted_literal(code, i)
            yield Token("string", text, i, end)
            i = end
            continue

        if ch == "'":
            text, end = _scan_ordinary_quoted_literal(code, i)
            yield Token("char", text, i, end)
            i = end
            continue

        if ch.isdigit() or (ch == "." and nxt.isdigit()):
            text, end = _scan_number_literal(code, i)
            yield Token("number", text, i, end)
            i = end
            continue

        if _is_ident_start(ch):
            text, end = _scan_identifier(code, i)
            yield Token("identifier", text, i, end)
            i = end
            continue

        matched = False
        for op in MULTI_CHAR_OPERATORS:
            if code.startswith(op, i):
                yield Token("operator", op, i, i + len(op))
                i += len(op)
                matched = True
                break
        if matched:
            continue

        if ch in "{}[]();,?:#":
            yield Token("punct", ch, i, i + 1)
        elif ch in "+-*/%<>=!&|^~.":
            yield Token("operator", ch, i, i + 1)
        else:
            yield Token("other", ch, i, i + 1)
        i += 1


def _collapse_whitespace_outside_literals_and_comments(code: str, preserve_comments: bool) -> str:
    """Collapse runs of horizontal whitespace to a single space, leaving literals/comments untouched."""
    parts: list[str] = []
    pending_space = False

    def flush_space() -> None:
        """Emit at most one pending space before the next token."""
        nonlocal pending_space
        if pending_space and parts and parts[-1] not in {" ", "\n"}:
            parts.append(" ")
        pending_space = False

    for tok in iter_cpp_lexical_tokens(code):
        if tok.kind == "whitespace":
            pending_space = True
            continue
        if tok.kind == "newline":
            pending_space = False
            if parts and parts[-1] == " ":
                parts.pop()
            if not parts or parts[-1] != "\n":
                parts.append("\n")
            continue
        if tok.kind == "comment" and not preserve_comments:
            if "\n" in tok.text:
                pending_space = False
                if parts and parts[-1] == " ":
                    parts.pop()
                if not parts or parts[-1] != "\n":
                    parts.append("\n")
            else:
                pending_space = True
            continue
        flush_space()
        parts.append(tok.text)

    flush_space()
    return "".join(parts)


def _limit_blank_lines(code: str, max_consecutive_blank_lines: int) -> str:
    """Trim trailing whitespace per line and cap consecutive blank lines."""
    if max_consecutive_blank_lines < 0:
        raise ValueError("max_consecutive_blank_lines must be >= 0")
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


def normalize_code(code: object, config: NormalizationConfig = DEFAULT_CONFIG) -> str:
    """Normalize one function's source text (whitespace/blank lines), preserving semantics."""
    normalized = _coerce_code(code, max_size=config.max_input_size)
    if not normalized:
        return ""
    normalized = _normalize_line_endings_and_unicode(normalized)
    if config.collapse_horizontal_whitespace:
        normalized = _collapse_whitespace_outside_literals_and_comments(
            normalized, preserve_comments=config.preserve_comments
        )
    else:
        normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    if not config.preserve_line_breaks:
        normalized = re.sub(r"\s*\n\s*", " ", normalized)
    normalized = _limit_blank_lines(normalized, config.max_consecutive_blank_lines)
    if len(normalized) > config.max_output_size:
        raise ValueError(f"Normalized code exceeds maximum output size {config.max_output_size}")
    return normalized


def normalize_code_series(
    codes: Union[pd.Series, Iterable[object]],
    config: NormalizationConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Apply normalize_code to every element of a series."""
    if not isinstance(codes, pd.Series):
        codes = pd.Series(list(codes))
    return codes.map(lambda value: normalize_code(value, config=config))


def add_normalized_code_column(
    frame: pd.DataFrame,
    source_column: str = "code",
    target_column: str = "normalized_code",
    config: NormalizationConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Add a normalized-code column derived from an existing source column."""
    if source_column not in frame.columns:
        raise KeyError(f"Missing source column: {source_column}")
    output = frame.copy()
    output[target_column] = normalize_code_series(output[source_column], config=config)
    return output


def representation_summary(series: pd.Series) -> dict[str, object]:
    """Summarize a text column's length distribution and content hash."""
    lengths = series.fillna("").map(len)
    sha = hashlib.sha256("\n".join(series.fillna("").head(10_000).tolist()).encode("utf-8", errors="replace")).hexdigest()
    return {
        "n_rows": int(series.shape[0]),
        "empty_rows": int((series.fillna("") == "").sum()),
        "mean_chars": float(lengths.mean()) if len(lengths) else 0.0,
        "median_chars": float(lengths.median()) if len(lengths) else 0.0,
        "max_chars": int(lengths.max()) if len(lengths) else 0,
        "first_10000_sha256": sha,
    }


normalization_summary = representation_summary
