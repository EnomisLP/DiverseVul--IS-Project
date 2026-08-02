"""Lightweight deterministic C/C++ structural feature extraction for CS1-EXP1.

The features in this module are source-level proxies, not compiler-verified
semantic static analysis. Comment and literal contents are excluded from code
token counts to reduce the risk of textual label leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Sequence

import numpy as np
import pandas as pd

STATIC_FEATURE_VERSION = "cs1-static-features-v1"

IDENTIFIER_RE = re.compile(r"\b[A-Za-z_]\w*\b")
NUMBER_RE = re.compile(r"\b(?:0[xX][0-9A-Fa-f]+|\d+(?:\.\d+)?)\b")
FUNCTION_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
ARRAY_ACCESS_RE = re.compile(r"\[[^\]\n]*\]")
PREPROCESSOR_RE = re.compile(r"(?m)^\s*#\s*\w+")
# Conservative cast proxy: require a built-in/standard type token, a struct/enum
# introducer, or a type-like identifier beginning with an uppercase letter. This
# intentionally avoids treating ordinary parenthesized variables such as ``(p)``
# as casts.
CAST_LIKE_RE = re.compile(
    r"\(\s*(?:(?:const|volatile|unsigned|signed|long|short)\s+|(?:struct|enum)\s+)*"
    r"(?:char|short|int|long|float|double|void|bool|size_t|ssize_t|"
    r"u?int(?:8|16|32|64)_t|[A-Z][A-Za-z_]\w*)(?:\s*\*+)?\s*\)"
)
POINTER_DECL_RE = re.compile(
    r"\b(?:char|short|int|long|float|double|void|bool|size_t|"
    r"uint(?:8|16|32|64)_t|int(?:8|16|32|64)_t)\s*\*+"
)

CONTROL_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "default",
    "return", "sizeof", "catch", "new", "delete",
}
INTEGER_TYPE_TOKENS = {
    "char", "short", "int", "long", "size_t", "ssize_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "unsigned", "signed",
}
MEMORY_APIS = {"malloc", "calloc", "realloc", "free", "memcpy", "memmove", "memset"}
STRING_APIS = {"strcpy", "strncpy", "strcat", "strncat", "strlen", "strcmp", "strncmp", "strdup", "strchr", "strstr"}
FORMAT_APIS = {"sprintf", "snprintf", "vsprintf", "vsnprintf", "printf", "fprintf", "vfprintf"}
INPUT_APIS = {"gets", "fgets", "scanf", "sscanf", "fscanf", "read", "recv"}
ALLOCATION_APIS = {"malloc", "calloc", "realloc", "strdup"}
DEALLOCATION_APIS = {"free", "delete"}
ASSERT_APIS = {"assert", "BUG_ON", "WARN_ON"}

FEATURE_COLUMNS = [
    "raw_char_count", "raw_line_count", "nonempty_line_count",
    "avg_nonempty_line_length", "max_line_length",
    "comment_char_count", "comment_line_count", "comment_char_ratio",
    "string_literal_count", "char_literal_count", "preprocessor_directive_count",
    "identifier_count", "unique_identifier_count", "identifier_diversity",
    "numeric_literal_count", "function_call_count", "unique_function_call_count",
    "if_count", "else_count", "for_count", "while_count", "do_count",
    "switch_count", "case_count", "goto_count", "return_count",
    "break_continue_count", "boolean_operator_count", "ternary_operator_count",
    "cyclomatic_complexity_proxy", "brace_open_count", "brace_close_count",
    "parenthesis_open_count", "array_access_count", "star_operator_count",
    "arrow_operator_count", "address_of_operator_count", "cast_like_count",
    "pointer_declaration_count", "integer_type_token_count",
    "assignment_operator_count", "comparison_operator_count", "bitwise_operator_count",
    "increment_decrement_count", "memory_api_call_count", "string_api_call_count",
    "format_api_call_count", "input_api_call_count", "allocation_api_call_count",
    "deallocation_api_call_count", "unsafe_api_presence_count", "sizeof_count",
    "null_token_count", "assert_call_count",
]

@dataclass(frozen=True)
class StaticFeatureConfig:
    source_id_column: str = "source_row_id"
    code_column: str = "code"
    progress_every: int = 25_000


def _count_word(tokens: Sequence[str], token: str) -> int:
    return int(sum(item == token for item in tokens))


def _count_api_calls(function_names: Sequence[str], api_names: set[str]) -> int:
    return int(sum(name in api_names for name in function_names))


def _sanitize_source(code: str) -> tuple[str, int, int, int, int]:
    """Mask comment and literal contents, preserving line breaks and code syntax."""
    out: list[str] = []
    i, n = 0, len(code)
    state = "normal"
    escaped = False
    comment_chars = comment_lines = string_literals = char_literals = 0
    block_comment_on_line = False

    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if state == "normal":
            if ch == "/" and nxt == "/":
                out.extend([" ", " "])
                i += 2
                state = "line_comment"
                comment_lines += 1
                continue
            if ch == "/" and nxt == "*":
                out.extend([" ", " "])
                i += 2
                state = "block_comment"
                comment_lines += 1
                block_comment_on_line = True
                continue
            if ch == '"':
                out.append(" ")
                state = "string"
                escaped = False
                string_literals += 1
                i += 1
                continue
            if ch == "'":
                out.append(" ")
                state = "char"
                escaped = False
                char_literals += 1
                i += 1
                continue
            out.append(ch)
            i += 1
            continue

        if state == "line_comment":
            comment_chars += 1
            if ch == "\n":
                out.append("\n")
                state = "normal"
            else:
                out.append(" ")
            i += 1
            continue

        if state == "block_comment":
            if ch == "*" and nxt == "/":
                out.extend([" ", " "])
                i += 2
                state = "normal"
                block_comment_on_line = False
                continue
            comment_chars += 1
            if ch == "\n":
                out.append("\n")
                if not block_comment_on_line:
                    comment_lines += 1
                block_comment_on_line = False
            else:
                out.append(" ")
                block_comment_on_line = True
            i += 1
            continue

        # string / char literal: mask contents but retain physical lines.
        out.append("\n" if ch == "\n" else " ")
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif state == "string" and ch == '"':
            state = "normal"
        elif state == "char" and ch == "'":
            state = "normal"
        i += 1

    return "".join(out), comment_chars, comment_lines, string_literals, char_literals


def extract_static_features(code: object) -> dict[str, float]:
    """Extract deterministic source-level proxy features for one function."""
    raw = "" if code is None else str(code)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    sanitized, comment_chars, comment_lines, strings, chars = _sanitize_source(raw)

    lines = raw.split("\n")
    nonempty = [line for line in lines if line.strip()]
    nonempty_lengths = [len(line) for line in nonempty]
    tokens = IDENTIFIER_RE.findall(sanitized)
    token_set = set(tokens)
    calls = [name for name in FUNCTION_CALL_RE.findall(sanitized) if name not in CONTROL_KEYWORDS]

    if_count = _count_word(tokens, "if")
    for_count = _count_word(tokens, "for")
    while_count = _count_word(tokens, "while")
    case_count = _count_word(tokens, "case")
    boolean_ops = sanitized.count("&&") + sanitized.count("||")
    ternary_ops = sanitized.count("?")

    memory_calls = _count_api_calls(calls, MEMORY_APIS)
    string_calls = _count_api_calls(calls, STRING_APIS)
    format_calls = _count_api_calls(calls, FORMAT_APIS)
    input_calls = _count_api_calls(calls, INPUT_APIS)

    values = {
        "raw_char_count": len(raw),
        "raw_line_count": len(lines),
        "nonempty_line_count": len(nonempty),
        "avg_nonempty_line_length": float(np.mean(nonempty_lengths)) if nonempty_lengths else 0.0,
        "max_line_length": max((len(line) for line in lines), default=0),
        "comment_char_count": comment_chars,
        "comment_line_count": comment_lines,
        "comment_char_ratio": float(comment_chars / len(raw)) if raw else 0.0,
        "string_literal_count": strings,
        "char_literal_count": chars,
        "preprocessor_directive_count": len(PREPROCESSOR_RE.findall(raw)),
        "identifier_count": len(tokens),
        "unique_identifier_count": len(token_set),
        "identifier_diversity": float(len(token_set) / len(tokens)) if tokens else 0.0,
        "numeric_literal_count": len(NUMBER_RE.findall(sanitized)),
        "function_call_count": len(calls),
        "unique_function_call_count": len(set(calls)),
        "if_count": if_count,
        "else_count": _count_word(tokens, "else"),
        "for_count": for_count,
        "while_count": while_count,
        "do_count": _count_word(tokens, "do"),
        "switch_count": _count_word(tokens, "switch"),
        "case_count": case_count,
        "goto_count": _count_word(tokens, "goto"),
        "return_count": _count_word(tokens, "return"),
        "break_continue_count": _count_word(tokens, "break") + _count_word(tokens, "continue"),
        "boolean_operator_count": boolean_ops,
        "ternary_operator_count": ternary_ops,
        "cyclomatic_complexity_proxy": 1 + if_count + for_count + while_count + case_count + boolean_ops + ternary_ops,
        "brace_open_count": sanitized.count("{"),
        "brace_close_count": sanitized.count("}"),
        "parenthesis_open_count": sanitized.count("("),
        "array_access_count": len(ARRAY_ACCESS_RE.findall(sanitized)),
        "star_operator_count": sanitized.count("*"),
        "arrow_operator_count": sanitized.count("->"),
        "address_of_operator_count": len(re.findall(r"(?<![&])&(?![&])", sanitized)),
        "cast_like_count": len(CAST_LIKE_RE.findall(sanitized)),
        "pointer_declaration_count": len(POINTER_DECL_RE.findall(sanitized)),
        "integer_type_token_count": int(sum(t in INTEGER_TYPE_TOKENS for t in tokens)),
        "assignment_operator_count": len(re.findall(r"(?<![=!<>])=(?!=)", sanitized)),
        "comparison_operator_count": len(re.findall(r"==|!=|<=|>=|(?<![<>=!])<(?![<=])|(?<![<>=!])>(?![>=])", sanitized)),
        "bitwise_operator_count": len(re.findall(r"(?<![|])\|(?![|])|(?<![&])&(?![&])|\^|<<|>>|~", sanitized)),
        "increment_decrement_count": sanitized.count("++") + sanitized.count("--"),
        "memory_api_call_count": memory_calls,
        "string_api_call_count": string_calls,
        "format_api_call_count": format_calls,
        "input_api_call_count": input_calls,
        "allocation_api_call_count": _count_api_calls(calls, ALLOCATION_APIS),
        "deallocation_api_call_count": _count_api_calls(calls, DEALLOCATION_APIS),
        "unsafe_api_presence_count": int((string_calls > 0) + (format_calls > 0) + (input_calls > 0)),
        "sizeof_count": _count_word(tokens, "sizeof"),
        "null_token_count": _count_word(tokens, "NULL") + _count_word(tokens, "nullptr") + _count_word(tokens, "null"),
        "assert_call_count": _count_api_calls(calls, ASSERT_APIS),
    }
    if list(values) != FEATURE_COLUMNS:
        raise RuntimeError("Static feature schema mismatch.")
    return {name: float(value) for name, value in values.items()}


def extract_static_feature_frame(frame: pd.DataFrame, config: StaticFeatureConfig = StaticFeatureConfig()) -> pd.DataFrame:
    """Extract deterministic features for each input row, with progress output."""
    required = [config.source_id_column, config.code_column]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if frame[config.source_id_column].isna().any() or frame[config.source_id_column].duplicated().any():
        raise ValueError("source_row_id must be complete and unique.")

    rows: list[dict[str, float]] = []
    start = time.perf_counter()
    total = len(frame)
    for pos, (source_id, code) in enumerate(zip(frame[config.source_id_column].tolist(), frame[config.code_column].tolist()), start=1):
        row = extract_static_features(code)
        row[config.source_id_column] = source_id
        rows.append(row)
        if config.progress_every > 0 and (pos % config.progress_every == 0 or pos == total):
            print(f"[static features] {pos:,}/{total:,} ({pos / total:.1%}) in {(time.perf_counter() - start) / 60:.2f} min", flush=True)

    result = pd.DataFrame(rows)[[config.source_id_column] + FEATURE_COLUMNS]
    array = result[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if len(result) != len(frame) or not np.isfinite(array).all():
        raise RuntimeError("Static feature extraction failed integrity checks.")
    return result


def summarize_static_features(static_frame: pd.DataFrame, source_id_column: str = "source_row_id") -> pd.DataFrame:
    """Create report-ready distribution diagnostics for every static feature."""
    required = [source_id_column] + FEATURE_COLUMNS
    missing = [col for col in required if col not in static_frame.columns]
    if missing:
        raise KeyError(f"Static feature frame is missing columns: {missing}")
    summary = static_frame[FEATURE_COLUMNS].describe(percentiles=[0.25, 0.50, 0.75, 0.95]).T.reset_index()
    summary = summary.rename(columns={"index": "feature"})
    summary["nonzero_rows"] = (static_frame[FEATURE_COLUMNS] > 0).sum().to_numpy()
    summary["nonzero_rate"] = (static_frame[FEATURE_COLUMNS] > 0).mean().to_numpy()
    return summary


def save_static_feature_artifacts(static_frame: pd.DataFrame, output_dir: Path | str, config: StaticFeatureConfig = StaticFeatureConfig(), source_dataset_path: Path | str | None = None) -> dict[str, Path]:
    """Persist the static feature cache, summary, and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "cs1_static_features_v1.parquet"
    summary_path = output_dir / "cs1_static_features_v1_summary.csv"
    metadata_path = output_dir / "cs1_static_features_v1_metadata.json"
    static_frame.to_parquet(parquet_path, index=False)
    summarize_static_features(static_frame, config.source_id_column).to_csv(summary_path, index=False)
    metadata = {
        "static_feature_version": STATIC_FEATURE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(static_frame)),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "source_dataset_path": str(source_dataset_path) if source_dataset_path else None,
        "note": "Deterministic lexical/structural proxy features; not compiler-verified semantic analysis.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"parquet": parquet_path, "summary_csv": summary_path, "metadata_json": metadata_path}


__all__ = [
    "FEATURE_COLUMNS", "STATIC_FEATURE_VERSION", "StaticFeatureConfig",
    "extract_static_features", "extract_static_feature_frame",
    "save_static_feature_artifacts", "summarize_static_features",
]
