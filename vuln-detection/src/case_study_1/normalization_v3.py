"""
Case Study 1 C/C++ normalization and identifier abstraction utilities.

This module keeps the original conservative normalization API while adding a
separate, explicit abstraction path for cross-project transfer experiments.

Design principles
-----------------
1. Do not overwrite existing normalized_code artifacts. Use this module to build
   an additional abstracted_code_v1 column.
2. Preserve character identity: no NFKC normalization is applied.
3. Treat C/C++ raw strings, ordinary strings, character literals, comments, and
   digit-separated numeric literals explicitly.
4. Identifier abstraction is lexical and deterministic, not a full parser. It is
   intended for TF-IDF / classical ML A/B studies, not semantic equivalence.

Recommended experiment
----------------------
A/B test normalized_code vs abstracted_code_v1 under the same frozen manifests
and nested development-only protocol. Do not touch the frozen outer holdout while
iterating on this representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal, NamedTuple, Optional, Union
import hashlib
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

NORMALIZATION_VERSION = "cs1-conservative-v3-no-nfkc"
ABSTRACTION_VERSION = "cs1-identifier-abstraction-v1"

MAX_INPUT_SIZE = 10_000_000
MAX_OUTPUT_SIZE = 10_000_000


@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration for conservative source-code normalization."""

    collapse_horizontal_whitespace: bool = True
    max_consecutive_blank_lines: int = 1
    preserve_comments: bool = True
    preserve_line_breaks: bool = True
    max_input_size: int = MAX_INPUT_SIZE
    max_output_size: int = MAX_OUTPUT_SIZE

    def __post_init__(self) -> None:
        if self.max_consecutive_blank_lines < 0:
            raise ValueError("max_consecutive_blank_lines must be >= 0")
        if self.max_input_size <= 0 or self.max_output_size <= 0:
            raise ValueError("max_input_size and max_output_size must be positive")


@dataclass(frozen=True)
class AbstractionConfig:
    """Configuration for lexical identifier abstraction."""

    # Comments often carry issue IDs, project names, commit hints, and prose.
    preserve_comments: bool = False

    # Replacing literal content reduces project-specific string paths/messages
    # while preserving the presence of string/char tokens.
    preserve_string_literal_content: bool = False
    preserve_char_literal_content: bool = False

    # Keep numeric constants exactly by default because sizes, bounds, and flags
    # may be security-relevant.
    preserve_numeric_literals: bool = True

    # Apply conservative whitespace cleanup before abstraction.
    normalize_first: bool = True

    # Placeholder names. Use uppercase tokens because sklearn word analyzer keeps
    # them as clear transferable lexical features when lower=False.
    variable_prefix: str = "VAR"
    function_prefix: str = "FUNC"
    field_prefix: str = "FIELD"
    type_prefix: str = "TYPE"
    macro_prefix: str = "MACRO"

    # Keep identifiers that are known language keywords, standard types, or
    # security-relevant APIs.
    preserve_keywords: bool = True
    preserve_builtin_types: bool = True
    preserve_api_whitelist: bool = True

    # Keep preprocessor directive names such as include/define/ifdef.
    preserve_preprocessor_directives: bool = True

    extra_api_whitelist: frozenset[str] = field(default_factory=frozenset)


DEFAULT_CONFIG = NormalizationConfig()
DEFAULT_ABSTRACTION_CONFIG = AbstractionConfig()


C_CPP_KEYWORDS: frozenset[str] = frozenset(
    {
        # C keywords
        "auto", "break", "case", "char", "const", "continue", "default",
        "do", "double", "else", "enum", "extern", "float", "for", "goto",
        "if", "inline", "int", "long", "register", "restrict", "return",
        "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
        "union", "unsigned", "void", "volatile", "while", "_Alignas",
        "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary",
        "_Noreturn", "_Static_assert", "_Thread_local",
        # C++ keywords
        "alignas", "alignof", "and", "and_eq", "asm", "bitand", "bitor",
        "bool", "catch", "char8_t", "char16_t", "char32_t", "class",
        "compl", "concept", "consteval", "constexpr", "constinit", "const_cast",
        "co_await", "co_return", "co_yield", "decltype", "delete", "dynamic_cast",
        "explicit", "export", "false", "friend", "mutable", "namespace", "new",
        "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq",
        "private", "protected", "public", "reinterpret_cast", "requires",
        "static_assert", "static_cast", "template", "this", "thread_local", "throw",
        "true", "try", "typename", "typeid", "using", "virtual", "wchar_t",
        "xor", "xor_eq",
    }
)

PREPROCESSOR_DIRECTIVES: frozenset[str] = frozenset(
    {
        "include", "define", "undef", "ifdef", "ifndef", "if", "elif", "else",
        "endif", "error", "warning", "pragma", "line", "defined", "import",
    }
)

BUILTIN_TYPES_AND_COMMON_ALIASES: frozenset[str] = frozenset(
    {
        "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
        "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t",
        "uint32_t", "uint64_t", "int_fast8_t", "int_fast16_t", "int_fast32_t",
        "int_fast64_t", "uint_fast8_t", "uint_fast16_t", "uint_fast32_t",
        "uint_fast64_t", "int_least8_t", "int_least16_t", "int_least32_t",
        "int_least64_t", "uint_least8_t", "uint_least16_t", "uint_least32_t",
        "uint_least64_t", "FILE", "DIR", "time_t", "clock_t", "pid_t",
        "off_t", "mode_t", "uid_t", "gid_t", "socklen_t", "va_list",
        "std", "string", "wstring", "vector", "map", "unordered_map", "set",
        "unordered_set", "unique_ptr", "shared_ptr", "weak_ptr", "optional",
    }
)

# Curated whitelist of standard/security-relevant APIs and common C/POSIX calls.
# The goal is to preserve transferable API-call signals while abstracting custom
# project-specific identifiers.
SECURITY_RELEVANT_API_WHITELIST: frozenset[str] = frozenset(
    {
        # C string/memory APIs
        "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strncmp", "strlen",
        "strnlen", "strchr", "strrchr", "strstr", "strtok", "strtok_r",
        "strdup", "strndup", "memcpy", "memmove", "memset", "memcmp", "memchr",
        "bcopy", "bzero", "explicit_bzero", "timingsafe_bcmp",
        # formatted I/O and parsing
        "printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf",
        "vsprintf", "vsnprintf", "scanf", "fscanf", "sscanf", "vscanf",
        "vfscanf", "vsscanf", "gets", "fgets", "puts", "fputs", "getc", "getchar",
        "putc", "putchar", "atoi", "atol", "atoll", "strtol", "strtoll", "strtoul",
        "strtoull", "strtod", "strtof", "strtold",
        # allocation/lifetime
        "malloc", "calloc", "realloc", "free", "alloca", "new", "delete",
        "mmap", "munmap", "mprotect", "brk", "sbrk",
        # file descriptors / filesystem
        "open", "openat", "close", "read", "write", "pread", "pwrite", "readv",
        "writev", "lseek", "stat", "lstat", "fstat", "access", "unlink", "remove",
        "rename", "chmod", "chown", "fopen", "freopen", "fclose", "fread",
        "fwrite", "fseek", "ftell", "fflush", "mktemp", "mkstemp", "tmpnam",
        # process/command execution
        "system", "popen", "pclose", "execve", "execl", "execle", "execlp",
        "execv", "execvp", "fork", "vfork", "clone", "wait", "waitpid", "kill",
        # network/socket APIs
        "socket", "bind", "listen", "accept", "accept4", "connect", "send",
        "sendto", "sendmsg", "recv", "recvfrom", "recvmsg", "setsockopt",
        "getsockopt", "select", "poll", "epoll_wait", "getaddrinfo",
        "freeaddrinfo", "inet_ntoa", "inet_ntop", "inet_aton", "inet_pton",
        # crypto-ish / randomness / auth common APIs
        "rand", "srand", "random", "arc4random", "getrandom", "RAND_bytes",
        "EVP_EncryptInit_ex", "EVP_DecryptInit_ex", "EVP_DigestInit_ex",
        "HMAC", "SHA1", "SHA256", "MD5", "SSL_read", "SSL_write",
        "SSL_connect", "SSL_accept", "SSL_get_verify_result",
        # bounds-safe variants / compiler intrinsics
        "strcpy_s", "strncpy_s", "strcat_s", "memcpy_s", "memmove_s",
        "sprintf_s", "snprintf_s", "__builtin_object_size", "__builtin_memcpy",
        "__builtin_strcpy", "__builtin___memcpy_chk", "__builtin___strcpy_chk",
        "__builtin___sprintf_chk", "__builtin___snprintf_chk",
        # assertions / error handling
        "assert", "abort", "exit", "errno", "perror", "strerror", "strerror_r",
    }
)

CONTROL_FLOW_CALL_LIKE_KEYWORDS: frozenset[str] = frozenset(
    {"if", "for", "while", "switch", "catch", "sizeof", "alignof", "decltype", "return"}
)

IDENT_START_RE = re.compile(r"[A-Za-z_]|")  # placeholder; actual checks use functions.


class Token(NamedTuple):
    kind: Literal[
        "identifier", "number", "string", "char", "raw_string", "comment",
        "whitespace", "newline", "operator", "punct", "other"
    ]
    text: str
    start: int
    end: int


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


def _normalize_line_endings_and_unicode(code: str) -> str:
    """Normalize line endings and remove BOM/NUL without Unicode NFKC changes."""
    code = code.lstrip("\ufeff")
    code = code.replace("\x00", "")
    return code.replace("\r\n", "\n").replace("\r", "\n")


def _is_ident_start(ch: str) -> bool:
    return ch == "_" or ch.isalpha()


def _is_ident_continue(ch: str) -> bool:
    return ch == "_" or ch.isalpha() or ch.isdigit()


def _scan_identifier(code: str, i: int) -> tuple[str, int]:
    j = i + 1
    while j < len(code) and _is_ident_continue(code[j]):
        j += 1
    return code[i:j], j


def _scan_line_comment(code: str, i: int) -> tuple[str, int]:
    j = i
    while j < len(code) and code[j] != "\n":
        j += 1
    if j < len(code):
        j += 1
    return code[i:j], j


def _scan_block_comment(code: str, i: int) -> tuple[str, int]:
    j = i + 2
    while j + 1 < len(code):
        if code[j] == "*" and code[j + 1] == "/":
            return code[i:j + 2], j + 2
        j += 1
    logger.warning("Unterminated block comment at index %d", i)
    return code[i:], len(code)


def _scan_ordinary_quoted_literal(code: str, quote_index: int) -> tuple[str, int]:
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
    """Scan u8\"...\", L\"...\", u'...', etc. Returns text, end, kind."""
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
    """Scan C++ raw string literals: R"delim(... )delim" with optional prefix."""
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
    """
    Scan C/C++ numeric-ish literal, including hex/bin prefixes, exponents,
    suffixes, decimal points, and C++14 digit separators such as 1'000.
    """
    n = len(code)
    j = i

    # Hex/bin/octal/decimal with suffixes. This is intentionally lexical and
    # permissive; it prevents apostrophes in 1'000 from being treated as chars.
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
    """Yield simple C/C++ lexical tokens without claiming full parse accuracy."""
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
    parts: list[str] = []
    pending_space = False

    def flush_space() -> None:
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
            # Preserve newline boundaries but not comment content.
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
    if not isinstance(codes, pd.Series):
        codes = pd.Series(list(codes))
    return codes.map(lambda value: normalize_code(value, config=config))


def add_normalized_code_column(
    frame: pd.DataFrame,
    source_column: str = "code",
    target_column: str = "normalized_code",
    config: NormalizationConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    if source_column not in frame.columns:
        raise KeyError(f"Missing source column: {source_column}")
    output = frame.copy()
    output[target_column] = normalize_code_series(output[source_column], config=config)
    return output


def _next_placeholder(mapping: dict[str, str], key: str, prefix: str) -> str:
    if key not in mapping:
        mapping[key] = f"{prefix}{len(mapping) + 1}"
    return mapping[key]


def _previous_significant(tokens: list[str]) -> str:
    for text in reversed(tokens):
        if text.strip():
            return text
    return ""


def _next_significant_token_kind_text(tokens: list[Token], index: int) -> tuple[str, str]:
    j = index + 1
    while j < len(tokens):
        if tokens[j].kind not in {"whitespace", "newline", "comment"}:
            return tokens[j].kind, tokens[j].text
        j += 1
    return "", ""


def _is_function_like_identifier(
    token_index: int,
    tokens: list[Token],
    identifier_text: str,
    previous_text: str,
) -> bool:
    if identifier_text in CONTROL_FLOW_CALL_LIKE_KEYWORDS:
        return False
    if previous_text in {".", "->", "::"}:
        return False
    next_kind, next_text = _next_significant_token_kind_text(tokens, token_index)
    return next_text == "("


def abstract_identifiers(
    code: object,
    config: AbstractionConfig = DEFAULT_ABSTRACTION_CONFIG,
    normalization_config: NormalizationConfig = DEFAULT_CONFIG,
) -> str:
    """
    Return a deterministic per-function abstraction of C/C++ identifiers.

    What is preserved by default:
    - C/C++ keywords and preprocessor directive names
    - common built-in/std types
    - curated security-relevant standard APIs
    - numeric literals
    - operators/punctuation/line boundaries

    What is abstracted:
    - user-defined variables -> VAR1, VAR2, ...
    - user-defined function calls/definitions -> FUNC1, FUNC2, ...
    - struct/class/enum names -> TYPE1, TYPE2, ...
    - field/member names after . or -> -> FIELD1, FIELD2, ...
    - macro-like all-caps identifiers -> MACRO1, MACRO2, ...

    This is not a semantic parser. It is a controlled lexical representation for
    cross-project TF-IDF experiments.
    """
    text = _coerce_code(code)
    if not text:
        return ""
    if config.normalize_first:
        # For abstraction, comments are usually removed unless explicitly kept.
        norm_cfg = NormalizationConfig(
            collapse_horizontal_whitespace=normalization_config.collapse_horizontal_whitespace,
            max_consecutive_blank_lines=normalization_config.max_consecutive_blank_lines,
            preserve_comments=config.preserve_comments,
            preserve_line_breaks=normalization_config.preserve_line_breaks,
            max_input_size=normalization_config.max_input_size,
            max_output_size=normalization_config.max_output_size,
        )
        text = normalize_code(text, config=norm_cfg)
    else:
        text = _normalize_line_endings_and_unicode(text)

    tokens = list(iter_cpp_lexical_tokens(text))
    api_whitelist = SECURITY_RELEVANT_API_WHITELIST | set(config.extra_api_whitelist)

    var_map: dict[str, str] = {}
    func_map: dict[str, str] = {}
    field_map: dict[str, str] = {}
    type_map: dict[str, str] = {}
    macro_map: dict[str, str] = {}

    output: list[str] = []
    significant_output: list[str] = []

    for idx, tok in enumerate(tokens):
        kind = tok.kind
        text_tok = tok.text

        if kind in {"whitespace", "newline"}:
            output.append(text_tok)
            continue

        if kind == "comment":
            if config.preserve_comments:
                output.append(text_tok)
            continue

        if kind in {"string", "raw_string"}:
            if config.preserve_string_literal_content:
                replacement = text_tok
            else:
                replacement = "STR_LITERAL"
            output.append(replacement)
            significant_output.append(replacement)
            continue

        if kind == "char":
            if config.preserve_char_literal_content:
                replacement = text_tok
            else:
                replacement = "CHAR_LITERAL"
            output.append(replacement)
            significant_output.append(replacement)
            continue

        if kind == "number":
            replacement = text_tok if config.preserve_numeric_literals else "NUM_LITERAL"
            output.append(replacement)
            significant_output.append(replacement)
            continue

        if kind != "identifier":
            output.append(text_tok)
            significant_output.append(text_tok)
            continue

        ident = text_tok
        previous = _previous_significant(significant_output)

        should_preserve = False
        if config.preserve_keywords and ident in C_CPP_KEYWORDS:
            should_preserve = True
        if config.preserve_preprocessor_directives and previous == "#" and ident in PREPROCESSOR_DIRECTIVES:
            should_preserve = True
        if config.preserve_builtin_types and ident in BUILTIN_TYPES_AND_COMMON_ALIASES:
            should_preserve = True
        if config.preserve_api_whitelist and ident in api_whitelist:
            should_preserve = True

        if should_preserve:
            replacement = ident
        elif previous in {"struct", "union", "enum", "class", "typename"}:
            replacement = _next_placeholder(type_map, ident, config.type_prefix)
        elif previous in {".", "->"}:
            replacement = _next_placeholder(field_map, ident, config.field_prefix)
        elif ident.isupper() and any(ch.isalpha() for ch in ident) and len(ident) > 1:
            # Common heuristic for project macros/constants.
            replacement = _next_placeholder(macro_map, ident, config.macro_prefix)
        elif _is_function_like_identifier(idx, tokens, ident, previous):
            replacement = _next_placeholder(func_map, ident, config.function_prefix)
        else:
            replacement = _next_placeholder(var_map, ident, config.variable_prefix)

        output.append(replacement)
        significant_output.append(replacement)

    abstracted = "".join(output)
    abstracted = _limit_blank_lines(abstracted, max_consecutive_blank_lines=1)
    if len(abstracted) > MAX_OUTPUT_SIZE:
        raise ValueError(f"Abstracted code exceeds maximum output size {MAX_OUTPUT_SIZE}")
    return abstracted


def abstract_code_series(
    codes: Union[pd.Series, Iterable[object]],
    config: AbstractionConfig = DEFAULT_ABSTRACTION_CONFIG,
    normalization_config: NormalizationConfig = DEFAULT_CONFIG,
) -> pd.Series:
    if not isinstance(codes, pd.Series):
        codes = pd.Series(list(codes))
    return codes.map(lambda value: abstract_identifiers(value, config=config, normalization_config=normalization_config))


def add_abstracted_code_column(
    frame: pd.DataFrame,
    source_column: str = "normalized_code",
    target_column: str = "abstracted_code_v1",
    config: AbstractionConfig = DEFAULT_ABSTRACTION_CONFIG,
    normalization_config: NormalizationConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    if source_column not in frame.columns:
        raise KeyError(f"Missing source column: {source_column}")
    output = frame.copy()
    output[target_column] = abstract_code_series(
        output[source_column], config=config, normalization_config=normalization_config
    )
    return output


def representation_summary(series: pd.Series) -> dict[str, object]:
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


# Backward-compatible alias used by some older tests/notebooks.
normalization_summary = representation_summary
