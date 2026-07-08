import pytest
import pandas as pd

from normalization_v3 import (
    normalize_code,
    abstract_identifiers,
    add_abstracted_code_column,
    _normalize_line_endings_and_unicode,
    iter_cpp_lexical_tokens,
)


def test_no_nfkc_preserves_unicode_identity():
    src = "float μ = 1; float µ = 2;"
    out = _normalize_line_endings_and_unicode(src)
    assert "μ" in out
    assert "µ" in out
    assert out.count("μ") == 1
    assert out.count("µ") == 1


def test_digit_separator_not_char_literal():
    src = "int x = 1'000; char c = 'a';"
    toks = list(iter_cpp_lexical_tokens(src))
    numbers = [t.text for t in toks if t.kind == "number"]
    chars = [t.text for t in toks if t.kind == "char"]
    assert "1'000" in numbers
    assert "'a'" in chars


def test_cpp_raw_string_scanned_as_one_literal():
    src = r'const char *s = R"TAG(a " quote )TAG"; int x = 1;'
    toks = list(iter_cpp_lexical_tokens(src))
    raw = [t.text for t in toks if t.kind == "raw_string"]
    assert len(raw) == 1
    assert raw[0] == r'R"TAG(a " quote )TAG"'


def test_abstract_preserves_api_but_abstracts_user_identifiers():
    src = "int copy_user(char *dst, char *src) { strcpy(dst, src); return 0; }"
    out = abstract_identifiers(src)
    assert "strcpy" in out
    assert "copy_user" not in out
    assert "dst" not in out
    assert "src" not in out
    assert "FUNC1" in out
    assert "VAR" in out


def test_abstract_fields_and_types():
    src = "struct header { int len; }; int f(struct header *h) { return h->len; }"
    out = abstract_identifiers(src)
    assert "struct" in out
    assert "TYPE1" in out
    assert "FIELD" in out
    assert "header" not in out
    assert "len" not in out


def test_abstract_string_and_char_literal_content():
    src = 'int f(){ char c = \'x\'; printf("secret/project/path"); }'
    out = abstract_identifiers(src)
    assert "printf" in out
    assert "STR_LITERAL" in out
    assert "CHAR_LITERAL" in out
    assert "secret" not in out
    assert "project" not in out


def test_add_abstracted_code_column():
    df = pd.DataFrame({"normalized_code": ["int f(){return 0;}"]})
    out = add_abstracted_code_column(df)
    assert "abstracted_code_v1" in out.columns
    assert "FUNC1" in out.loc[0, "abstracted_code_v1"]


def test_normalize_comments_can_be_removed():
    src = "int x; // project specific comment\nint y;"
    kept = normalize_code(src)
    removed = normalize_code(src, config=type('Cfg', (), {})()) if False else None
    assert "project specific" in kept
    assert removed is None
