"""Tests for the frozen output-extraction rule.

extract.py is pure (only `re`), so these run anywhere with zero heavy deps —
the most fundamental guarantee in the repo is also the cheapest to check.
"""

from text2sql.eval.extract import EXTRACTION_VERSION, extract_sql, looks_empty


def test_plain_sql_passes_through_and_collapses_whitespace():
    assert extract_sql("SELECT   a\nFROM   t") == "SELECT a FROM t"


def test_strips_terminated_sql_fence():
    assert extract_sql("```sql\nSELECT a FROM t\n```") == "SELECT a FROM t"


def test_strips_plain_fence_without_language():
    assert extract_sql("```\nSELECT a FROM t\n```") == "SELECT a FROM t"


def test_handles_unterminated_fence():
    assert extract_sql("```sql\nSELECT a FROM t") == "SELECT a FROM t"


def test_takes_text_after_scaffold():
    assert extract_sql("### SQL\nSELECT a FROM t") == "SELECT a FROM t"


def test_scaffold_then_fence():
    raw = "### SQL\n```sql\nSELECT a FROM t\n```"
    assert extract_sql(raw) == "SELECT a FROM t"


def test_strips_leading_label():
    assert extract_sql("SQL: SELECT a FROM t") == "SELECT a FROM t"


def test_drops_trailing_prose_after_semicolon():
    raw = "SELECT a FROM t; this query returns all rows in table t."
    assert extract_sql(raw) == "SELECT a FROM t"


def test_keeps_only_first_statement():
    raw = "SELECT a FROM t; SELECT b FROM u"
    assert extract_sql(raw) == "SELECT a FROM t"


def test_empty_and_whitespace():
    assert extract_sql("") == ""
    assert extract_sql("   \n\t") == ""
    assert looks_empty(extract_sql(""))
    assert not looks_empty("SELECT 1")


def test_idempotent_on_clean_sql():
    once = extract_sql("```sql\nSELECT a FROM t;\n```")
    assert extract_sql(once) == once


def test_version_is_stamped():
    assert isinstance(EXTRACTION_VERSION, str) and EXTRACTION_VERSION