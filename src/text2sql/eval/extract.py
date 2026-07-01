"""The frozen output-extraction rule: raw model text -> candidate SQL string.

This is the single most bug-prone component in the whole project. A model's raw
output can arrive wrapped in ```sql fences, prefixed with "SQL:", echoing the
"### SQL" scaffold, trailing an explanation, or spanning multiple lines. If
extraction is sloppy, a *correct* query gets mangled into a wrong one and your
execution accuracy silently drops — for every model equally, so you might not
even notice. That's why this lives in one tiny, versioned, heavily-tested module
and is applied identically to base and fine-tuned outputs.

The rule (deterministic, order matters):
  1. take everything after the last "### SQL" scaffold, if the model echoed it
  2. if a fenced code block is present, use its contents (unterminated fences
     handled too)
  3. strip a leading "SQL:"-style label
  4. cut at the first statement terminator ";" (Spider gold has none, so the
     ";" itself is dropped for evaluator compatibility)
  5. collapse all whitespace/newlines to single spaces

`EXTRACTION_VERSION` is recorded alongside results so a change to this rule is
traceable — if the rule changes, prior numbers are no longer comparable.
"""

from __future__ import annotations

import re

__all__ = ["EXTRACTION_VERSION", "extract_sql", "looks_empty"]

EXTRACTION_VERSION = "v1"

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN_FENCE = re.compile(r"^```(?:sql)?\s*", re.IGNORECASE)
_CLOSE_FENCE = re.compile(r"\s*```$")
_LEADING_LABEL = re.compile(r"^\s*sql\s*[:>\-]*\s*", re.IGNORECASE)
_WS = re.compile(r"\s+")


def extract_sql(text: str) -> str:
    """Extract a single candidate SQL query from raw model output.

    Returns "" if there's nothing usable. Idempotent on already-clean SQL.
    """
    if not text:
        return ""

    s = text.strip()

    # (1) If the model echoed the "### SQL" scaffold, keep only what follows.
    if "### SQL" in s:
        s = s.rsplit("### SQL", 1)[-1]

    # (2) Prefer the contents of a fenced code block; handle unterminated fences.
    m = _FENCE.search(s)
    if m:
        s = m.group(1)
    else:
        s = _OPEN_FENCE.sub("", s.strip())
        s = _CLOSE_FENCE.sub("", s)

    # (3) Drop a leading "SQL:" / "sql -" style label.
    s = _LEADING_LABEL.sub("", s.strip())

    # (4) Keep only the first statement; drop the terminating ";".
    s = s.split(";", 1)[0]

    # (5) Normalize whitespace so scoring is insensitive to formatting.
    s = _WS.sub(" ", s).strip()

    return s


def looks_empty(sql: str) -> bool:
    """True if extraction produced nothing meaningful."""
    return not sql or not sql.strip()


# --------------------------------------------------------------------------- #
# Manual check: echo some model output and see what survives extraction.
#   python -m text2sql.eval.extract '```sql
#   SELECT count(*) FROM singer;
#   ```'
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    print(repr(extract_sql(raw)))