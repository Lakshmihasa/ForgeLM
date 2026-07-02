"""Adapter: official Spider hardness buckets (optional).

difficulty.py ships a working string heuristic and takes an injected `hardness_fn`.
This wires the OFFICIAL buckets instead. Because official hardness requires
PARSING each query against its schema, you can't compute it from the SQL string
alone — so the pattern is: precompute a {gold_sql: bucket} map once (you have the
schemas at eval time), then inject a lookup.

    from text2sql.eval.official.hardness import precompute_hardness, make_hardness_fn
    hmap = precompute_hardness(test_examples, db_root="data/raw/spider/database")
    fn = make_hardness_fn(hmap)   # falls back to the heuristic on misses
    print(ladder.difficulty_table("finetuned", hardness_fn=fn))

Requires official/_vendor/ populated with Spider's evaluation.py + process_sql.py.
The upstream API (Evaluator, get_sql, Schema, get_schema) follows the Yale Spider
repo; adjust the imports here if your vendored version renames them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

__all__ = ["official_hardness", "precompute_hardness", "make_hardness_fn"]


def _load_upstream():
    """Import the vendored parser/evaluator lazily (raises if not vendored)."""
    from ._vendor.process_sql import Schema, get_schema, get_sql  # type: ignore
    from ._vendor.evaluation import Evaluator  # type: ignore

    return Schema, get_schema, get_sql, Evaluator


def official_hardness(gold_sql: str, db_path: str) -> str:
    """Bucket one gold query using the official evaluator + the db's schema."""
    Schema, get_schema, get_sql, Evaluator = _load_upstream()
    schema = Schema(get_schema(db_path))
    parsed = get_sql(schema, gold_sql)
    return Evaluator().eval_hardness(parsed)


def precompute_hardness(
    examples: Iterable,
    db_root: str,
) -> dict[str, str]:
    """Build {gold_sql: bucket} for a set of examples (each needs .db_id, .query).

    Best-effort: any query that fails to parse is skipped, so a single bad row
    can't abort the whole breakdown. Keyed by gold SQL string — the same query
    parses to the same structure and thus the same bucket regardless of db.
    """
    out: dict[str, str] = {}
    for ex in examples:
        db_path = str(Path(db_root) / ex.db_id / f"{ex.db_id}.sqlite")
        try:
            out[ex.query] = official_hardness(ex.query, db_path)
        except Exception:
            continue
    return out


def make_hardness_fn(hmap: dict[str, str]) -> Callable[[str], str]:
    """Turn a precomputed map into a difficulty.py-compatible hardness_fn.

    Falls back to difficulty's string heuristic for any query missing from the
    map (e.g. one that failed to parse), so the breakdown always classifies.
    """
    from ..difficulty import _heuristic_hardness

    def hardness_fn(gold_sql: str) -> str:
        return hmap.get(gold_sql) or _heuristic_hardness(gold_sql)

    return hardness_fn