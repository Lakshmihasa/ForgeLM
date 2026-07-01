"""Execution accuracy — the metric.

Scores predictions by RUNNING them: execute the predicted SQL and the gold SQL
against the database, compare the result sets. Two queries that look nothing
alike but return the same rows both count as correct; a plausible-looking query
that returns the wrong rows is correctly marked wrong. That objectivity is why
this domain was chosen.

Test-suite execution accuracy: a prediction is correct only if it matches gold
on EVERY seeded instance of the database. A single instance has false positives
(e.g. two different queries that both return an empty set); requiring agreement
across several differently-seeded instances catches them. Point `test_suite_root`
at the multi-instance databases to enable this; otherwise it degrades to plain
single-DB execution accuracy.

We prefer the vendored official evaluator (eval/official/) for the actual
match — result-set comparison has real subtleties (ORDER BY sensitivity, SELECT *
column permutations) that are easy to get wrong. A careful fallback is provided
so the module runs without the official code, but validate against the official
evaluator before publishing numbers.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .harness import Prediction

__all__ = [
    "ExampleResult",
    "ExecutionReport",
    "score_predictions",
    "make_scorer",
]

# Prefer the official evaluator's match function if it's been vendored.
try:  # pragma: no cover - depends on vendored code being present
    from .official.exec_eval import eval_exec_match as _official_exec_match

    _HAS_OFFICIAL = True
except Exception:
    _HAS_OFFICIAL = False

_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)
_QUERY_TIMEOUT_S = 10.0


@dataclass
class ExampleResult:
    id: str
    db_id: str
    correct: bool
    error: str | None = None  # predicted-query execution error, if any


@dataclass
class ExecutionReport:
    results: list[ExampleResult]

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_correct(self) -> int:
        return sum(r.correct for r in self.results)

    @property
    def execution_accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total else 0.0

    def by_id(self) -> dict[str, bool]:
        """{example_id: correct} — the paired vector stats.py bootstraps over."""
        return {r.id: r.correct for r in self.results}


# --------------------------------------------------------------------------- #
# Low-level execution + comparison (fallback path)
# --------------------------------------------------------------------------- #
def _run_query(db_path: str, sql: str, timeout_s: float = _QUERY_TIMEOUT_S):
    """Execute `sql`; return (rows, None) or (None, error_message).

    A time-based progress handler aborts pathological queries so one bad
    prediction can't hang the whole eval.
    """
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.time() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 1000)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall(), None
    except Exception as e:  # invalid SQL, timeout, missing column, ...
        return None, str(e)
    finally:
        conn.close()


def _rows_match(gold_rows, pred_rows, order_sensitive: bool) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    if order_sensitive:
        return list(gold_rows) == list(pred_rows)
    # order-insensitive: compare as multisets of rows
    try:
        return Counter(map(tuple, gold_rows)) == Counter(map(tuple, pred_rows))
    except TypeError:
        # unhashable cell (rare); fall back to sorted comparison by string repr
        return sorted(map(repr, gold_rows)) == sorted(map(repr, pred_rows))


def _match_on_instance(db_path: str, predicted: str, gold: str) -> bool:
    """Whether predicted and gold agree on one database instance."""
    if _HAS_OFFICIAL:
        try:
            return bool(_official_exec_match(db_path, predicted, gold))
        except Exception:
            pass  # fall through to the local comparison
    order_sensitive = bool(_ORDER_BY.search(gold))
    gold_rows, gold_err = _run_query(db_path, gold)
    if gold_err is not None:
        # A gold query that won't run means the instance is unusable; skip it by
        # treating as non-informative (caller handles no-usable-instance case).
        return False
    pred_rows, pred_err = _run_query(db_path, predicted)
    if pred_err is not None:
        return False
    return _rows_match(gold_rows, pred_rows, order_sensitive)


# --------------------------------------------------------------------------- #
# DB path resolution
# --------------------------------------------------------------------------- #
def _instance_paths(
    db_id: str, db_root: str, test_suite_root: str | None
) -> list[str]:
    """All database instances to test this db_id against.

    Test-suite mode: every *.sqlite under test_suite_root/<db_id>/. Plain mode:
    the single db_root/<db_id>/<db_id>.sqlite.
    """
    if test_suite_root:
        suite_dir = Path(test_suite_root) / db_id
        if suite_dir.is_dir():
            instances = sorted(str(p) for p in suite_dir.glob("*.sqlite"))
            if instances:
                return instances
    return [str(Path(db_root) / db_id / f"{db_id}.sqlite")]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_predictions(
    predictions: Sequence[Prediction],
    db_root: str,
    test_suite_root: str | None = None,
) -> ExecutionReport:
    """Score predictions with (test-suite) execution accuracy.

    A prediction is correct iff it matches gold on ALL usable instances of its
    database. An empty prediction, an invalid query, or disagreement on any
    instance -> incorrect.
    """
    results: list[ExampleResult] = []
    for p in predictions:
        if not p.predicted_sql.strip():
            results.append(ExampleResult(p.id, p.db_id, False, error="empty prediction"))
            continue

        paths = _instance_paths(p.db_id, db_root, test_suite_root)
        correct = True
        err: str | None = None
        for db_path in paths:
            if not Path(db_path).exists():
                correct, err = False, f"missing db: {db_path}"
                break
            if not _match_on_instance(db_path, p.predicted_sql, p.gold):
                correct = False
                break
        results.append(ExampleResult(p.id, p.db_id, correct, error=err))

    return ExecutionReport(results=results)


def make_scorer(
    db_root: str, test_suite_root: str | None = None
) -> Callable[[Sequence[Prediction]], dict[str, float]]:
    """Build a harness-compatible scorer bound to these database directories.

    Returns {"execution_accuracy": ...} — the shape harness.evaluate and the
    training callbacks expect.
    """

    def scorer(predictions: Sequence[Prediction]) -> dict[str, float]:
        report = score_predictions(predictions, db_root, test_suite_root)
        return {"execution_accuracy": report.execution_accuracy}

    return scorer
