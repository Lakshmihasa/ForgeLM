"""Tests for execution-accuracy scoring against a real sqlite database.

Forces the local fallback comparison (no vendored official evaluator) so the
result-set / ordering logic is what's under test.
"""

import sqlite3

import pytest

torch = pytest.importorskip("torch")  # execution imports harness, which needs torch

from text2sql.eval.execution import make_scorer, score_predictions  # noqa: E402
from text2sql.eval.harness import Prediction  # noqa: E402


@pytest.fixture(autouse=True)
def _force_fallback(monkeypatch):
    monkeypatch.setattr("text2sql.eval.execution._HAS_OFFICIAL", False)


@pytest.fixture
def db_root(tmp_path):
    """Create db_root/mydb/mydb.sqlite with a 3-row table."""
    db_dir = tmp_path / "mydb"
    db_dir.mkdir()
    conn = sqlite3.connect(db_dir / "mydb.sqlite")
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b"), (3, "c")])
    conn.commit()
    conn.close()
    return str(tmp_path)


def _pred(gold, predicted, db_id="mydb", pid="x"):
    return Prediction(
        id=pid, db_id=db_id, question="q", gold=gold,
        raw_output=predicted, predicted_sql=predicted,
    )


def test_correct_when_same_result(db_root):
    p = _pred("SELECT id FROM t", "SELECT id FROM t WHERE id <= 3")
    assert score_predictions([p], db_root).results[0].correct


def test_incorrect_when_different_result(db_root):
    p = _pred("SELECT id FROM t", "SELECT id FROM t WHERE id = 1")
    assert not score_predictions([p], db_root).results[0].correct


def test_invalid_sql_is_incorrect(db_root):
    p = _pred("SELECT id FROM t", "SELECT nope FROM t")
    assert not score_predictions([p], db_root).results[0].correct


def test_empty_prediction_is_incorrect(db_root):
    p = _pred("SELECT id FROM t", "")
    r = score_predictions([p], db_root).results[0]
    assert not r.correct and r.error == "empty prediction"


def test_missing_db_is_incorrect(db_root):
    p = _pred("SELECT id FROM t", "SELECT id FROM t", db_id="ghost")
    r = score_predictions([p], db_root).results[0]
    assert not r.correct and "missing db" in (r.error or "")


def test_order_insensitive_when_gold_has_no_order_by(db_root):
    # gold has no ORDER BY -> row order should not matter
    p = _pred("SELECT id FROM t", "SELECT id FROM t ORDER BY id DESC")
    assert score_predictions([p], db_root).results[0].correct


def test_order_sensitive_when_gold_has_order_by(db_root):
    # gold orders ascending; prediction orders descending -> mismatch
    p = _pred("SELECT id FROM t ORDER BY id", "SELECT id FROM t ORDER BY id DESC")
    assert not score_predictions([p], db_root).results[0].correct


def test_report_aggregate_and_by_id(db_root):
    preds = [
        _pred("SELECT id FROM t", "SELECT id FROM t", pid="ok"),
        _pred("SELECT id FROM t", "SELECT id FROM t WHERE id = 1", pid="bad"),
    ]
    report = score_predictions(preds, db_root)
    assert report.n_total == 2 and report.n_correct == 1
    assert report.execution_accuracy == 0.5
    assert report.by_id() == {"ok": True, "bad": False}


def test_make_scorer_returns_expected_shape(db_root):
    scorer = make_scorer(db_root)
    out = scorer([_pred("SELECT id FROM t", "SELECT id FROM t")])
    assert set(out) == {"execution_accuracy"} and out["execution_accuracy"] == 1.0