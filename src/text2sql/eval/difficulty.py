"""Per-difficulty breakdown of execution accuracy.

A single aggregate number hides the story. Spider tags each query easy / medium /
hard / extra by structural complexity (joins, nesting, aggregations). The
convincing result is the gap between base and fine-tuned OPENING on the hard /
extra buckets — that shows the model learned the difficult structures, not just
easy lookups. If gains sit only in `easy`, that's a real signal to report
honestly, not hide.

Hardness is properly computed by the official Spider evaluator from the PARSED
sql. If you've vendored it, pass its `eval_hardness` as `hardness_fn`. The
built-in fallback approximates Spider's component-counting rules from the gold
SQL string — good enough to structure the analysis, but note in your writeup
which one produced the reported buckets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .execution import ExecutionReport
from .harness import Prediction

__all__ = ["BUCKETS", "BucketStat", "classify_hardness", "breakdown", "format_table"]

BUCKETS = ("easy", "medium", "hard", "extra")

# --- heuristic component detectors (string-level approximation) ------------- #
_JOIN = re.compile(r"\bjoin\b", re.IGNORECASE)
_SELECT = re.compile(r"\bselect\b", re.IGNORECASE)
_AGG = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)
_SETOP = re.compile(r"\b(union|except|intersect)\b", re.IGNORECASE)


def _present(pattern: str, sql: str) -> int:
    return 1 if re.search(pattern, sql, re.IGNORECASE) else 0


def _heuristic_hardness(gold_sql: str) -> str:
    """Approximate Spider's easy/medium/hard/extra from the gold SQL string.

    Mirrors the spirit of Spider's component counts:
      comp1  : where, group by, order by, limit, joins, or, like
      comp2  : nested selects + set operations (union/except/intersect)
      others : aggregations + multiple AND-conditions
    then applies thresholds close to the official rules.
    """
    s = gold_sql

    comp1 = (
        _present(r"\bwhere\b", s)
        + _present(r"\bgroup\s+by\b", s)
        + _present(r"\border\s+by\b", s)
        + _present(r"\blimit\b", s)
        + len(_JOIN.findall(s))
        + _present(r"\bor\b", s)
        + _present(r"\blike\b", s)
    )
    comp2 = max(0, len(_SELECT.findall(s)) - 1) + len(_SETOP.findall(s))
    others = len(_AGG.findall(s)) + (1 if len(re.findall(r"\band\b", s, re.IGNORECASE)) > 1 else 0)

    if comp1 <= 1 and others == 0 and comp2 == 0:
        return "easy"
    if (others <= 2 and comp1 <= 1 and comp2 == 0) or (
        comp1 <= 2 and others < 2 and comp2 == 0
    ):
        return "medium"
    if (
        (others > 2 and comp1 <= 2 and comp2 == 0)
        or (2 < comp1 <= 3 and others <= 2 and comp2 == 0)
        or (comp1 <= 1 and others == 0 and comp2 <= 1)
    ):
        return "hard"
    return "extra"


def classify_hardness(
    gold_sql: str, hardness_fn: Callable[[str], str] | None = None
) -> str:
    """Bucket a gold query. Prefers an injected official hardness_fn."""
    if hardness_fn is not None:
        return hardness_fn(gold_sql)
    return _heuristic_hardness(gold_sql)


@dataclass
class BucketStat:
    n: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


def breakdown(
    predictions: Sequence[Prediction],
    report: ExecutionReport,
    hardness_fn: Callable[[str], str] | None = None,
) -> dict[str, BucketStat]:
    """Join predictions (for gold SQL -> hardness) with results (for correctness)."""
    correct_by_id = report.by_id()
    stats = {b: BucketStat() for b in BUCKETS}
    for p in predictions:
        if p.id not in correct_by_id:
            continue
        bucket = classify_hardness(p.gold, hardness_fn)
        stats[bucket].n += 1
        stats[bucket].correct += int(correct_by_id[p.id])
    return stats


def format_table(stats: dict[str, BucketStat], *, name: str = "") -> str:
    """Render a per-difficulty table (also fine for the README / writeup)."""
    total_n = sum(s.n for s in stats.values())
    total_c = sum(s.correct for s in stats.values())
    header = f"{name}  " if name else ""
    lines = [f"{header}execution accuracy by difficulty:"]
    lines.append(f"  {'bucket':<8} {'n':>6} {'correct':>8} {'acc':>7}")
    for b in BUCKETS:
        s = stats[b]
        lines.append(f"  {b:<8} {s.n:>6} {s.correct:>8} {s.accuracy:>7.3f}")
    overall = total_c / total_n if total_n else 0.0
    lines.append(f"  {'ALL':<8} {total_n:>6} {total_c:>8} {overall:>7.3f}")
    return "\n".join(lines)