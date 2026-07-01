"""The baseline ladder — the fair comparison, assembled.

The headline result is not "the fine-tune scores X" but "the fine-tune beats a
FAIRLY-prompted base model by Y." This module builds that comparison by running
each condition through the exact same harness + execution scorer:

    1. base model, zero-shot   (floor)
    2. base model, few-shot    (the fair baseline to beat)
    3. fine-tuned model         (the result)

Few-shot examples are drawn ONLY from train databases, so the base model never
sees an eval schema. Model loading (base vs base+adapter) is the caller's job —
scripts/evaluate.py loads each and calls evaluate_condition; this module owns the
orchestration and reporting, and computes the paired gap CI that answers "is the
win real?"
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..data.format import Example
from ..data.prompt import FewShotExample, PromptBuilder
from . import stats
from .difficulty import BucketStat, breakdown, format_table
from .execution import ExecutionReport, score_predictions
from .harness import HarnessConfig, Prediction, generate_predictions

__all__ = [
    "ConditionResult",
    "LadderResult",
    "select_few_shot",
    "evaluate_condition",
]


def select_few_shot(
    train: list[Example], k: int = 5, seed: int = 13
) -> list[FewShotExample]:
    """Deterministically pick k in-context examples from TRAIN.

    Fixed across every prompt of the few-shot baseline, and never sourced from
    val/test — so the baseline is fair, not leaky. Deterministic given the seed.
    """
    if k <= 0 or not train:
        return []
    rng = random.Random(seed)
    chosen = rng.sample(train, min(k, len(train)))
    return [FewShotExample(db_id=e.db_id, question=e.question, sql=e.query) for e in chosen]


@dataclass
class ConditionResult:
    name: str
    predictions: list[Prediction]
    report: ExecutionReport

    @property
    def accuracy(self) -> float:
        return self.report.execution_accuracy


def evaluate_condition(
    name: str,
    model,
    tokenizer,
    examples: list[Example],
    builder: PromptBuilder,
    *,
    db_root: str,
    test_suite_root: str | None = None,
    few_shot: list[FewShotExample] | None = None,
    config: HarnessConfig | None = None,
) -> ConditionResult:
    """Run one ladder rung end-to-end: generate -> score."""
    preds = generate_predictions(
        model, tokenizer, examples, builder, few_shot=few_shot or (), config=config
    )
    report = score_predictions(preds, db_root, test_suite_root)
    return ConditionResult(name=name, predictions=preds, report=report)


@dataclass
class LadderResult:
    conditions: list[ConditionResult] = field(default_factory=list)

    def add(self, condition: ConditionResult) -> "LadderResult":
        self.conditions.append(condition)
        return self

    def _get(self, name: str) -> ConditionResult:
        for c in self.conditions:
            if c.name == name:
                return c
        raise KeyError(f"no condition named {name!r}")

    # ---- reporting -------------------------------------------------------- #
    def table(self) -> str:
        """Top-line execution accuracy per condition."""
        width = max((len(c.name) for c in self.conditions), default=9)
        lines = [f"{'condition':<{width}}  {'EX':>7}  {'n':>6}"]
        for c in self.conditions:
            lines.append(f"{c.name:<{width}}  {c.accuracy:>7.4f}  {c.report.n_total:>6}")
        return "\n".join(lines)

    def difficulty_table(self, name: str, hardness_fn=None) -> str:
        c = self._get(name)
        stats_by_bucket = breakdown(c.predictions, c.report, hardness_fn)
        return format_table(stats_by_bucket, name=name)

    def gap_ci(
        self, baseline: str, candidate: str, *, n_boot: int = 10000, seed: int = 0
    ) -> stats.CI:
        """Paired bootstrap CI on (candidate - baseline) execution accuracy.

        Use baseline = the few-shot base, candidate = the fine-tuned model. A
        positive interval that excludes 0 is your defensible 'the fine-tune wins'.
        """
        a = self._get(baseline).report.by_id()
        b = self._get(candidate).report.by_id()
        return stats.paired_gap_ci(a, b, n_boot=n_boot, seed=seed)

    def summary(self, baseline: str, candidate: str, hardness_fn=None) -> str:
        """One block: ladder table, gap CI, and per-difficulty for the winner."""
        parts = ["=== baseline ladder ===", self.table(), ""]
        ci = self.gap_ci(baseline, candidate)
        parts.append(
            f"gap ({candidate} - {baseline}): {ci}  "
            f"{'(excludes 0)' if ci.low > 0 else '(includes 0 — not significant)'}"
        )
        parts.append("")
        parts.append(self.difficulty_table(candidate, hardness_fn))
        return "\n".join(parts)