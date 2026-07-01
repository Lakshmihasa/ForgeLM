"""The evaluation harness — the parity guarantee, in code.

Every model in this project — base zero-shot, base few-shot, fine-tuned — is run
through THIS function, so the only thing that differs between them is the model
weights and the few-shot examples. Same schema serialization (via the shared
PromptBuilder), same chat template, same decoding, same extraction rule. That is
what makes "the fine-tune beats the base" a fair claim rather than an artifact of
prompting one of them differently.

Scope: the harness produces predictions (extracted SQL). Scoring — execution
accuracy against the databases — is a separate concern (eval/execution.py) and is
injected as a callable, so this module has no dependency on the evaluator and can
be unit-tested with a stub scorer. That same seam is how the training callbacks
get a periodic-EX `eval_fn`.

Decoding is greedy (do_sample=False, temperature 0) so execution accuracy is
reproducible run to run. Generation is batched and LEFT-padded; the model's EOS
naturally stops each sequence (it was trained to emit EOS after the query).
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

from ..data.format import Example
from ..data.prompt import FewShotExample, PromptBuilder
from .extract import EXTRACTION_VERSION, extract_sql

__all__ = [
    "HarnessConfig",
    "Prediction",
    "generate_predictions",
    "evaluate",
    "save_predictions",
    "load_predictions",
]


@dataclass
class HarnessConfig:
    max_new_tokens: int = 256
    batch_size: int = 8
    do_sample: bool = False  # greedy / temperature 0 -> reproducible EX


@dataclass
class Prediction:
    id: str
    db_id: str
    question: str
    gold: str            # gold SQL
    raw_output: str      # exactly what the model produced
    predicted_sql: str   # after extract_sql
    extraction_version: str = EXTRACTION_VERSION


# --------------------------------------------------------------------------- #
# Generation state
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _generation_state(model, tokenizer):
    """Put model+tokenizer into a safe batched-generation state and restore it.

    * eval() so dropout is off
    * use_cache=True for fast generation (it was False during training for
      gradient checkpointing)
    * left padding so generated tokens align at the right edge of the batch
    """
    was_training = model.training
    prev_use_cache = getattr(model.config, "use_cache", False)
    prev_padding_side = tokenizer.padding_side

    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    try:
        yield
    finally:
        tokenizer.padding_side = prev_padding_side
        model.config.use_cache = prev_use_cache
        if was_training:
            model.train()


def _chunks(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# --------------------------------------------------------------------------- #
# Core: predictions
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    examples: Sequence[Example],
    builder: PromptBuilder,
    *,
    few_shot: Sequence[FewShotExample] = (),
    config: HarnessConfig | None = None,
) -> list[Prediction]:
    """Run one model over `examples` and return extracted-SQL predictions.

    `few_shot` is a fixed set prepended to every prompt (empty for zero-shot and
    for the fine-tuned model; the same k examples for the few-shot baseline).
    """
    config = config or HarnessConfig()
    predictions: list[Prediction] = []

    with _generation_state(model, tokenizer):
        for batch in _chunks(list(examples), config.batch_size):
            prompts = [
                builder.render_inference_prompt(ex.db_id, ex.question, few_shot)
                for ex in batch
            ]
            # add_special_tokens=False: the chat template already added BOS/etc.
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)

            out = model.generate(
                **enc,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Keep only newly generated tokens (prompt is left-padded to a fixed len).
            gen = out[:, enc["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for ex, raw in zip(batch, decoded):
                predictions.append(
                    Prediction(
                        id=ex.id,
                        db_id=ex.db_id,
                        question=ex.question,
                        gold=ex.query,
                        raw_output=raw,
                        predicted_sql=extract_sql(raw),
                    )
                )

    return predictions


# --------------------------------------------------------------------------- #
# Predictions + scoring (scorer injected)
# --------------------------------------------------------------------------- #
def evaluate(
    model,
    tokenizer,
    examples: Sequence[Example],
    builder: PromptBuilder,
    scorer: Callable[[Sequence[Prediction]], dict[str, float]],
    *,
    few_shot: Sequence[FewShotExample] = (),
    config: HarnessConfig | None = None,
) -> tuple[list[Prediction], dict[str, float]]:
    """Generate predictions and score them with the injected `scorer`.

    `scorer` is typically eval/execution.execution_accuracy (test-suite EX). This
    is also the shape the training callbacks want: wrap this and return the
    metrics dict as the callback's `eval_fn`.
    """
    predictions = generate_predictions(
        model, tokenizer, examples, builder, few_shot=few_shot, config=config
    )
    metrics = scorer(predictions)
    return predictions, metrics


# --------------------------------------------------------------------------- #
# Persistence — generate once, score / break down / bootstrap offline
# --------------------------------------------------------------------------- #
def save_predictions(predictions: Sequence[Prediction], path: str | Path) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    return len(predictions)


def load_predictions(path: str | Path) -> list[Prediction]:
    out: list[Prediction] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Prediction(**json.loads(line)))
    return out