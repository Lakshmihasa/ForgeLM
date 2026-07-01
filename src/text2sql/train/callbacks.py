"""Training callbacks: real-metric eval and qualitative sample logging.

The Trainer already logs loss / learning-rate / grad-norm / eval-loss to W&B via
`report_to="wandb"`, so there's no need for a loss-logging callback. What's
missing — and what actually tells you whether the fine-tune is working — is
*execution accuracy on the held-out val set*, plus eyeballing a few generations
to catch degenerate output early.

Both callbacks are decoupled from the eval package by dependency injection: you
pass in a callable that does the real work. Once `eval/harness.py` exists you
wrap it in a small function and hand it here — so this module has no import on
scoring code and stays testable with a stub.

Generation-mode gotchas these callbacks handle (each is a real footgun):
  * model.eval() during generation, restore train() after
  * re-enable the KV cache (use_cache=True) for fast generation, restore False
    (it was disabled for gradient checkpointing during training)
  * left-pad for batched generation, restore right padding for training
"""

from __future__ import annotations

import contextlib
from typing import Callable

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

try:  # optional; callbacks degrade to printing if wandb isn't active
    import wandb

    _HAS_WANDB = True
except Exception:  # pragma: no cover
    _HAS_WANDB = False

__all__ = [
    "generation_mode",
    "ExecutionAccuracyCallback",
    "SampleGenerationsCallback",
]

# eval_fn:   (model, tokenizer) -> {metric_name: value}
EvalFn = Callable[[object, object], dict[str, float]]
# sample_fn: (model, tokenizer, n) -> list of {"question","db_id","predicted","gold"}
SampleFn = Callable[[object, object, int], list[dict[str, str]]]


@contextlib.contextmanager
def generation_mode(model, tokenizer):
    """Temporarily flip model + tokenizer into a safe batched-generation state,
    then restore the training state exactly. Use this to wrap any generation done
    from inside a callback."""
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


class ExecutionAccuracyCallback(TrainerCallback):
    """Run held-out execution accuracy each time the Trainer evaluates.

    Parameters
    ----------
    eval_fn : callable(model, tokenizer) -> {metric: value}
        Typically wraps eval/harness on a fixed val subset. Return e.g.
        {"execution_accuracy": 0.63}.
    prefix : str
        Namespaces the logged keys, e.g. "val" -> "val/execution_accuracy".
    """

    def __init__(self, eval_fn: EvalFn, prefix: str = "val"):
        self.eval_fn = eval_fn
        self.prefix = prefix

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        metrics: dict | None = None,
        **kwargs,
    ):
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if model is None or tokenizer is None:
            return control

        with generation_mode(model, tokenizer):
            results = self.eval_fn(model, tokenizer)

        namespaced = {f"{self.prefix}/{k}": float(v) for k, v in results.items()}

        # Surface in the console.
        pretty = "  ".join(f"{k}={v:.4f}" for k, v in namespaced.items())
        print(f"[step {state.global_step}] {pretty}")

        # Log to W&B at this step.
        if _HAS_WANDB and wandb.run is not None:
            wandb.log(namespaced, step=state.global_step)

        # Also fold into the metrics dict so metric_for_best_model can select on
        # it (set metric_for_best_model to e.g. "val/execution_accuracy",
        # greater_is_better=True). Best-effort across HF versions.
        if metrics is not None:
            metrics.update(namespaced)
            # HF's best-metric tracking expects an "eval_"-prefixed key; mirror it.
            for k, v in namespaced.items():
                metrics.setdefault(f"eval_{k.replace('/', '_')}", v)

        return control


class SampleGenerationsCallback(TrainerCallback):
    """Log a handful of (question, predicted SQL, gold SQL) rows each eval, so
    degenerate output (empty strings, repetition, prose instead of SQL) is caught
    by eye long before the final run.

    Parameters
    ----------
    sample_fn : callable(model, tokenizer, n) -> list[dict]
        Returns rows with keys: question, db_id, predicted, gold.
    num_samples : int
    """

    def __init__(self, sample_fn: SampleFn, num_samples: int = 8):
        self.sample_fn = sample_fn
        self.num_samples = num_samples

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if model is None or tokenizer is None:
            return control

        with generation_mode(model, tokenizer):
            rows = self.sample_fn(model, tokenizer, self.num_samples)

        if _HAS_WANDB and wandb.run is not None:
            table = wandb.Table(columns=["question", "db_id", "predicted", "gold"])
            for r in rows:
                table.add_data(r["question"], r["db_id"], r["predicted"], r["gold"])
            wandb.log({"samples": table}, step=state.global_step)
        else:
            print(f"--- sample generations @ step {state.global_step} ---")
            for r in rows[: min(3, len(rows))]:
                print(f"Q: {r['question']}")
                print(f"  pred: {r['predicted']}")
                print(f"  gold: {r['gold']}")

        return control