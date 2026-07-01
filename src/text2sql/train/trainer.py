"""Supervised fine-tuning loop for the text-to-SQL QLoRA model.

Consumes the pieces already built:
  * qlora.build_model / load_tokenizer  -> the 4-bit base + LoRA adapters
  * prompt.PromptBuilder.build_training_example -> tokenized, prompt-MASKED
    examples (loss only on the SQL + EOS)
  * data.format.read_jsonl / splits     -> train / val jsonl

Design choices worth defending:
  * We use plain `transformers.Trainer` on PRE-TOKENIZED examples rather than
    letting a higher-level trainer re-format text. The whole project rests on the
    training prompt being byte-identical to the eval prompt, and that parity is
    guaranteed by PromptBuilder — so we must not hand the trainer raw text it
    might re-template or re-tokenize with different special tokens.
  * Optimizer is `paged_adamw_8bit`: 8-bit Adam states (a quarter the memory of
    fp32 moments) with paging to CPU to survive memory spikes — a QLoRA staple.
  * Training is zero-shot (no in-context examples): the model learns the mapping
    from the target, so few-shot is empty here. Few-shot is a *baseline* trick,
    not a fine-tuning one.
  * Eval during training is eval LOSS (cheap, automatic). Execution accuracy is
    the metric that actually matters, but it needs generation + the evaluator, so
    it's added via ExecutionAccuracyCallback (callbacks.py) rather than baked in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from ..data.format import read_jsonl
from ..data.prompt import PromptBuilder
from ..data.schema import SchemaStore
from .qlora import QLoRAConfig, build_model, load_tokenizer

__all__ = ["TrainConfig", "CausalDataCollator", "train"]


@dataclass
class TrainConfig:
    output_dir: str = "outputs/llama3-8b-qlora-r16"

    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch = 4 * 4 = 16
    learning_rate: float = 2e-4  # standard QLoRA LR
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3

    max_seq_length: int = 1024
    optim: str = "paged_adamw_8bit"

    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"  # see note on EX-based selection
    greater_is_better: bool = False

    seed: int = 13
    report_to: str = "wandb"
    run_name: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class CausalDataCollator:
    """Pads a batch of pre-tokenized causal-LM examples.

    input_ids  -> pad_token_id
    attention  -> 0
    labels     -> -100 (so padding never contributes to the loss)

    Respects the tokenizer's padding_side (right during training).
    """

    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        right = self.tokenizer.padding_side != "left"

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            n_pad = max_len - len(f["input_ids"])
            ids_pad = [pad_id] * n_pad
            att_pad = [0] * n_pad
            lbl_pad = [self.label_pad_token_id] * n_pad
            if right:
                batch["input_ids"].append(f["input_ids"] + ids_pad)
                batch["attention_mask"].append(f["attention_mask"] + att_pad)
                batch["labels"].append(f["labels"] + lbl_pad)
            else:
                batch["input_ids"].append(ids_pad + f["input_ids"])
                batch["attention_mask"].append(att_pad + f["attention_mask"])
                batch["labels"].append(lbl_pad + f["labels"])

        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def _build_dataset(
    jsonl_path: str,
    builder: PromptBuilder,
    max_length: int,
) -> Dataset:
    """Tokenize a jsonl split into masked training examples (zero-shot prompts)."""
    examples = read_jsonl(jsonl_path)
    rows = [
        builder.build_training_example(
            ex.db_id, ex.question, ex.query, max_length=max_length
        )
        for ex in examples
    ]
    return Dataset.from_list(rows)


def train(
    qlora_cfg: QLoRAConfig,
    train_cfg: TrainConfig,
    *,
    tables_path: str,
    train_jsonl: str,
    val_jsonl: str | None = None,
    serialize_kwargs: dict | None = None,
    callbacks: Sequence[TrainerCallback] | None = None,
) -> Trainer:
    """Run the fine-tune and save the LoRA adapter to `output_dir`.

    `serialize_kwargs` must match what evaluation will use (the schema
    representation is part of the frozen contract).
    """
    schema_store = SchemaStore(tables_path)
    tokenizer = load_tokenizer(qlora_cfg)
    builder = PromptBuilder(
        schema_store, tokenizer, serialize_kwargs=serialize_kwargs
    )

    model = build_model(qlora_cfg)

    train_ds = _build_dataset(train_jsonl, builder, train_cfg.max_seq_length)
    eval_ds = (
        _build_dataset(val_jsonl, builder, train_cfg.max_seq_length)
        if val_jsonl
        else None
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        optim=train_cfg.optim,
        bf16=bf16,
        fp16=not bf16,
        # Gradient checkpointing is already enabled on the model in build_model;
        # don't re-enable it here or the use_reentrant setting can conflict.
        gradient_checkpointing=False,
        logging_steps=train_cfg.logging_steps,
        eval_strategy="steps" if eval_ds is not None else "no",  # older HF: evaluation_strategy
        eval_steps=train_cfg.eval_steps,
        save_strategy="steps",
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=train_cfg.load_best_model_at_end and eval_ds is not None,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=train_cfg.greater_is_better,
        report_to=[train_cfg.report_to] if train_cfg.report_to else [],
        run_name=train_cfg.run_name,
        seed=train_cfg.seed,
        remove_unused_columns=False,  # our columns ARE the model inputs
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalDataCollator(tokenizer),
        callbacks=list(callbacks) if callbacks else None,
    )

    trainer.train()

    # Save the adapter (not the full base) + tokenizer.
    trainer.model.save_pretrained(train_cfg.output_dir)
    tokenizer.save_pretrained(train_cfg.output_dir)
    return trainer


# NOTE on checkpoint selection:
# The eval spec calls for selecting on held-out EXECUTION ACCURACY, not loss.
# eval_loss is a cheap proxy and the default here. For true EX-based selection,
# attach ExecutionAccuracyCallback (callbacks.py) so val EX is logged each eval,
# then either eyeball the best checkpoint from the W&B history or set
# metric_for_best_model to the callback's logged key (greater_is_better=True).