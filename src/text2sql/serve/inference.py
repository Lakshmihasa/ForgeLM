"""Inference wrapper for serving the fine-tuned text-to-SQL model.

Loads the 4-bit base + LoRA adapter once and answers requests. Critically it
reuses the SAME PromptBuilder (schema serialization + chat template) and the SAME
extract_sql as training and evaluation — so the deployed model behaves exactly
like the one you measured. `serialize_kwargs` MUST match what training/eval used;
that's the serving end of the parity contract.

Two tradeoffs made explicit (both are fair interview questions about "one-click
deploy"):

  * Adapter attached, not merged. On a 4-bit base you can't merge LoRA weights
    in, so the adapter rides alongside — a small per-token overhead for a big
    memory saving. If you serve an fp16 base instead, set merge_adapter=True to
    fold the adapter in and drop that overhead.
  * One request at a time. A threading.Lock serializes GPU access so concurrent
    HTTP requests don't corrupt a shared CUDA context. That's correct but caps
    throughput at one generation at a time — the honest MVP. Real throughput
    needs continuous batching (vLLM), which is the v2 serving path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..data.prompt import PromptBuilder
from ..data.schema import SchemaStore
from ..eval.extract import extract_sql, looks_empty

__all__ = ["InferenceConfig", "GenResult", "Text2SQLModel"]


@dataclass
class InferenceConfig:
    base_model: str
    tables_path: str
    adapter_dir: str | None = None
    serialize_kwargs: dict = field(default_factory=dict)  # MUST match training/eval
    load_in_4bit: bool = True
    compute_dtype: str = "bfloat16"
    max_new_tokens: int = 256
    merge_adapter: bool = False  # only valid on an fp16 base
    device_map: object = field(default_factory=lambda: {"": 0})


@dataclass
class GenResult:
    sql: str
    raw_output: str
    prompt_tokens: int
    completion_tokens: int


def _dtype(name: str) -> torch.dtype:
    if name == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if name in ("bfloat16", "float16"):
        return torch.float16
    return torch.float32


class Text2SQLModel:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self._lock = Lock()

        # Tokenizer: prefer the one saved with the adapter (guaranteed matching
        # special tokens / chat template).
        tok_src = cfg.adapter_dir or cfg.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # for batched generation

        quant = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=_dtype(cfg.compute_dtype),
            )
            if cfg.load_in_4bit
            else None
        )

        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=quant,
            torch_dtype=_dtype(cfg.compute_dtype),
            device_map=cfg.device_map,
        )

        if cfg.adapter_dir:
            model = PeftModel.from_pretrained(model, cfg.adapter_dir)
            if cfg.merge_adapter and not cfg.load_in_4bit:
                model = model.merge_and_unload()

        model.eval()
        model.config.use_cache = True  # serving wants the KV cache ON
        self.model = model

        self.store = SchemaStore(cfg.tables_path)
        self.builder = PromptBuilder(
            self.store, self.tokenizer, serialize_kwargs=cfg.serialize_kwargs
        )

        self.device = str(next(model.parameters()).device)

    # ------------------------------------------------------------------ #
    def has_db(self, db_id: str) -> bool:
        return db_id in self.store

    @torch.no_grad()
    def generate(
        self, question: str, db_id: str, max_new_tokens: int | None = None
    ) -> GenResult:
        if not self.has_db(db_id):
            raise KeyError(f"unknown db_id: {db_id!r}")

        prompt = self.builder.render_inference_prompt(db_id, question)  # zero-shot
        enc = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        prompt_len = enc["input_ids"].shape[1]

        with self._lock:  # serialize GPU access across concurrent requests
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                do_sample=False,  # greedy — deterministic, matches eval
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = out[0, prompt_len:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenResult(
            sql=extract_sql(raw),
            raw_output=raw,
            prompt_tokens=int(prompt_len),
            completion_tokens=int(new_tokens.shape[0]),
        )

    def info(self) -> dict:
        return {
            "model": self.cfg.base_model,
            "adapter": self.cfg.adapter_dir,
            "device": self.device,
        }