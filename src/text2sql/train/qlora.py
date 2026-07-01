"""QLoRA model + config for text-to-SQL fine-tuning.

QLoRA lets a 7-8B model fine-tune on a single ~16-24 GB GPU. Where the memory
goes, and how QLoRA cuts it (this is the first question every interviewer asks):

  Full fine-tuning of a 7B model in mixed precision needs, very roughly:
    * weights            ~14 GB (fp16)
    * gradients          ~14 GB (one per trainable weight)
    * Adam optimizer     ~28 GB+ (two fp32 moments per weight, often + fp32 copy)
    * activations        variable
  -> well over 60 GB. QLoRA changes the arithmetic on three axes:
    1. the base model is loaded in 4-bit NF4, so the frozen weights are ~1/4 the
       size and carry NO gradient / optimizer state (they're frozen);
    2. only the low-rank LoRA adapters are trainable — typically <1% of params —
       so gradients + Adam states exist for that tiny slice only;
    3. gradient checkpointing recomputes activations in the backward pass instead
       of storing them, trading compute for memory.
  NF4 (normal-float-4) is information-theoretically suited to the roughly-normal
  distribution of pretrained weights; double quantization further quantizes the
  quantization constants (~0.4 bits/param saved); compute still happens in
  bf16, so only storage is 4-bit.

This module owns the config and model construction. Training loop lives in
trainer.py; hyperparameters live in configs/train/*.yaml and load into
QLoRAConfig via from_dict.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

__all__ = ["QLoRAConfig", "bnb_config", "lora_config", "load_tokenizer", "build_model"]


# Attention + MLP projections. This covers essentially all linear layers in
# Llama-3 / Mistral except the embeddings and the LM head. The QLoRA paper found
# adapting *all* linear layers (not just q/v) is what lets LoRA match full
# fine-tuning quality — so target the full set, not the minimal one.
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class QLoRAConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # LoRA
    r: int = 16
    lora_alpha: int = 32  # effective scaling = lora_alpha / r (here 2.0)
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))

    # 4-bit quantization
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: str = "bfloat16"  # bf16 compute; falls back to fp16 if unsupported

    # runtime
    gradient_checkpointing: bool = True
    attn_implementation: str | None = "sdpa"  # or "flash_attention_2" if installed
    device_map: object = field(default_factory=lambda: {"": 0})  # single GPU
    trust_remote_code: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "QLoRAConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        warnings.warn("bf16 unsupported on this GPU; falling back to fp16.")
        return torch.float16
    if name == "float16":
        return torch.float16
    return torch.float32


def bnb_config(cfg: QLoRAConfig) -> BitsAndBytesConfig:
    """4-bit NF4 quantization config for loading the frozen base model."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_resolve_dtype(cfg.compute_dtype),
    )


def lora_config(cfg: QLoRAConfig) -> LoraConfig:
    """The trainable low-rank adapters."""
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=cfg.task_type,
        target_modules=cfg.target_modules,
    )


def load_tokenizer(cfg: QLoRAConfig) -> PreTrainedTokenizerBase:
    """Tokenizer with a pad token and right padding for training.

    Many base checkpoints (Llama-3, Mistral) ship without a pad token; reusing
    EOS is standard. Pad positions are masked out of the loss by the collator, so
    the model still learns EOS properly. Training pads on the RIGHT; batched
    *generation* at eval time pads on the LEFT (handled in the serving/eval code).
    """
    tok = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=cfg.trust_remote_code
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(cfg: QLoRAConfig) -> PreTrainedModel:
    """Load the frozen base model in 4-bit."""
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config(cfg),
        device_map=cfg.device_map,
        torch_dtype=_resolve_dtype(cfg.compute_dtype),
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=cfg.trust_remote_code,
    )
    # KV cache is incompatible with gradient checkpointing and unused in training.
    model.config.use_cache = False
    return model


def build_model(cfg: QLoRAConfig) -> PreTrainedModel:
    """Load base in 4-bit, prepare for k-bit training, attach LoRA adapters.

    Returns a PEFT model ready to hand to the trainer. Prints the trainable
    parameter count (should be well under 1% of total).
    """
    model = load_base_model(cfg)

    # Casts layernorms to fp32, enables input grads, and wires gradient
    # checkpointing so the 4-bit frozen base can be back-propagated through.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    model = get_peft_model(model, lora_config(cfg))

    if cfg.gradient_checkpointing:
        model.config.use_cache = False
        model.enable_input_require_grads()

    model.print_trainable_parameters()
    return model


# --------------------------------------------------------------------------- #
# Sanity check: python -m text2sql.train.qlora [model_name]
# Loads the model and reports trainable params + GPU memory footprint, so you can
# confirm it FITS before launching a real training run.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    cfg = QLoRAConfig()
    if len(sys.argv) > 1:
        cfg.model_name = sys.argv[1]

    print(f"Loading {cfg.model_name} in 4-bit ...")
    model = build_model(cfg)
    tok = load_tokenizer(cfg)

    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory after load: {mem_gb:.2f} GB")
    print("Tokenizer pad token:", tok.pad_token, "| padding side:", tok.padding_side)