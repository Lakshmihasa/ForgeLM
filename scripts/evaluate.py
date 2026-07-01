#!/usr/bin/env python
"""Run the baseline ladder and report the headline result.

  python scripts/evaluate.py --config configs/eval.yaml \
      --adapter outputs/llama3-8b-qlora-r16 --few-shot-k 5

Conditions (all through the identical harness + execution scorer):
  base_zero_shot | base_few_shot (the fair baseline) | finetuned

Config shape:
  qlora: { model_name, ... }            # base model + quant settings
  data:
    tables: data/raw/spider/tables.json
    test_jsonl: data/processed/test.jsonl
    train_jsonl: data/processed/train.jsonl
    serialize_kwargs: {}                # MUST match training/serve
    db_root: data/raw/spider/database
    test_suite_root: null               # set for test-suite EX
  eval: { batch_size: 8, max_new_tokens: 256, few_shot_k: 5, seed: 13 }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from peft import PeftModel

from text2sql.common.config import merge_configs
from text2sql.common.logging import get_logger, setup_logging
from text2sql.common.seed import set_seed
from text2sql.data.format import read_jsonl
from text2sql.data.prompt import PromptBuilder
from text2sql.data.schema import SchemaStore
from text2sql.eval.baselines import LadderResult, evaluate_condition, select_few_shot
from text2sql.eval.harness import HarnessConfig, save_predictions
from text2sql.train.qlora import QLoRAConfig, load_base_model, load_tokenizer

log = get_logger("evaluate")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True, help="trained LoRA adapter dir")
    ap.add_argument("--few-shot-k", type=int, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    setup_logging()
    cfg = merge_configs(args.config, overrides=args.set)
    data, ev = cfg["data"], cfg.get("eval", {})
    set_seed(ev.get("seed", 13))

    k = args.few_shot_k if args.few_shot_k is not None else ev.get("few_shot_k", 5)
    db_root = data["db_root"]
    test_suite_root = data.get("test_suite_root")
    serialize_kwargs = data.get("serialize_kwargs", {})
    hcfg = HarnessConfig(
        batch_size=ev.get("batch_size", 8),
        max_new_tokens=ev.get("max_new_tokens", 256),
    )

    # Data
    test = read_jsonl(data["test_jsonl"])
    train = read_jsonl(data["train_jsonl"])
    few_shot = select_few_shot(train, k=k, seed=ev.get("seed", 13))

    # Model: load base once, build the shared prompt builder.
    qlora_cfg = QLoRAConfig.from_dict(cfg.get("qlora", {}))
    base = load_base_model(qlora_cfg)
    tokenizer = load_tokenizer(qlora_cfg)
    builder = PromptBuilder(SchemaStore(data["tables"]), tokenizer, serialize_kwargs=serialize_kwargs)

    def run(name, model, fs):
        log.info("evaluating condition: %s", name)
        return evaluate_condition(
            name, model, tokenizer, test, builder,
            db_root=db_root, test_suite_root=test_suite_root, few_shot=fs, config=hcfg,
        )

    ladder = LadderResult()
    ladder.add(run("base_zero_shot", base, []))
    ladder.add(run("base_few_shot", base, few_shot))

    # Fine-tuned = base + adapter. Wrap after the base conditions have run.
    finetuned = PeftModel.from_pretrained(base, args.adapter)
    ladder.add(run("finetuned", finetuned, []))

    # Report
    summary = ladder.summary(baseline="base_few_shot", candidate="finetuned")
    print("\n" + summary + "\n")

    # Persist predictions + a machine-readable results file.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in ladder.conditions:
        save_predictions(c.predictions, out / f"pred_{c.name}.jsonl")
    ci = ladder.gap_ci("base_few_shot", "finetuned")
    results = {
        "conditions": {c.name: c.accuracy for c in ladder.conditions},
        "gap_finetuned_minus_base_few_shot": {
            "estimate": ci.estimate, "low": ci.low, "high": ci.high, "n": ci.n,
        },
        "few_shot_k": k,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "summary.txt").write_text(summary)
    print(f"predictions + results.json + summary.txt -> {out}")


if __name__ == "__main__":
    main()