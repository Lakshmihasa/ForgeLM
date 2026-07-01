#!/usr/bin/env python
"""Fine-tune with QLoRA.

  python scripts/train.py --config configs/train/qlora_r16.yaml \
      --set train.run_name=r16 qlora.r=16

Expected config shape:
  qlora: { ... QLoRAConfig fields ... }
  train: { ... TrainConfig fields ... }
  data:
    tables: data/raw/spider/tables.json
    train_jsonl: data/processed/train.jsonl
    val_jsonl: data/processed/val.jsonl
    serialize_kwargs: {}          # MUST match eval/serve
    db_root: data/raw/spider/database         # only needed for --ex-eval
    test_suite_root: null

Pass --ex-eval N to run execution accuracy on N val examples at each eval step
(logs val/execution_accuracy + a few sample generations). Needs data.db_root.
"""

from __future__ import annotations

import argparse

from text2sql.common.config import load_config, merge_configs, save_yaml
from text2sql.common.logging import get_logger, setup_logging
from text2sql.common.seed import set_seed
from text2sql.data.format import read_jsonl
from text2sql.data.prompt import PromptBuilder
from text2sql.data.schema import SchemaStore
from text2sql.eval.execution import score_predictions
from text2sql.eval.harness import HarnessConfig, generate_predictions
from text2sql.train.callbacks import ExecutionAccuracyCallback, SampleGenerationsCallback
from text2sql.train.qlora import QLoRAConfig
from text2sql.train.trainer import TrainConfig, train

log = get_logger("train")


def _build_callbacks(cfg: dict, subset_n: int):
    """Wire periodic EX-accuracy + sample-generation callbacks on a val subset."""
    data = cfg["data"]
    store = SchemaStore(data["tables"])
    serialize_kwargs = data.get("serialize_kwargs", {})
    val_subset = read_jsonl(data["val_jsonl"])[:subset_n]
    db_root = data["db_root"]
    test_suite_root = data.get("test_suite_root")
    hcfg = HarnessConfig(batch_size=8, max_new_tokens=256)

    def eval_fn(model, tokenizer):
        builder = PromptBuilder(store, tokenizer, serialize_kwargs=serialize_kwargs)
        preds = generate_predictions(model, tokenizer, val_subset, builder, config=hcfg)
        report = score_predictions(preds, db_root, test_suite_root)
        return {"execution_accuracy": report.execution_accuracy}

    def sample_fn(model, tokenizer, n):
        builder = PromptBuilder(store, tokenizer, serialize_kwargs=serialize_kwargs)
        preds = generate_predictions(model, tokenizer, val_subset[:n], builder, config=hcfg)
        return [
            {"question": p.question, "db_id": p.db_id, "predicted": p.predicted_sql, "gold": p.gold}
            for p in preds
        ]

    return [ExecutionAccuracyCallback(eval_fn), SampleGenerationsCallback(sample_fn)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=None, help="dotlist overrides")
    ap.add_argument("--ex-eval", type=int, default=0, help="val subset size for EX eval (0=off)")
    args = ap.parse_args()

    setup_logging()
    cfg = merge_configs(args.config, overrides=args.set)

    qlora_cfg = QLoRAConfig.from_dict(cfg.get("qlora", {}))
    train_cfg = TrainConfig.from_dict(cfg.get("train", {}))
    data = cfg["data"]
    set_seed(train_cfg.seed)

    callbacks = _build_callbacks(cfg, args.ex_eval) if args.ex_eval > 0 else None

    log.info("fine-tuning %s (r=%s) -> %s", qlora_cfg.model_name, qlora_cfg.r, train_cfg.output_dir)
    train(
        qlora_cfg,
        train_cfg,
        tables_path=data["tables"],
        train_jsonl=data["train_jsonl"],
        val_jsonl=data.get("val_jsonl"),
        serialize_kwargs=data.get("serialize_kwargs", {}),
        callbacks=callbacks,
    )

    # Snapshot the fully-resolved config next to the adapter for reproducibility.
    save_yaml(cfg, f"{train_cfg.output_dir}/resolved_config.yaml")
    print(f"adapter + config saved under {train_cfg.output_dir}")


if __name__ == "__main__":
    main()