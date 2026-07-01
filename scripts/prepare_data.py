#!/usr/bin/env python
"""Build the processed dataset from raw Spider.

raw Spider  ->  train_full.jsonl + test.jsonl  ->  train.jsonl + val.jsonl (+ manifest)

  python scripts/prepare_data.py --spider-dir data/raw/spider --out-dir data/processed

Steps: normalize + dedup + filter (format), database-level train/val split (splits),
and a hard leakage assertion between train and test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from text2sql.common.logging import get_logger, setup_logging
from text2sql.data.format import find_leakage, format_split, read_jsonl
from text2sql.data.schema import SchemaStore
from text2sql.data.splits import split_by_database, write_splits

log = get_logger("prepare_data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spider-dir", required=True, help="root of the Spider release")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--n-val-dbs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    setup_logging()
    spider = Path(args.spider_dir)
    out = Path(args.out_dir)
    tables = spider / "tables.json"
    store = SchemaStore(tables)

    # Normalize raw -> jsonl (train files may be split across two files).
    train_raw = [p for p in (spider / "train_spider.json", spider / "train_others.json") if p.exists()]
    s1 = format_split(train_raw, out / "train_full.jsonl", store)
    s2 = format_split(spider / "dev.json", out / "test.jsonl", store)
    log.info("normalized: train_full kept=%s, test kept=%s", s1["kept"], s2["kept"])

    # Database-level train/val split with a pinned manifest.
    result = split_by_database(
        read_jsonl(out / "train_full.jsonl"),
        n_val_dbs=args.n_val_dbs,
        seed=args.seed,
    )
    write_splits(
        result,
        out / "train.jsonl",
        out / "val.jsonl",
        out / "split_manifest.json",
    )
    log.info("split: %s", result.stats)

    # Hard leakage guard: no (question, query) may cross train <-> test.
    test = read_jsonl(out / "test.jsonl")
    leak = find_leakage(result.train, test)
    if leak:
        raise SystemExit(f"ABORT: {len(leak)} examples leak between train and test")
    log.info("leakage check passed (train vs test)")

    print(
        f"train={len(result.train)}  val={len(result.val)}  test={len(test)}  "
        f"val_dbs={result.stats['val_dbs']}  ->  {out}"
    )


if __name__ == "__main__":
    main()