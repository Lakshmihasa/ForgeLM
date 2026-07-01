"""Database-level train/val split.

`format.py` produces `train_full.jsonl` (all Spider training examples) and
`test.jsonl` (Spider dev). This module carves `train_full` into `train` and
`val` — and the one rule that makes the whole eval honest is:

    split by DATABASE, never by row.

Val holds out ~15-20 entire databases. Because those schemas never appear in
training, checkpoint selection on val is tested under the *same* unseen-schema
condition as the final dev evaluation — so you don't pick a checkpoint that
memorized seen schemas but generalizes badly. A row-level split would leak
schemas from train into val and quietly flatter every checkpoint you pick.

The split is deterministic from a seed and the exact held-out database list is
written to a manifest, so a run is fully reproducible and "which databases were
in val?" is answerable from a committed file, not memory.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .format import Example, find_leakage, read_jsonl, write_jsonl

__all__ = ["SplitResult", "select_val_databases", "split_by_database", "write_splits"]


@dataclass(frozen=True)
class SplitResult:
    train: list[Example]
    val: list[Example]
    val_db_ids: list[str]
    stats: dict = field(default_factory=dict)


def select_val_databases(
    db_ids: Iterable[str],
    *,
    n_val_dbs: int | None = 20,
    val_fraction: float | None = None,
    db_sizes: dict[str, int] | None = None,
    seed: int = 13,
) -> list[str]:
    """Deterministically choose which databases become the val set.

    Sorting the db_ids before the seeded shuffle makes the choice independent of
    the input file's row order — same seed always yields the same held-out DBs.

    If `val_fraction` is given, databases are accumulated until they cover roughly
    that share of *examples* (needs `db_sizes`); otherwise a fixed `n_val_dbs`
    count is used.
    """
    rng = random.Random(seed)
    ordered = sorted(set(db_ids))  # deterministic base order
    rng.shuffle(ordered)

    if val_fraction is not None:
        if db_sizes is None:
            raise ValueError("val_fraction requires db_sizes")
        total = sum(db_sizes.values())
        target = total * val_fraction
        acc, chosen = 0, []
        for db in ordered:
            if acc >= target:
                break
            chosen.append(db)
            acc += db_sizes.get(db, 0)
        return sorted(chosen)

    n = n_val_dbs if n_val_dbs is not None else 20
    if n >= len(ordered):
        raise ValueError(
            f"n_val_dbs={n} >= total databases={len(ordered)}; nothing left to train on"
        )
    return sorted(ordered[:n])


def split_by_database(
    examples: list[Example],
    *,
    val_db_ids: Iterable[str] | None = None,
    n_val_dbs: int | None = 20,
    val_fraction: float | None = None,
    seed: int = 13,
) -> SplitResult:
    """Partition examples into train/val with disjoint database sets.

    Pass `val_db_ids` to reproduce an exact prior split (e.g. from a manifest);
    otherwise the val databases are selected deterministically from the seed.
    Raises if any (question, query) leaks across the two splits.
    """
    by_db: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_db[ex.db_id].append(ex)

    if val_db_ids is None:
        db_sizes = {db: len(v) for db, v in by_db.items()}
        val_db_ids = select_val_databases(
            by_db.keys(),
            n_val_dbs=n_val_dbs,
            val_fraction=val_fraction,
            db_sizes=db_sizes,
            seed=seed,
        )
    val_set = set(val_db_ids)

    train = [ex for ex in examples if ex.db_id not in val_set]
    val = [ex for ex in examples if ex.db_id in val_set]

    # Guards: database disjointness and no example leakage.
    train_dbs = {e.db_id for e in train}
    if train_dbs & val_set:
        raise AssertionError("database overlap between train and val")
    leak = find_leakage(train, val)
    if leak:
        raise ValueError(
            f"{len(leak)} (question, query) pairs leak across train/val; "
            "the split is not clean"
        )

    stats = {
        "train_examples": len(train),
        "val_examples": len(val),
        "train_dbs": len(train_dbs),
        "val_dbs": len(val_set),
        "val_example_fraction": round(len(val) / max(len(examples), 1), 4),
        "seed": seed,
    }
    return SplitResult(train=train, val=val, val_db_ids=sorted(val_set), stats=stats)


def write_splits(
    result: SplitResult,
    out_train: str | Path,
    out_val: str | Path,
    manifest_path: str | Path | None = None,
) -> dict:
    """Write train.jsonl, val.jsonl, and a manifest pinning the exact held-out
    databases + seed so the split is reproducible."""
    write_jsonl(result.train, out_train)
    write_jsonl(result.val, out_val)

    manifest = {
        "seed": result.stats.get("seed"),
        "val_db_ids": result.val_db_ids,
        **result.stats,
        "out_train": str(out_train),
        "out_val": str(out_val),
    }
    if manifest_path is not None:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(manifest_path).open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    return manifest


# --------------------------------------------------------------------------- #
# CLI
#   python -m text2sql.data.splits \
#       --in data/processed/train_full.jsonl \
#       --out-train data/processed/train.jsonl \
#       --out-val data/processed/val.jsonl \
#       --manifest data/processed/split_manifest.json \
#       --n-val-dbs 20 --seed 13
# Reproduce an exact split with:  --val-dbs db_a db_b ...
# --------------------------------------------------------------------------- #
def _main() -> None:
    ap = argparse.ArgumentParser(description="Database-level train/val split.")
    ap.add_argument("--in", dest="in_path", required=True, help="train_full.jsonl")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--n-val-dbs", type=int, default=20)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--val-dbs", nargs="+", default=None, help="reproduce an exact split"
    )
    args = ap.parse_args()

    examples = read_jsonl(args.in_path)
    result = split_by_database(
        examples,
        val_db_ids=args.val_dbs,
        n_val_dbs=args.n_val_dbs,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    manifest = write_splits(result, args.out_train, args.out_val, args.manifest)

    print("=== split stats ===")
    for k, v in result.stats.items():
        print(f"{k:>22}: {v}")
    print(f"{'val_dbs (first 5)':>22}: {result.val_db_ids[:5]} ...")
    if args.manifest:
        print(f"{'manifest':>22}: {manifest['out_train']!r} split pinned")


if __name__ == "__main__":
    _main()