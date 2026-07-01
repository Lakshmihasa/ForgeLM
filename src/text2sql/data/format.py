"""Normalize raw Spider examples into clean, minimal jsonl records.

Raw Spider files (`train_spider.json`, `train_others.json`, `dev.json`) carry a
lot of fields we don't need (token lists, the parsed `sql` tree). This module
distills each example down to what training and evaluation actually consume:

    {"id": ..., "db_id": ..., "question": ..., "query": <gold SQL>}

and does the unglamorous data hygiene that most portfolio projects skip:

  * dedup      — drop exact (db_id, question, query) repeats
  * filtering  — drop malformed rows and rows whose db_id has no schema
  * leakage    — a helper to assert no (question, query) appears in two splits

Difficulty (easy/medium/hard/extra) is intentionally NOT computed here. It's a
function of the parsed SQL + schema and is produced at eval time by the official
Spider evaluator in `eval/difficulty.py`, so records stay evaluator-independent
and lean. Keeping that seam clean means format.py has no dependency on the
scoring code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .schema import SchemaStore

__all__ = [
    "Example",
    "normalize_example",
    "load_raw",
    "build_examples",
    "write_jsonl",
    "read_jsonl",
    "find_leakage",
    "format_split",
]

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Example:
    """One normalized text-to-SQL example. `query` is the gold SQL."""

    id: str
    db_id: str
    question: str
    query: str


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    """Strip and collapse internal whitespace to single spaces."""
    return _WS.sub(" ", (text or "").strip())


def _make_id(db_id: str, question: str, query: str) -> str:
    """Deterministic content id — stable across runs, so results join back and
    duplicates collapse to the same id."""
    payload = f"{db_id}\x1f{question}\x1f{query}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def normalize_example(raw: dict) -> Example | None:
    """Distill one raw Spider row into an Example, or None if unusable.

    Spider stores the gold SQL under the `query` key and the natural-language
    question under `question`.
    """
    db_id = (raw.get("db_id") or "").strip()
    question = _clean(raw.get("question", ""))
    query = _clean(raw.get("query", ""))
    if not db_id or not question or not query:
        return None
    return Example(
        id=_make_id(db_id, question, query),
        db_id=db_id,
        question=question,
        query=query,
    )


# --------------------------------------------------------------------------- #
# Loading / building
# --------------------------------------------------------------------------- #
def _as_list(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def load_raw(paths: str | Path | Iterable[str | Path]) -> list[dict]:
    """Load and concatenate one or more raw Spider JSON files."""
    out: list[dict] = []
    for p in _as_list(paths):
        with Path(p).open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        out.extend(data)
    return out


def build_examples(
    raw: Iterable[dict],
    schema_store: SchemaStore | None = None,
    *,
    dedup: bool = True,
    drop_unknown_db: bool = True,
) -> tuple[list[Example], dict[str, int]]:
    """Normalize + filter + dedup raw rows.

    Returns (examples, stats). Every example is guaranteed to have a non-empty
    question and query, and — when a schema_store is given and drop_unknown_db is
    on — a db_id that can actually be serialized and executed against.
    """
    examples: list[Example] = []
    seen: set[str] = set()
    stats: Counter[str] = Counter()

    for raw_ex in raw:
        stats["total"] += 1
        ex = normalize_example(raw_ex)
        if ex is None:
            stats["malformed"] += 1
            continue
        if (
            drop_unknown_db
            and schema_store is not None
            and ex.db_id not in schema_store
        ):
            stats["unknown_db"] += 1
            continue
        if dedup and ex.id in seen:
            stats["duplicate"] += 1
            continue
        seen.add(ex.id)
        examples.append(ex)
        stats["kept"] += 1

    return examples, dict(stats)


# --------------------------------------------------------------------------- #
# Leakage check (defensive; splits should already be database-disjoint)
# --------------------------------------------------------------------------- #
def find_leakage(a: Iterable[Example], b: Iterable[Example]) -> list[tuple[str, str]]:
    """Return (question, query) pairs present in BOTH splits.

    With a proper database-level split this must be empty. Run it as an assertion
    in the data pipeline so a leak fails loudly instead of silently inflating
    your eval number.
    """
    keys_b = {(e.question, e.query) for e in b}
    overlap = {(e.question, e.query) for e in a if (e.question, e.query) in keys_b}
    return sorted(overlap)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def write_jsonl(examples: Iterable[Example], path: str | Path) -> int:
    """Write examples as jsonl; returns the count written."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[Example]:
    """Read a jsonl file back into Examples."""
    out: list[Example] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(Example(**json.loads(line)))
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def format_split(
    raw_paths: str | Path | Iterable[str | Path],
    out_path: str | Path,
    schema_store: SchemaStore | None = None,
    *,
    dedup: bool = True,
    drop_unknown_db: bool = True,
) -> dict[str, int | str]:
    """Load raw -> normalize/filter/dedup -> write jsonl. Returns stats."""
    raw = load_raw(raw_paths)
    examples, stats = build_examples(
        raw, schema_store, dedup=dedup, drop_unknown_db=drop_unknown_db
    )
    written = write_jsonl(examples, out_path)
    return {**stats, "written": written, "out_path": str(out_path)}


# --------------------------------------------------------------------------- #
# CLI
#   python -m text2sql.data.format \
#       --tables data/raw/spider/tables.json \
#       --raw data/raw/spider/train_spider.json data/raw/spider/train_others.json \
#       --out data/processed/train_full.jsonl
# splits.py then carves train_full.jsonl into train/val by database.
# --------------------------------------------------------------------------- #
def _main() -> None:
    ap = argparse.ArgumentParser(description="Normalize raw Spider JSON to jsonl.")
    ap.add_argument("--raw", nargs="+", required=True, help="raw Spider json file(s)")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--tables", help="tables.json (enables unknown-db filtering)")
    ap.add_argument("--keep-dupes", action="store_true", help="disable dedup")
    ap.add_argument(
        "--keep-unknown-db",
        action="store_true",
        help="keep rows whose db_id has no schema",
    )
    args = ap.parse_args()

    store = SchemaStore(args.tables) if args.tables else None
    stats = format_split(
        args.raw,
        args.out,
        store,
        dedup=not args.keep_dupes,
        drop_unknown_db=not args.keep_unknown_db,
    )

    print("=== format stats ===")
    for key in ("total", "kept", "written", "duplicate", "malformed", "unknown_db"):
        if key in stats:
            print(f"{key:>12}: {stats[key]}")
    print(f"{'out_path':>12}: {stats['out_path']}")


if __name__ == "__main__":
    _main()