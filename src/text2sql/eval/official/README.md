# Vendored official Spider evaluator

`execution.py` and `difficulty.py` prefer the **official** Spider / test-suite
evaluator over their built-in fallbacks — because result-set comparison
(ORDER BY, SELECT * column permutations, value plugging) and hardness scoring
have real subtleties that are easy to get wrong. This directory is where that
upstream code lives.

**It is not committed by this repo.** These are third-party files with their own
licenses; you vendor them yourself and keep their upstream license headers.

## What to drop in

Place the upstream files under `official/_vendor/` (add an `__init__.py` there):

| file | from | used by |
|---|---|---|
| `exec_eval.py` | github.com/taoyds/test-suite-sql-eval | execution (test-suite EX) |
| `evaluation.py` | github.com/taoyds/spider (or test-suite-sql-eval) | hardness |
| `process_sql.py` | github.com/taoyds/spider | hardness (SQL parsing) |

```
official/
├── README.md              # this file
├── __init__.py            # is_available()
├── exec_eval.py           # ADAPTER (ours) -> eval_exec_match(db_path, predicted, gold)
├── hardness.py            # ADAPTER (ours) -> official difficulty buckets (optional)
└── _vendor/
    ├── __init__.py        # you add this (empty file)
    ├── exec_eval.py       # upstream
    ├── evaluation.py      # upstream
    └── process_sql.py     # upstream
```

## How it wires up

- **Execution.** `execution.py` does `from .official.exec_eval import eval_exec_match`.
  Our `exec_eval.py` adapter imports the upstream matcher from `_vendor/` and
  bridges its signature. If `_vendor/` is empty, the import fails and
  `execution.py` transparently falls back to its local comparison — nothing
  breaks, you just get plain single-DB accuracy with the caveat noted there.

- **Hardness.** `difficulty.py` uses a string heuristic by default and accepts an
  injected `hardness_fn`. To use the official buckets, see `hardness.py`:
  precompute a `{gold_sql: bucket}` map once and pass `make_hardness_fn(map)`.

## Licensing / attribution

The upstream files retain their original licenses (Spider is Apache-2.0; check
the test-suite repo for its terms). Keep the upstream headers intact and add an
entry to your repo's `NOTICE` / third-party licenses. Do not relicense them.

## Version note

Upstream function signatures have drifted across versions (keyword names,
argument order). Both adapters call defensively and document the single line to
adjust if your vendored copy differs. Validate against a handful of known
examples after vendoring — if your **base** model's EX looks wildly off, the
adapter bridge is the first place to check.