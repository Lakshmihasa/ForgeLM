#!/usr/bin/env python
"""LoRA-rank ablation.

Trains and evaluates at each rank via the existing scripts (so every run is a
normal, reproducible invocation), then collects dev EX into one table + plot.

  python scripts/sweep_rank.py --train-config configs/train/qlora_r16.yaml \
      --eval-config configs/eval.yaml --ranks 8 16 32 64
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--out", default="results/ablations")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for r in args.ranks:
        run_dir = f"outputs/llama3-8b-qlora-r{r}"
        # train (alpha = 2r keeps scaling constant across the sweep)
        run([
            sys.executable, "scripts/train.py", "--config", args.train_config,
            "--set", f"qlora.r={r}", f"qlora.lora_alpha={2*r}",
            f"train.output_dir={run_dir}", f"train.run_name=r{r}",
        ])
        # evaluate
        res_dir = out / f"r{r}"
        run([
            sys.executable, "scripts/evaluate.py", "--config", args.eval_config,
            "--adapter", f"{run_dir}/adapter", "--out", str(res_dir),
        ])
        results = json.loads((res_dir / "results.json").read_text())
        ex = results["conditions"].get("finetuned")
        rows.append({"rank": r, "dev_ex": ex})
        print(f"rank {r}: dev EX = {ex}")

    (out / "rank_sweep.json").write_text(json.dumps(rows, indent=2))
    print("\nrank  dev_ex")
    for row in rows:
        print(f"{row['rank']:>4}  {row['dev_ex']}")

    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot([r["rank"] for r in rows], [r["dev_ex"] for r in rows], "o-")
        plt.xlabel("LoRA rank")
        plt.ylabel("dev execution accuracy")
        plt.title("LoRA rank ablation")
        plt.grid(True, alpha=0.3)
        fig = Path("results/figures")
        fig.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig / "rank_vs_ex.png", dpi=150, bbox_inches="tight")
        print(f"plot -> {fig / 'rank_vs_ex.png'}")
    except ImportError:
        print("(install matplotlib for the rank-vs-EX plot)")


if __name__ == "__main__":
    main()