# text2sql-qlora

**QLoRA fine-tune of Llama-3-8B for text-to-SQL, measured by execution accuracy against a *fair few-shot baseline* on database schemas the model never saw during training.**

The point of this project is not "I fine-tuned an LLM." It's **honest measurement**: every model — base zero-shot, base few-shot, and fine-tuned — runs through one frozen evaluation harness, scored with the official Spider evaluator via execution accuracy. The headline number is the delta over a base model that got a *fair* shot, not a strawman. 

---

## Results

> ⚠️ **Placeholder table — fill in after your eval run (Day 5). Do not publish invented numbers.**

Execution accuracy (test-suite EX) on the Spider **dev** set (databases unseen in training):

| Model | Prompt | Dev EX | Easy | Medium | Hard | Extra |
|---|---|---:|---:|---:|---:|---:|
| Llama-3-8B (base) | zero-shot | _XX.X_ | _XX.X_ | _XX.X_ | _XX.X_ | _XX.X_ |
| Llama-3-8B (base) | **few-shot (k=5)** | _XX.X_ | _XX.X_ | _XX.X_ | _XX.X_ | _XX.X_ |
| **Llama-3-8B + QLoRA (ours)** | zero-shot | **_XX.X_** | _XX.X_ | _XX.X_ | _XX.X_ | _XX.X_ |

**Headline:** fine-tuned model improves dev EX by **_+XX.X_ points over the few-shot base** (bootstrap 95% CI: _[XX.X, XX.X]_, n≈1,000). Gains are largest on the **hard / extra** buckets — see [`docs/writeup.md`](docs/writeup.md).

---

## Why the eval is trustworthy

This is the part most fine-tuning projects skip. Four guarantees:

- **Fair baseline.** The base model is evaluated *few-shot with the schema in the prompt*, so "the fine-tune wins" isn't an artifact of prompting the base model badly.
- **Official evaluator, test-suite EX.** Queries are scored by the vendored Spider evaluator against multiple seeded database instances, so a coincidental result-set match (e.g. two queries both returning empty) is caught.
- **Unseen schemas.** Spider's dev databases don't appear in training, and the validation split is held out **by database** — so the number measures generalization, not memorization.
- **Harness parity, enforced in CI.** `tests/test_harness_parity.py` proves base and fine-tuned models traverse byte-identical serialization → prompt → extraction → evaluator. Same plumbing, always.

Full protocol: [`docs/eval_spec.md`](docs/eval_spec.md).

---

## Quickstart

**Prerequisites:** Python 3.10+, a CUDA GPU with ≥16 GB VRAM (for 4-bit 8B), a Hugging Face token, and (optional) a Weights & Biases key.

```bash
git clone <your-repo-url> && cd text2sql-qlora
pip install -e .
cp .env.example .env        # add HF_TOKEN, WANDB_API_KEY

make data                   # download Spider + build train/val/test jsonl
make eval                   # run the baseline ladder (no GPU training needed)
make train                  # QLoRA fine-tune (configs/train/qlora_r16.yaml)
make eval MODEL=finetuned   # score the fine-tuned model vs the ladder
make serve                  # launch the FastAPI inference endpoint
```

Swap the base model (Mistral-7B ↔ Llama-3-8B) or the LoRA rank by pointing at a different file in `configs/` — no code changes.

---

## How it works

```
raw Spider/BIRD ──▶ schema serialization + prompt formatting ──▶ train/val/test jsonl (DB-level split)
                                                                      │
                                          ┌───────────────────────────┤
                                          ▼                           ▼
                                    QLoRA fine-tune            frozen eval harness
                                    (4-bit NF4 + LoRA)         (extract → execute → official EX)
                                          │                           │
                                          └────────────┬──────────────┘
                                                        ▼
                                          FastAPI endpoint + A/B UI (base vs fine-tuned)
```

- **Training:** base model loaded in 4-bit (NF4), only low-rank LoRA adapters trained; loss curves and periodic held-out EX logged to W&B.
- **Eval:** greedy decode (temp 0) for reproducibility; per-difficulty breakdown and a bootstrap CI on the gap.
- **Serving:** `POST /generate` loads base + adapter, returns SQL, and logs latency + token counts.

---

## Project structure

```
src/text2sql/data/    schema serialization, prompt/chat-template, DB-level splits
src/text2sql/train/   QLoRA config, SFT trainer, W&B callbacks
src/text2sql/eval/    the harness — extraction, execution accuracy, official evaluator, stats
src/text2sql/serve/   FastAPI app, inference, observability
configs/              every run is a versioned YAML (reproducibility)
results/              committed EX tables, ablation results, figures
docs/                 eval_spec.md, model_card.md, writeup.md
tests/                extraction, execution, and harness-parity tests
```

Full rationale in [`docs/project_structure.md`](docs/project_structure.md).

---

## Ablation

> Placeholder — fill after the rank sweep (Day 6).

LoRA rank sweep (`r ∈ {8, 16, 32, 64}`), held-out EX:

| Rank | Dev EX | Notes |
|---:|---:|---|
| 8 | _XX.X_ | _underfit?_ |
| 16 | _XX.X_ | _sweet spot?_ |
| 32 | _XX.X_ | |
| 64 | _XX.X_ | _diminishing / overfit hard bucket?_ |

Plot: `results/figures/rank_vs_ex.png`.

---

## Stack

PyTorch · Hugging Face Transformers · PEFT (LoRA) · bitsandbytes (4-bit QLoRA) · TRL · FastAPI · Weights & Biases · Docker · Streamlit (A/B UI)

---

## Limitations

- Reported on Spider **dev**; the hidden test set is leaderboard-gated.
- Single base model, single domain (text-to-SQL) — by design, this is a scoped MVP.
- No multi-GPU / distributed training in this version.

See [`docs/model_card.md`](docs/model_card.md) for data, metrics, and intended use.

---

## Roadmap (v2)

- BIRD as a second, harder held-out benchmark
- vLLM serving (continuous batching, adapter hot-swap)
- Multi-model / multi-adapter registry
- Arbitrary dataset upload + preprocessing
- Drift detection on production traffic

---

## Acknowledgements

Built on the [Spider](https://yale-lily.github.io/spider) and [BIRD](https://bird-bench.github.io/) text-to-SQL benchmarks. Fine-tuning via Hugging Face PEFT and TRL.
