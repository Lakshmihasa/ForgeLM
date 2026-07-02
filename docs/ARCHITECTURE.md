# Architecture

## One principle

**The prompt a model is trained on is byte-identical to the prompt it is evaluated and served on.** Everything below is organized to make that true and to keep evaluation trustworthy — not to make training convenient.

## Data flow

```mermaid
flowchart TD
    raw[raw Spider json] --> fmt[data.format<br/>normalize, dedup, filter]
    fmt --> full[train_full.jsonl]
    fmt --> test[test.jsonl]
    full --> split[data.splits<br/>DB-level train/val + manifest]
    split --> train[train.jsonl]
    split --> val[val.jsonl]

    subgraph prompt_path[shared prompt path]
      schema[data.schema<br/>serialize] --> prompt[data.prompt<br/>chat template + masking]
    end

    train --> trainer[train.trainer<br/>QLoRA SFT]
    prompt --> trainer
    trainer --> adapter[LoRA adapter]

    test --> harness[eval.harness<br/>render→generate→extract]
    prompt --> harness
    adapter --> harness
    harness --> exec[eval.execution<br/>test-suite EX]
    exec --> report[per-example results]
    report --> diff[eval.difficulty]
    report --> stats[eval.stats<br/>bootstrap CI]
    report --> ladder[eval.baselines<br/>ladder + gap CI]

    adapter --> serve[serve.app<br/>FastAPI]
    prompt --> serve
```

## Module responsibilities

| Package | Owns |
|---|---|
| `data` | schema serialization (the contract), prompt/chat-template + loss masking, DB-level splits, jsonl normalization |
| `train` | QLoRA config (4-bit NF4 + LoRA), SFT trainer + masked collator, W&B / EX-eval callbacks |
| `eval` | the harness (parity), extraction rule, test-suite execution accuracy, difficulty buckets, bootstrap CI, baseline ladder |
| `serve` | FastAPI app, model+adapter inference, observability, request/response schemas |
| `common` | config loading, logging, seeding |

## Parity, concretely

- `data.prompt.PromptBuilder.build_training_example` builds its prompt via the same `render_inference_prompt` the harness and server use — one code path.
- `tests/test_harness_parity.py` asserts the training prompt tokens equal the inference prompt tokens and that the prompt is masked (`-100`) while the completion is supervised.
- `serialize_kwargs` is the single dict that must match across train/eval/serve; it lives in the configs' `data:` block.

## Scoring seams (dependency injection)

The harness produces predictions; scoring is injected (`eval.execution.make_scorer`). The training callbacks take an `eval_fn`/`sample_fn`. This keeps the harness independent of the evaluator (testable with a stub) and lets the official Spider evaluator be vendored in without touching core code (`eval/official/`).