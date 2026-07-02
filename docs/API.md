# Serving API

Base URL: `http://localhost:8000`. Start with `scripts/serve.py` or `docker compose up serve`.

## POST /generate

Generate SQL for a question against a known database.

Request:
```json
{
  "question": "How many singers do we have?",
  "db_id": "concert_singer",
  "max_new_tokens": 256,
  "include_raw": false
}
```
`question`, `db_id` required. `max_new_tokens` optional (1–1024). `include_raw` returns the model's pre-extraction output.

Response `200`:
```json
{
  "sql": "SELECT count(*) FROM singer",
  "db_id": "concert_singer",
  "request_id": "b1e2...",
  "latency_ms": 812.4,
  "prompt_tokens": 240,
  "completion_tokens": 9,
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "raw_output": null
}
```

Errors: `400` unknown `db_id`; `500` generation failure. Both return `{"error": "...", "request_id": null}`.

```bash
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"question":"how many singers are there?","db_id":"concert_singer"}'
```

## GET /health

```json
{ "status": "ok", "model": "...", "adapter": "outputs/.../adapter", "device": "cuda:0", "uptime_s": 42.1 }
```

## GET /metrics

Rolling, in-process metrics over recent requests:
```json
{
  "total_requests": 128,
  "error_rate": 0.0,
  "empty_sql_rate": 0.02,
  "latency_ms_p50": 640.0,
  "latency_ms_p95": 1180.0,
  "latency_ms_p99": 1500.0,
  "avg_prompt_tokens": 232.5,
  "avg_completion_tokens": 12.3
}
```
`empty_sql_rate` is a cheap output-drift signal — a rising value suggests inputs are drifting from the training distribution.

## Notes

- Greedy decoding (deterministic), matching eval.
- One generation at a time (GPU access is lock-serialized); a `def` endpoint runs it in a threadpool so the event loop stays free. High throughput is a v2 concern (vLLM continuous batching).
- `serialize_kwargs` (env `SERIALIZE_KWARGS`) must match training/eval, or output diverges from reported numbers.