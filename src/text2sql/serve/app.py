"""FastAPI service for the fine-tuned text-to-SQL model.

Endpoints:
  GET  /health   liveness + which model/adapter/device is loaded
  POST /generate question + db_id -> SQL, with latency + token counts
  GET  /metrics  rolling latency percentiles, error rate, empty-SQL rate

The model loads ONCE at startup (lifespan) and lives in app state. /generate is a
sync def so Starlette runs it in a threadpool — that keeps the event loop free
while the (blocking, lock-serialized) GPU generation runs.

Configuration is via environment variables:
  BASE_MODEL        (required)  e.g. meta-llama/Meta-Llama-3-8B-Instruct
  TABLES_PATH       (required)  path to Spider tables.json
  ADAPTER_DIR       (optional)  trained LoRA adapter; omit to serve the base
  SERIALIZE_KWARGS  (optional)  JSON; MUST match training/eval serialization
  MAX_NEW_TOKENS    (optional)  default 256
  LOAD_IN_4BIT      (optional)  "1"/"0", default "1"
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..eval.extract import looks_empty
from .inference import InferenceConfig, Text2SQLModel
from .observability import Metrics
from .schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
)


def _load_config() -> InferenceConfig:
    base_model = os.environ.get("BASE_MODEL")
    tables_path = os.environ.get("TABLES_PATH")
    if not base_model or not tables_path:
        raise RuntimeError("BASE_MODEL and TABLES_PATH must be set")
    return InferenceConfig(
        base_model=base_model,
        tables_path=tables_path,
        adapter_dir=os.environ.get("ADAPTER_DIR") or None,
        serialize_kwargs=json.loads(os.environ.get("SERIALIZE_KWARGS", "{}")),
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "256")),
        load_in_4bit=os.environ.get("LOAD_IN_4BIT", "1") == "1",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = Text2SQLModel(_load_config())
    app.state.metrics = Metrics()
    app.state.start_time = time.time()
    yield
    # nothing to tear down; process exit frees the GPU


app = FastAPI(title="text2sql-qlora", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model: Text2SQLModel = request.app.state.model
    info = model.info()
    return HealthResponse(
        status="ok",
        model=info["model"],
        adapter=info["adapter"],
        device=info["device"],
        uptime_s=time.time() - request.app.state.start_time,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    model: Text2SQLModel = request.app.state.model
    metrics: Metrics = request.app.state.metrics
    request_id = str(uuid.uuid4())

    if not model.has_db(req.db_id):
        raise HTTPException(status_code=400, detail=f"unknown db_id: {req.db_id!r}")

    try:
        with metrics.track(request_id, req.db_id) as rec:
            result = model.generate(req.question, req.db_id, req.max_new_tokens)
            rec.prompt_tokens = result.prompt_tokens
            rec.completion_tokens = result.completion_tokens
            rec.empty = looks_empty(result.sql)
    except Exception as e:  # generation failure -> 500, already recorded
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(
        sql=result.sql,
        db_id=req.db_id,
        request_id=request_id,
        latency_ms=round(rec.latency_ms, 2),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        model=model.info()["model"],
        raw_output=result.raw_output if req.include_raw else None,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics(request: Request) -> MetricsResponse:
    return MetricsResponse(**request.app.state.metrics.snapshot())


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=str(exc.detail)).model_dump(),
    )