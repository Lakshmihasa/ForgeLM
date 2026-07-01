"""Observability: structured request logs + in-memory metrics.

"Full observability" on a resume has to mean something you can point at. Here it's
two concrete things, kept dependency-light (stdlib only):

  * a structured JSON log line per request (id, latency, tokens, db, outcome)
  * a rolling metrics snapshot: latency p50/p95/p99, error rate, average token
    counts, and the EMPTY-SQL RATE

The empty-SQL rate is a deliberate, cheap drift signal: if the fraction of
requests where the model produces no usable query starts climbing, the input
distribution has likely shifted away from what the model was trained on
(unfamiliar schemas, out-of-domain questions). It's the honest MVP version of
"drift detection" — a real metric you can watch, not a dashboard prop. A
production build would export these to Prometheus; this keeps them in-process.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock

__all__ = ["RequestRecord", "Metrics"]

logger = logging.getLogger("text2sql.serve")


@dataclass
class RequestRecord:
    request_id: str
    db_id: str
    start: float
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    empty: bool = False
    success: bool = False
    error: str | None = None


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


class Metrics:
    """Thread-safe rolling metrics over the most recent `window` requests."""

    def __init__(self, window: int = 1000):
        self._lock = Lock()
        self._latencies: deque[float] = deque(maxlen=window)
        self.total = 0
        self.errors = 0
        self.empty = 0
        self.prompt_tokens_sum = 0
        self.completion_tokens_sum = 0

    def _record(self, rec: RequestRecord) -> None:
        with self._lock:
            self.total += 1
            if not rec.success:
                self.errors += 1
            if rec.empty:
                self.empty += 1
            self._latencies.append(rec.latency_ms)
            self.prompt_tokens_sum += rec.prompt_tokens
            self.completion_tokens_sum += rec.completion_tokens

    @contextmanager
    def track(self, request_id: str, db_id: str):
        """Time a request, log it, and fold it into the metrics — even on error.

        Yields a RequestRecord the caller fills in (tokens, empty). Success is set
        automatically if the block completes without raising; exceptions are
        recorded and re-raised.
        """
        rec = RequestRecord(request_id=request_id, db_id=db_id, start=time.perf_counter())
        try:
            yield rec
            rec.success = True
        except Exception as e:
            rec.error = str(e)
            raise
        finally:
            rec.latency_ms = (time.perf_counter() - rec.start) * 1000.0
            self._record(rec)
            self._log(rec)

    def _log(self, rec: RequestRecord) -> None:
        logger.info(
            json.dumps(
                {
                    "event": "generate",
                    "request_id": rec.request_id,
                    "db_id": rec.db_id,
                    "latency_ms": round(rec.latency_ms, 2),
                    "prompt_tokens": rec.prompt_tokens,
                    "completion_tokens": rec.completion_tokens,
                    "empty": rec.empty,
                    "success": rec.success,
                    "error": rec.error,
                }
            )
        )

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self._latencies)
            total = self.total or 1  # avoid div by zero in rates
            return {
                "total_requests": self.total,
                "error_rate": self.errors / total,
                "empty_sql_rate": self.empty / total,
                "latency_ms_p50": round(_percentile(lat, 0.50), 2),
                "latency_ms_p95": round(_percentile(lat, 0.95), 2),
                "latency_ms_p99": round(_percentile(lat, 0.99), 2),
                "avg_prompt_tokens": self.prompt_tokens_sum / total,
                "avg_completion_tokens": self.completion_tokens_sum / total,
            }