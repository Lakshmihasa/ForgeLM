"""One logger for the whole repo.

All modules log under the `text2sql` namespace (serve/observability already emits
`text2sql.serve`). `setup_logging` configures that single parent logger once;
child loggers inherit it. A JSON formatter is available so logs are
machine-parseable in a deployment, and it forwards any `extra=` fields on the log
call — so structured events (latency, tokens, db_id) survive as real keys rather
than being mashed into the message string.
"""

from __future__ import annotations

import json
import logging
import sys

__all__ = ["setup_logging", "get_logger", "JsonFormatter"]

_ROOT_NAME = "text2sql"

# LogRecord attributes that are NOT user-supplied extras.
_STD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line, including any `extra=` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(
    level: int | str = "INFO",
    *,
    json_format: bool = False,
    stream=sys.stderr,
) -> logging.Logger:
    """Configure the `text2sql` logger once and return it. Idempotent."""
    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)
    logger.propagate = False  # don't double-log through the root logger

    # Replace handlers so repeated calls don't stack duplicates.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    handler = logging.StreamHandler(stream)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger, e.g. get_logger('train') -> 'text2sql.train'."""
    if name == _ROOT_NAME or name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")