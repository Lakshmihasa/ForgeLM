#!/usr/bin/env python
"""Launch the inference API.

  python scripts/serve.py --config configs/eval.yaml \
      --adapter outputs/llama3-8b-qlora-r16 --port 8000

Reads model/data settings, exports the env vars serve/app.py consumes at startup,
then runs uvicorn. serialize_kwargs is passed through and MUST match what the
model was trained and evaluated with.
"""

from __future__ import annotations

import argparse
import json
import os

from text2sql.common.config import merge_configs
from text2sql.common.logging import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", help="YAML with qlora/data sections")
    ap.add_argument("--base-model", help="override base model id")
    ap.add_argument("--tables", help="override tables.json path")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (optional)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    setup_logging(json_format=True)

    cfg = merge_configs(args.config, overrides=args.set) if args.config else {}
    qlora = cfg.get("qlora", {})
    data = cfg.get("data", {})

    base_model = args.base_model or qlora.get("model_name")
    tables = args.tables or data.get("tables")
    if not base_model or not tables:
        raise SystemExit("need --base-model/--tables (or a config providing them)")

    # Hand configuration to serve/app.py via the environment it reads at lifespan.
    os.environ["BASE_MODEL"] = base_model
    os.environ["TABLES_PATH"] = tables
    if args.adapter:
        os.environ["ADAPTER_DIR"] = args.adapter
    os.environ["SERIALIZE_KWARGS"] = json.dumps(data.get("serialize_kwargs", {}))

    import uvicorn

    uvicorn.run("text2sql.serve.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()