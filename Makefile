# text2sql-qlora — common workflows.
# Override ADAPTER=... for eval/serve; SPIDER_DIR / OUT for data.

ADAPTER    ?= outputs/llama3-8b-qlora-r16/adapter
SPIDER_DIR ?= data/raw/spider
OUT        ?= data/processed
TRAIN_CFG  ?= configs/train/qlora_r16.yaml
EVAL_CFG   ?= configs/eval.yaml

.PHONY: install download data train eval sweep serve ui test lint fmt vendor-official clean

install:
	pip install -e ".[dev]"

download:
	python scripts/download_data.py --out $(SPIDER_DIR)

data:
	python scripts/prepare_data.py --spider-dir $(SPIDER_DIR) --out-dir $(OUT)

train:
	python scripts/train.py --config $(TRAIN_CFG)

eval:
	python scripts/evaluate.py --config $(EVAL_CFG) --adapter $(ADAPTER)

sweep:
	python scripts/sweep_rank.py --train-config $(TRAIN_CFG) --eval-config $(EVAL_CFG)

serve:
	python scripts/serve.py --config $(EVAL_CFG) --adapter $(ADAPTER)

ui:
	streamlit run ui/app.py

test:
	pytest -q

lint:
	ruff check src tests scripts ui

fmt:
	ruff format src tests scripts ui

vendor-official:
	bash scripts/vendor_official.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache *.egg-info src/*.egg-info