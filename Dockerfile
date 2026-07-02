# GPU image for serving (and training). Requires the NVIDIA Container Toolkit on
# the host and `--gpus all` at runtime.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY configs ./configs
COPY scripts ./scripts

# Serving config comes from env (see .env.example): BASE_MODEL, TABLES_PATH,
# ADAPTER_DIR, SERIALIZE_KWARGS. Data + adapters are mounted at runtime.
EXPOSE 8000
CMD ["uvicorn", "text2sql.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]