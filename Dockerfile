# Serving image: one immutable champion model per image.
#   make bundle && docker build -t readmission-api:<version> .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MODEL_BUNDLE=/app/model_bundle \
    INFERENCE_LOG_PATH=/app/logs/inference.jsonl \
    REPORTS_DIR=/app/reports

# libgomp1: OpenMP runtime for XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[monitor]" && useradd --create-home --uid 10001 api \
    && mkdir -p logs reports && chown api logs reports
COPY model_bundle ./model_bundle

USER api
EXPOSE 8020
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8020/ready')"
# One worker per container keeps Prometheus counters exact; scale with replicas instead.
CMD ["uvicorn", "ml_pipeline.serve:app", "--host", "0.0.0.0", "--port", "8020"]
