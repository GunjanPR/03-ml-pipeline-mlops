"""Latency benchmark for a running API: single-record and batched /predict calls.

    python scripts/bench_api.py --url http://127.0.0.1:8020 --requests 1000

Records are drawn from the held-out test split (data/processed/test.parquet) or from
examples/request.json when the split is not available. Writes reports/api_latency.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from ml_pipeline.data import FEATURES


def load_records(n: int) -> list[dict]:
    path = Path("data/processed/test.parquet")
    if path.exists():
        return pd.read_parquet(path)[FEATURES].head(n).to_dict("records")
    records = json.loads(Path("examples/request.json").read_text())["records"]
    return (records * (n // len(records) + 1))[:n]


def run(client: httpx.Client, url: str, payloads: list[dict]) -> list[float]:
    latencies = []
    for payload in payloads:
        start = time.perf_counter()
        client.post(f"{url}/predict", json=payload).raise_for_status()
        latencies.append(time.perf_counter() - start)
    return latencies


def summary(latencies: list[float], rows_per_call: int) -> dict:
    ms = np.array(latencies) * 1000
    total = sum(latencies)
    return {
        "calls": len(latencies),
        "rows_per_call": rows_per_call,
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(statistics.fmean(ms)),
        "rows_per_second": len(latencies) * rows_per_call / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8020")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--label", default="local")
    args = parser.parse_args()

    records = load_records(max(args.requests, args.batch * 50))
    single = [{"records": [r]} for r in records[: args.requests]]
    batches = [
        {"records": records[i : i + args.batch]} for i in range(0, args.batch * 50, args.batch)
    ]
    with httpx.Client(timeout=30) as client:
        client.get(f"{args.url}/ready").raise_for_status()
        run(client, args.url, single[:50])  # warm-up
        result = {
            "label": args.label,
            "url": args.url,
            "machine": f"{platform.processor() or platform.machine()} / {platform.system()}",
            "single": summary(run(client, args.url, single), 1),
            "batched": summary(run(client, args.url, batches), args.batch),
        }
    out = Path("reports/api_latency.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
