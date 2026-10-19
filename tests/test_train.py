import json

from conftest import fake_raw
from ml_pipeline.data import clean, split
from ml_pipeline.train import train


def test_baseline_trains_logs_and_reports(settings):
    settings.processed_dir.mkdir(parents=True)
    for name, frame in split(clean(fake_raw(900)), seed=0).items():
        frame.to_parquet(settings.processed_dir / f"{name}.parquet", index=False)

    result = train("baseline", settings)

    assert result["test"]["roc_auc"] > 0.6  # the fake data has a real signal
    assert result["mlflow_run_id"]
    saved = json.loads((settings.reports_dir / "baseline_metrics.json").read_text())
    assert saved["test"]["n"] == result["test"]["n"]
