"""Export the champion into a self-contained bundle for the serving image.

    python -m ml_pipeline.export --out model_bundle

The bundle holds the MLflow model directory, `metadata.json` (registry name, version,
threshold, metrics) and `reference.parquet` (a training sample with predictions, used by the
drift monitor). The Docker image copies it in, so every image serves exactly one immutable
model version and needs no access to the tracking server at runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from ml_pipeline.config import Settings, get_settings
from ml_pipeline.train import CHAMPION

log = logging.getLogger(__name__)

METADATA = "metadata.json"
REFERENCE = "reference.parquet"
MODEL_DIR = "model"


def export(out: Path, settings: Settings | None = None, alias: str = CHAMPION) -> dict:
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    from ml_pipeline.monitor import reference_frame

    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    name = settings.registered_model_name
    version = MlflowClient().get_model_version_by_alias(name, alias)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{version.version}", dst_path=str(out / MODEL_DIR)
    )
    model = mlflow.sklearn.load_model(str(out / MODEL_DIR))
    reference_frame(settings, model).to_parquet(out / REFERENCE, index=False)
    metadata = {
        "registered_model": name,
        "version": str(version.version),
        "alias": alias,
        "model_kind": version.tags.get("model_kind"),
        "threshold": float(version.tags.get("threshold", 0.5)),
        "val_roc_auc": float(version.tags.get("val_roc_auc", "nan")),
        "test_roc_auc": float(version.tags.get("test_roc_auc", "nan")),
        "run_id": version.run_id,
    }
    (out / METADATA).write_text(json.dumps(metadata, indent=2))
    log.info("exported %s v%s (%s) -> %s", name, version.version, metadata["model_kind"], out)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("model_bundle"))
    parser.add_argument("--alias", default=CHAMPION)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    export(args.out, alias=args.alias)


if __name__ == "__main__":
    main()
