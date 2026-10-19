"""Settings loaded from the environment and `.env` (see `.env.example`)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "ml-pipeline"
    registered_model_name: str = "tabular-classifier"

    data_dir: Path = Path("./data")
    reports_dir: Path = Path("./reports")
    dataset_url: str = (
        "https://archive.ics.uci.edu/static/public/296/"
        "diabetes+130-us+hospitals+for+years+1999-2008.zip"
    )

    random_seed: int = 42
    optuna_n_trials: int = 50

    min_auc_lift: float = 0.01
    drift_threshold: float = 0.3

    api_host: str = "0.0.0.0"
    api_port: int = 8020
    log_level: str = "info"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
