"""Settings loaded from the environment and `.env` (see `.env.example`)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Settings(BaseSettings):
    """Precedence: constructor arguments > environment > `.env` > `params.yaml` > defaults.
    `params.yaml` holds the experiment parameters that DVC tracks."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", yaml_file="params.yaml")

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

    # Model the API serves; empty = models:/<REGISTERED_MODEL_NAME>@champion
    model_uri: str = ""
    # Each /predict call appends its inputs and outputs here (JSON lines), for drift monitoring
    inference_log_path: Path = Path("./logs/inference.jsonl")
    max_batch_size: int = 1000

    api_host: str = "0.0.0.0"
    api_port: int = 8020
    log_level: str = "info"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @property
    def resolved_model_uri(self) -> str:
        return self.model_uri or f"models:/{self.registered_model_name}@champion"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
