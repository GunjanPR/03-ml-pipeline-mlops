# Windows: install make with `choco install make`, or run from Git Bash.
PYTHON ?= python

.PHONY: install lint format test data run train-xgboost gate repro serve mlflow-ui

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

data:
	$(PYTHON) -m ml_pipeline.data prepare

run:
	$(PYTHON) -m ml_pipeline.train

train-xgboost:
	$(PYTHON) -m ml_pipeline.train --model xgboost

gate:
	$(PYTHON) -m ml_pipeline.promote

repro:                      # full DVC pipeline: fetch -> prepare -> train x2 -> gate
	$(PYTHON) -m dvc repro

serve:
	$(PYTHON) -m uvicorn ml_pipeline.serve:app --reload --host 0.0.0.0 --port 8020

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
