# Windows: install make with `choco install make`, or run from Git Bash.
PYTHON ?= python

.PHONY: install lint format test data run train-xgboost gate repro monitor simulate bundle up down k8s-deploy k8s-delete serve mlflow-ui

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

monitor:                    # drift check over the inference log
	$(PYTHON) -m ml_pipeline.monitor

simulate:                   # drift scenarios, end-to-end API loop, retrain demo
	$(PYTHON) -m ml_pipeline.simulate --batches 20 --e2e --retrain-demo

bundle:                     # champion -> model_bundle/ (copied into the image)
	$(PYTHON) -m ml_pipeline.export --out model_bundle

up: bundle
	docker compose up --build -d

down:
	docker compose down

k8s-deploy: bundle         # Docker Desktop Kubernetes or kind (see k8s/README.md)
	docker build -t readmission-api:local .
	kubectl apply -k k8s/
	kubectl -n readmission rollout status deploy/readmission-api --timeout=180s

k8s-delete:
	kubectl delete -k k8s/

serve:
	$(PYTHON) -m uvicorn ml_pipeline.serve:app --reload --host 0.0.0.0 --port 8020

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
