# ML Pipeline + MLOps

An end-to-end model lifecycle: validated data, training, MLflow registry, a FastAPI service on Kubernetes, Evidently drift monitoring, and GitHub Actions quality gates.

> Part of the Live Projects portfolio. See [SHARED_STANDARDS.md](../SHARED_STANDARDS.md) for conventions and the milestone definition of done.

## Overview

This project shows production ML habits on a public tabular classification problem:
predicting **30-day hospital readmission** for diabetic inpatients, using the
[UCI Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296)
dataset (101,766 encounters, 71,518 patients). Encounters that ended in death or hospice are
dropped (they cannot be readmitted), leaving 99,343. Train / val / test (70 / 15 / 15) are
split **by patient**, so no patient appears in two splits. The pipeline covers the whole
lifecycle:

- Data versioned with **DVC** and validated with **Pandera**
- Training with scikit-learn and XGBoost, tuned with Optuna
- Experiments and a staged model registry in **MLflow**
- Serving through a FastAPI service on a local **Kubernetes** (kind/minikube) cluster
- Drift monitoring with **Evidently**, which triggers retraining
- **GitHub Actions**, which gates registry promotion on tests and a model-quality threshold

## Tech Stack

| Layer | Tools |
|---|---|
| Data versioning / validation | DVC, Pandera |
| Training | scikit-learn, XGBoost, Optuna, SHAP |
| Tracking / registry | MLflow |
| Serving | FastAPI, Docker, Kubernetes (kind or minikube) |
| Monitoring | Evidently, Prometheus, Grafana |
| CI/CD | GitHub Actions |

## Architecture

| Component | Responsibility |
|---|---|
| **DVC pipeline** | `dvc.yaml` stages: fetch → validate → featurize → train → evaluate |
| **Validation** | Pandera schemas reject bad data before training |
| **Training** | scikit-learn/XGBoost with an Optuna search; logs params, metrics, and SHAP plots to MLflow |
| **Registry** | MLflow model registry: `None → Staging → Production`, gated by CI |
| **Serving** | FastAPI `/predict` loads the Production model; deployed to kind/minikube |
| **Monitoring** | Evidently drift reports on logged inference data; Prometheus metrics; alert above threshold |
| **Retraining** | Scheduled or drift-triggered job re-runs `dvc repro` and re-registers the model |

```
 raw data ──► DVC: validate ─► featurize ─► train (Optuna) ─► evaluate
    (DVC remote)                               │                │
                                               ▼                ▼
                                    MLflow tracking ──► MLflow registry
                                                        (Staging→Production)
                                                                │  CI gate
 GitHub Actions: lint, test, quality gate ──────────────────────┤
                                                                ▼
             clients ──► FastAPI /predict  (Docker on Kubernetes)
                               │ inference log
                               ▼
                      Evidently drift report ──► alert ──► retrain job
                      Prometheus ──► Grafana
```

## Setup

### Prerequisites
- Python 3.11
- Git (DVC builds on top of git)
- Docker Desktop
- `kubectl` plus **kind** or **minikube** for the Kubernetes deploy (M6)
- Optional: a DVC remote (a local folder, Google Drive, or S3-compatible storage)
- Optional: GNU Make (see the Windows note below)

### Install
```bash
git clone <repo-url> 03-ml-pipeline-mlops && cd 03-ml-pipeline-mlops
py -3.11 -m venv .venv            # Windows; on macOS/Linux use: python3.11 -m venv .venv
source .venv/Scripts/activate     # Git Bash; PowerShell: .venv\Scripts\Activate.ps1; macOS/Linux: source .venv/bin/activate
make install                      # = pip install -e ".[dev]"
```

### Environment config
```bash
cp .env.example .env              # then fill in values; .env is git-ignored
```
`MLFLOW_TRACKING_URI` defaults to the local `./mlruns` folder. Point it at a running
`mlflow server` to use the UI and registry. Set the DVC remote and the drift and quality
thresholds here too.

## How to Run

> **Windows note:** `make` isn't installed on Windows by default. Install it with `choco install make`, or run the
> targets from **Git Bash**. You can also run the commands in the `Makefile` yourself
> (for example, `python -m pytest` instead of `make test`).

### Run the pipeline
```bash
make data                             # download (once), clean, patient-grouped split -> data/processed/
make run                              # = python -m ml_pipeline.train  (later: dvc repro)
make mlflow-ui                        # MLflow UI on http://localhost:5000 (SQLite store: mlflow.db)
```
`make run` prepares the data itself if `data/processed/` is empty. Metrics go to MLflow and to
`reports/<model>_metrics.json`.

### Serve the model (M2+) and deploy (M6)
```bash
make serve                            # FastAPI /predict on http://localhost:8020
# M6: kind create cluster && kubectl apply -f k8s/
```

### Lint and test
```bash
make lint     # ruff check + ruff format --check
make test     # pytest (unit tests only; integration tests are marked `integration`)
make format   # auto-fix lint and formatting
```

## Evaluation & Results

- **Baseline:** a logistic regression trained on the same features and split.
- **Metrics:** ROC-AUC, PR-AUC, F1, and calibration (Brier score) on a held-out test set.
- **MLOps checks:** one command reproduces the retrain; drift on a simulated shift is
  detected within one batch; the CI quality gate blocks promotion of models that perform
  worse.
- **Threshold:** the F1 decision threshold is chosen on validation and applied once to test.

### Measured baseline (M1)

Logistic regression on scaled counts + one-hot categoricals (ICD-9 codes grouped as in
Strack et al., 2014). Test set: 14,851 encounters, 11.1% readmitted within 30 days.

| Model | ROC-AUC (95% CI) | PR-AUC | F1 | Brier | Fit | Predict |
|---|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.668 (0.653–0.681) | 0.223 | 0.276 | 0.094 | 2.6 s | 7.5 ms / 1k rows |

Measured 2026-10-19 on a laptop CPU (Python 3.11, scikit-learn). CI is a 1,000-sample bootstrap.
Results for the tuned models are *targets until measured*.

## Milestones

Milestones follow [SHARED_STANDARDS.md](../SHARED_STANDARDS.md). Each one has a definition of done there.

### M1: Scaffold & baseline
- [x] Repo scaffold: `pyproject.toml`, Makefile, `.gitignore`, `.env.example`
- [x] Smoke test passes (`make test`)
- [x] Dataset chosen; baseline logistic regression trained and scored

### M2: Core functionality
- [ ] Feature engineering + XGBoost training with Optuna search
- [ ] MLflow tracking (params, metrics, SHAP artifacts)
- [ ] FastAPI `/predict` service loading the registered model

### M3: Data / storage layer
- [ ] DVC data versioning + remote
- [ ] `dvc.yaml` pipeline stages; `dvc repro` reproduces end to end
- [ ] Pandera data validation stage

### M4: Evaluation & testing
- [ ] Unit tests for features, validation, and API
- [ ] Model quality gate vs baseline (blocks registry promotion)
- [ ] Evaluation report on held-out test set

### M5: Observability / monitoring
- [ ] Evidently drift reports on inference logs
- [ ] Prometheus metrics + Grafana dashboard; drift alert threshold
- [ ] Drift-triggered / scheduled retraining job

### M6: Containerization & CI/CD
- [ ] Dockerfile for the serving image
- [ ] Kubernetes manifests; deploy to kind/minikube
- [ ] GitHub Actions: lint, test, train-and-gate, image build

### M7: Docs, benchmarks & demo
- [ ] README with measured results vs baseline
- [ ] Drift simulation write-up
- [ ] Demo recording

## Project Layout
```
03-ml-pipeline-mlops/
├── src/ml_pipeline/
│   ├── config.py        # pydantic-settings (reads .env)
│   ├── data.py          # fetch, clean, patient-grouped split
│   ├── features.py      # shared preprocessing
│   ├── evaluate.py      # ROC-AUC, PR-AUC, F1, Brier, bootstrap CI
│   └── train.py         # train + MLflow logging
├── tests/
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```
