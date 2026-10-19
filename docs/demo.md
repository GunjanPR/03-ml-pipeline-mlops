# Demo: from raw data to a monitored model

A real session on the development laptop (Intel Core i5-13420H, Windows 11, Docker
Desktop 29.8), recorded on 2026-10-19. Outputs are copied from the terminal; long
JSON is shortened where marked.

## 1. Reproduce the pipeline

```text
$ make repro          # dvc repro
Running stage 'fetch':
Running stage 'prepare':
Running stage 'train_baseline':
INFO baseline v1 test ROC-AUC 0.6682 [0.6537, 0.6815] PR-AUC 0.2200 F1 0.2741 Brier 0.0945 (fit 3.2s)
Running stage 'train_xgboost':
INFO optuna best val ROC-AUC 0.6744 after 50 trials
INFO xgboost v2 test ROC-AUC 0.6717 [0.6580, 0.6858] PR-AUC 0.2351 F1 0.2884 Brier 0.0936 (fit 197.5s)
Running stage 'gate':
INFO gate BLOCKED: v2 vs v1 (baseline) AUC 0.6717 vs 0.6682, lift +0.0035 [-0.0041, +0.0108]
     - AUC lift +0.0035 < required +0.0100; lift CI lower bound -0.0041 is not above 0
```

The tuned model is registered as `challenger` but never becomes `champion`.

## 2. Ship the champion

```text
$ make up             # export bundle -> docker build -> API + Prometheus + Grafana
INFO exported tabular-classifier v1 (baseline) -> model_bundle
 Container 03-ml-pipeline-mlops-prometheus-1 Started
 Container 03-ml-pipeline-mlops-api-1 Started
 Container 03-ml-pipeline-mlops-grafana-1 Started
```

## 3. Call the API

```text
$ curl -s localhost:8020/ready
{"status":"ready","model_version":"1"}

$ curl -s -X POST localhost:8020/predict -H "Content-Type: application/json" -d @examples/request.json
{"model_version":"1","threshold":0.13661711591504602,"predictions":[{"probability":0.09956452724880034,"readmit_30d":false},{"probability":0.09933522946547729,"readmit_30d":false}]}

$ curl -s -X POST localhost:8020/predict -H "Content-Type: application/json" -d '{"records": [{"age": "[70-80)"}]}'
{"detail":[{"type":"missing","loc":["body","records",0,"time_in_hospital"],"msg":"Field required", ...]}
```

Incomplete records are rejected with a 422 that names every missing field. They never reach
the model.

## 4. Drift arrives

1,500 encounters from an older, sicker case mix were sent through `/predict`. Then the
monitor job ran over the inference log:

```text
$ docker compose --profile jobs run --rm monitor
INFO drift share 0.22 (threshold 0.30) on 14242 rows -> DRIFT: key feature drift: discharge_disposition_id, medical_specialty, payer_code

$ curl -s localhost:8020/metrics | grep -E "^mlp_(predictions_total|drift|predict_path)"
mlp_predictions_total{model_version="1"} 8792.0
mlp_predict_path_total{path="fast"} 2041.0
mlp_predict_path_total{path="sklearn"} 50.0
mlp_drift_share 0.22448979591836735
mlp_drift_detected 1.0
```

The share of drifted features (22%) is below the 30% threshold, but three of the model's
key features moved, so the alarm fires. Prometheus raises `FeatureDriftDetected`, and
`python -m ml_pipeline.monitor --retrain` would retrain the challenger and send it through
the same gate.

![Grafana dashboard](grafana-dashboard.png)

*Grafana after the session. The latency drop at 04:05 is the fast scoring path (server-side
handler time 20 ms → 0.7 ms). The heatmap shows the predicted-risk distribution per minute.*

## 5. Kubernetes

The same image is deployed by CI to a kind cluster on every push: `kubectl apply -k k8s/`,
rollout, a `/predict` call through the Service, and one run of the drift CronJob (see the
`image-and-k8s` job in [the CI workflow](../.github/workflows/ci.yml)).
