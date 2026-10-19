# Kubernetes deployment

Manifests for a local cluster (Docker Desktop Kubernetes or kind), applied with Kustomize.

| File | What it creates |
|---|---|
| `namespace.yaml` | `readmission` namespace |
| `configmap.yaml` | Runtime settings (bundle path, drift threshold, batch limit) |
| `storage.yaml` | 1 Gi PVC for per-pod inference logs and drift reports |
| `deployment.yaml` | 2-replica API Deployment (non-root, read-only capabilities, probes, limits) + ClusterIP Service |
| `cronjob-monitor.yaml` | Daily Evidently drift check over every pod's log |
| `hpa.yaml` | CPU autoscaling 2-6 replicas (needs metrics-server) |

The image carries exactly one model version (`make bundle` exports the registry champion
into `model_bundle/`, and the Dockerfile copies it in), so a rollout is a model release and a
`kubectl rollout undo` is a model rollback.

## Deploy

```bash
make bundle                                   # champion -> model_bundle/
docker build -t readmission-api:local .
kind load docker-image readmission-api:local  # kind only; Docker Desktop shares images
kubectl apply -k k8s/
kubectl -n readmission rollout status deploy/readmission-api
kubectl -n readmission port-forward svc/readmission-api 8080:80
curl -X POST localhost:8080/predict -H 'Content-Type: application/json' -d @examples/request.json
```

`make k8s-deploy` runs the first four steps (Docker Desktop). Run the drift check now
instead of waiting for 06:00:

```bash
kubectl -n readmission create job drift-now --from=cronjob/drift-monitor
kubectl -n readmission logs job/drift-now
```

## Notes

* **Probes:** `/ready` returns 503 until the model is loaded, so a pod with a broken bundle
  never receives traffic and the rolling update (`maxUnavailable: 0`) stops.
* **Logs:** each pod writes `logs/<pod-name>.jsonl` on the shared PVC; the monitor reads
  them all. `ReadWriteOnce` works on a single-node cluster; on a multi-node cluster use a
  `ReadWriteMany` storage class or ship logs to object storage.
* **HPA:** kind and Docker Desktop do not include metrics-server. Without it the HPA stays
  at `<unknown>` and the Deployment keeps its 2 replicas.
* **CI:** `.github/workflows/ci.yml` validates these manifests with kubeconform, deploys them
  to a kind cluster, calls `/predict` through the Service and runs the drift CronJob once.
