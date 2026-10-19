# Resume bullets: ML Pipeline + MLOps

Every number below is measured, and the source file is named next to it. Pick 2-3 bullets
per resume. The `Resumes/` folder has not been edited.

**Project line:** ML Pipeline + MLOps: 30-Day Readmission Risk | Python, XGBoost, Optuna,
SHAP, MLflow, DVC, Pandera, Evidently, FastAPI, Docker, Kubernetes, GitHub Actions |
github.com/GunjanPR/03-ml-pipeline-mlops

1. Built an end-to-end MLOps pipeline for 30-day readmission risk on 101,766 hospital
   encounters, with DVC-versioned, Pandera-validated data, Optuna-tuned XGBoost, an MLflow
   registry and a FastAPI service on Kubernetes. `dvc repro` reproduces every metric from
   the raw download. *(reports/\*.json, dvc.lock)*

2. Designed a champion/challenger promotion gate (paired-bootstrap ROC-AUC lift with a 95%
   CI, plus a calibration check). It blocked a 50-trial tuned XGBoost whose lift over
   logistic regression (+0.0035, CI −0.004 to +0.011, 14,851 test encounters) was not
   significant, which kept a 60× cheaper, interpretable model in production.
   *(reports/gate.json)*

3. Improved drift detection from 0 of 5 to 4 of 5 simulated data shifts, each caught within
   one 1,000-encounter batch (20/20 batches per shift), with 0/20 false alarms on unshifted
   data. The fix was to alarm on drift in the model's top SHAP features and in its output
   distribution, not only on the share of drifted features. *(reports/drift_simulation.md)*

4. Restored ROC-AUC by 10% (0.607 → 0.668) against the stale deployed model on 14,851
   held-out encounters after a simulated EHR code remap. Drift triggered the retraining,
   and the model was promoted only because the gate measured a significant lift
   (+0.061, CI +0.048 to +0.074). *(reports/drift_simulation.md)*

5. Shipped CI/CD (GitHub Actions) that lints and runs 53 tests, reproduces the DVC
   pipeline on real data, gates the model, builds an immutable serving image with the
   champion baked in, and deploys it to a kind Kubernetes cluster to call `/predict` and
   run the drift CronJob on every push. *(.github/workflows/ci.yml)*

**Short version (one line):** Built a gated MLOps pipeline (MLflow, DVC, Evidently,
Kubernetes) whose drift alarms caught 4/5 simulated shifts with 0 false alarms and whose
statistical promotion gate correctly refused a non-significant XGBoost upgrade.
