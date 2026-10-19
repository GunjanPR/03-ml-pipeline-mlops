# Resume bullets: ML Pipeline + MLOps

Every number below is measured, and the source file is named next to it. Pick 2-3 bullets
per resume. The `Resumes/` folder has not been edited.

**Project line:** ML Pipeline + MLOps: 30-Day Readmission Risk | Python, XGBoost, Optuna,
SHAP, MLflow, DVC, Pandera, Evidently, FastAPI, Docker, Kubernetes, GitHub Actions |
github.com/GunjanPR/03-ml-pipeline-mlops

1. Built an end-to-end MLOps pipeline for 30-day readmission risk on 101,766 hospital
   encounters, with DVC-versioned, Pandera-validated data, Optuna-tuned XGBoost, an MLflow
   registry and a FastAPI service on Kubernetes. `dvc repro` rebuilds every metric from
   the raw download, with bit-identical processed data across Windows and Linux.
   *(reports/\*.json, dvc.lock)*

2. Designed a champion/challenger promotion gate (paired-bootstrap ROC-AUC lift with a 95%
   CI and a minimum lift, plus a calibration check). It blocked a 50-trial tuned XGBoost
   whose lift over logistic regression (+0.0087, CI +0.001 to +0.016, 14,851 test
   encounters) was real but below the +0.01 bar, which kept a 60× cheaper, interpretable
   model in production. Re-running the search on another OS moved the tuned model's AUC by
   0.005, which is why the bar exists. *(reports/gate.json, README "Reproducibility")*

3. Built a pre-deployment model audit (calibration ECE, top-k capture, subgroup AUC and
   sensitivity with bootstrap CIs) as a DVC stage. Both models are calibrated (ECE ≤ 0.008),
   the top 10% of scores catches 2.4× the random share of readmissions, and the largest gap
   is by age (AUC 0.745 under 40 vs 0.619 over 80), not race; a race-blind ablation cost
   0.0002 AUC. *(reports/audit.md, reports/audit_race_ablation.md)*

4. Improved drift detection from 0 of 5 to 4 of 5 simulated data shifts, each caught within
   one 1,000-encounter batch (20/20 batches per shift), with 0/20 false alarms on unshifted
   data. The fix was to alarm on drift in the model's top SHAP features and in its output
   distribution, not only on the share of drifted features. *(reports/drift_simulation.md)*

5. Restored ROC-AUC by 10% (0.607 → 0.668) against the stale deployed model on 14,851
   held-out encounters after a simulated EHR code remap. Drift triggered the retraining,
   and the model was promoted only because the gate measured a significant lift
   (+0.061, CI +0.048 to +0.074). *(reports/drift_simulation.md)*

6. Shipped CI/CD (GitHub Actions) that lints and runs 63 tests with a coverage floor, reproduces the DVC
   pipeline on real data, gates the model, builds an immutable serving image with the
   champion baked in, and deploys it to a kind Kubernetes cluster to call `/predict` and
   run the drift CronJob on every push. *(.github/workflows/ci.yml)*

7. Cut server-side `/predict` latency for single-record requests by 30× (about 21 ms →
   0.70 ms; host end-to-end p50 3.2 ms over 1,000 calls) against the scikit-learn pipeline.
   The fix was a fast encoder built from the fitted preprocessor that only switches on
   after it reproduces the sklearn probabilities to 1e-9, plus asynchronous inference
   logging. *(reports/api_latency.json, README "Serving")*

**Short version (one line):** Built a gated MLOps pipeline (MLflow, DVC, Evidently,
Kubernetes) whose drift alarms caught 4/5 simulated shifts with 0 false alarms and whose
statistical promotion gate correctly refused a non-significant XGBoost upgrade.
