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
   CI and a minimum lift, plus a calibration check) and evaluated it on a time-ordered split
   (train on the earliest 70% of encounters, test on the latest 15%). It promoted a 50-trial
   tuned XGBoost over logistic regression with lift +0.031 (CI +0.024 to +0.039, 14,901 test
   encounters), stable across 5 Optuna seeds (+0.031 to +0.041). *(reports/history/v5_time_gate.json, reports/robustness.md)*

2b. Found that a random patient-grouped split reversed the decision: there the lift was only
   +0.008 and the gate blocked XGBoost, because the random split flattered the linear baseline
   (0.668 vs 0.645 on later data). Switched the gate to the time-ordered split and added a
   guard so it never compares models trained on different splits. Both models beat the LACE
   clinical readmission index (0.590) by +0.055 to +0.087 AUC. *(README "Robustness")*

3. Built a pre-deployment model audit (calibration ECE, top-k capture, subgroup AUC and
   sensitivity with bootstrap CIs) as a DVC stage. The top 10% of XGBoost's scores catches
   2.5× the random share of readmissions; the audit also caught an 11% over-prediction caused
   by the readmission rate falling over time. The largest gap is by age (AUC 0.790 under 40 vs
   0.652 over 80), not race; a race-blind ablation changed AUC by 0.0002.
   *(reports/audit.md, reports/audit_race_ablation.md)*

3b. Traced the age gap to the data, not the model: prior inpatient stays predict readmission
   with AUC 0.71 under 40 but 0.57 over 80, and over-80s are readmitted at 10.7% with no prior
   stay. Five model variants (comorbidity flags from full ICD-9 codes, per-age-band models,
   tuned XGBoost) left the gap at 0.15-0.17, and per-band thresholds did not significantly
   equalise sensitivity (paired-bootstrap CI -0.080 to +0.168), so neither was shipped.
   *(reports/age_gap.md)*

4. Improved drift detection from 0 of 5 to 4 of 5 simulated data shifts, each caught within
   one 1,000-encounter batch (20/20 batches per shift), with 0/20 false alarms on unshifted
   data. The fix was to alarm on drift in the model's top SHAP features and in its output
   distribution, not only on the share of drifted features. Diagnosed and fixed 20/20 false
   alarms after the time-split switch (a stale reference period). *(reports/drift_simulation.md)*

5. Restored ROC-AUC by 15% (0.590 → 0.679) against the stale deployed model on 14,901
   held-out encounters after a simulated EHR code remap. Drift triggered the retraining,
   and the model was promoted only because the gate measured a significant lift
   (+0.090, CI +0.074 to +0.106); for an ICD-10 recoding that barely hurt accuracy, the same
   gate kept the champion. *(reports/drift_simulation.md)*

6. Shipped CI/CD (GitHub Actions) that lints and runs 71 tests with a coverage floor, reproduces the DVC
   pipeline on real data, gates the model, builds an immutable serving image with the
   champion baked in, and deploys it to a kind Kubernetes cluster to call `/predict` and
   run the drift CronJob on every push. *(.github/workflows/ci.yml)*

7. Cut server-side `/predict` latency for single-record requests by 30× (about 21 ms →
   0.70 ms) with a fast encoder built from the fitted preprocessor that only switches on
   after it reproduces the sklearn probabilities to 1e-9. When XGBoost became champion the
   check caught a mismatch (XGBoost reads absent sparse entries as missing); after the fix,
   end-to-end p50 is 1.39 ms on a Ryzen 7 9800X3D desktop. *(README "Serving")*

**Short version (one line):** Built a gated MLOps pipeline (MLflow, DVC, Evidently,
Kubernetes) whose drift alarms caught 4/5 simulated shifts with 0 false alarms and whose
statistical promotion gate, evaluated on a time-ordered split, promoted XGBoost on a
significant +0.031 AUC lift.
