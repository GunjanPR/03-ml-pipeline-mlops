| scenario             |   batches |   alarm_rate |   share_rule_rate |   key_feature_rule_rate |   mean_drift_share |   prediction_drift_rate |   roc_auc |   positive_rate |   mean_prediction | most_flagged                                                  |
|:---------------------|----------:|-------------:|------------------:|------------------------:|-------------------:|------------------------:|----------:|----------------:|------------------:|:--------------------------------------------------------------|
| no_shift             |        20 |        1.000 |             0.000 |                   1.000 |              0.065 |                   0.000 |     0.672 |           0.106 |             0.119 | medical_specialty, number_diagnoses, payer_code               |
| older_sicker         |        20 |        1.000 |             0.000 |                   1.000 |              0.181 |                   1.000 |     0.655 |           0.146 |             0.150 | age, medical_specialty, number_diagnoses                      |
| emergency_intake     |        20 |        1.000 |             0.000 |                   1.000 |              0.113 |                   0.000 |     0.668 |           0.114 |             0.124 | admission_source_id, admission_type_id, medical_specialty     |
| icd10_migration      |        20 |        1.000 |             0.000 |                   1.000 |              0.126 |                   0.250 |     0.662 |           0.107 |             0.118 | diag_1_group, diag_2_group, diag_3_group                      |
| lab_panel_change     |        20 |        0.950 |             0.000 |                   0.950 |              0.062 |                   0.050 |     0.673 |           0.106 |             0.121 | num_lab_procedures, medical_specialty, number_diagnoses       |
| discharge_code_remap |        20 |        1.000 |             0.000 |                   1.000 |              0.080 |                   1.000 |     0.587 |           0.107 |             0.158 | discharge_disposition_id, medical_specialty, number_diagnoses |

## Retraining after a shift

| shift                | model_kind   |   stale_auc_before_shift |   stale_auc_after_shift |   retrained_auc_after_shift |   auc_lift |   auc_lift_ci_low |   auc_lift_ci_high | gate          |
|:---------------------|:-------------|-------------------------:|------------------------:|----------------------------:|-----------:|------------------:|-------------------:|:--------------|
| discharge_code_remap | xgboost      |                   0.6766 |                  0.5895 |                      0.6793 |     0.0898 |            0.0737 |             0.1064 | promote       |
| icd10_migration      | xgboost      |                   0.6766 |                  0.6751 |                      0.6755 |     0.0004 |           -0.0032 |             0.0037 | keep champion |

## End to end

1000 `older_sicker` rows sent through `/predict`; monitor on the inference log: drift share 0.27, drifted = True.
