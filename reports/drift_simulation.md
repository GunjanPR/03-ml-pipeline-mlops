| scenario             |   batches |   alarm_rate |   share_rule_rate |   key_feature_rule_rate |   mean_drift_share |   prediction_drift_rate |   roc_auc |   positive_rate |   mean_prediction | most_flagged                                                     |
|:---------------------|----------:|-------------:|------------------:|------------------------:|-------------------:|------------------------:|----------:|----------------:|------------------:|:-----------------------------------------------------------------|
| no_shift             |        20 |        1.000 |             0.950 |                   1.000 |              0.320 |                   0.150 |     0.672 |           0.106 |             0.119 | admission_source_id, admission_type_id, diag_2_group             |
| older_sicker         |        20 |        1.000 |             1.000 |                   1.000 |              0.382 |                   1.000 |     0.655 |           0.146 |             0.150 | admission_source_id, admission_type_id, age                      |
| emergency_intake     |        20 |        1.000 |             1.000 |                   1.000 |              0.360 |                   0.950 |     0.668 |           0.114 |             0.124 | admission_source_id, admission_type_id, diag_2_group             |
| icd10_migration      |        20 |        1.000 |             1.000 |                   1.000 |              0.359 |                   0.050 |     0.662 |           0.107 |             0.118 | admission_source_id, admission_type_id, diag_1_group             |
| lab_panel_change     |        20 |        1.000 |             0.900 |                   1.000 |              0.321 |                   0.500 |     0.673 |           0.106 |             0.121 | admission_source_id, admission_type_id, diag_2_group             |
| discharge_code_remap |        20 |        1.000 |             0.900 |                   1.000 |              0.318 |                   1.000 |     0.587 |           0.107 |             0.158 | admission_source_id, admission_type_id, discharge_disposition_id |

## Retraining after a shift

| shift                | model_kind   |   stale_auc_before_shift |   stale_auc_after_shift |   retrained_auc_after_shift |   auc_lift |   auc_lift_ci_low |   auc_lift_ci_high | gate          |
|:---------------------|:-------------|-------------------------:|------------------------:|----------------------------:|-----------:|------------------:|-------------------:|:--------------|
| discharge_code_remap | xgboost      |                   0.6766 |                  0.5895 |                      0.6793 |     0.0898 |            0.0737 |             0.1064 | promote       |
| icd10_migration      | xgboost      |                   0.6766 |                  0.6751 |                      0.6755 |     0.0004 |           -0.0032 |             0.0037 | keep champion |

## End to end

1000 `older_sicker` rows sent through `/predict`; monitor on the inference log: drift share 0.39, drifted = True.
