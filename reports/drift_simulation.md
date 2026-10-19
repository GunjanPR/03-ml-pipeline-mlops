| scenario             |   batches |   alarm_rate |   share_rule_rate |   key_feature_rule_rate |   mean_drift_share |   prediction_drift_rate |   roc_auc |   positive_rate |   mean_prediction | most_flagged                                              |
|:---------------------|----------:|-------------:|------------------:|------------------------:|-------------------:|------------------------:|----------:|----------------:|------------------:|:----------------------------------------------------------|
| no_shift             |        20 |        0.000 |             0.000 |                   0.000 |              0.000 |                   0.000 |     0.677 |           0.113 |             0.115 |                                                           |
| older_sicker         |        20 |        1.000 |             0.000 |                   1.000 |              0.210 |                   1.000 |     0.646 |           0.143 |             0.145 | age, number_diagnoses, number_inpatient                   |
| emergency_intake     |        20 |        1.000 |             0.000 |                   1.000 |              0.136 |                   0.100 |     0.669 |           0.117 |             0.119 | admission_source_id, admission_type_id, medical_specialty |
| icd10_migration      |        20 |        1.000 |             0.000 |                   1.000 |              0.061 |                   0.300 |     0.665 |           0.113 |             0.110 | diag_1_group, diag_2_group, diag_3_group                  |
| lab_panel_change     |        20 |        0.000 |             0.000 |                   0.000 |              0.022 |                   0.000 |     0.665 |           0.112 |             0.116 | num_lab_procedures, time_in_hospital, total_prior_visits  |
| discharge_code_remap |        20 |        1.000 |             0.000 |                   1.000 |              0.021 |                   1.000 |     0.607 |           0.113 |             0.152 | discharge_disposition_id, medical_specialty               |

## Retraining after a shift

| shift                | model_kind   |   stale_auc_before_shift |   stale_auc_after_shift |   retrained_auc_after_shift |   auc_lift |   auc_lift_ci_low |   auc_lift_ci_high | gate          |
|:---------------------|:-------------|-------------------------:|------------------------:|----------------------------:|-----------:|------------------:|-------------------:|:--------------|
| discharge_code_remap | baseline     |                   0.6682 |                  0.6068 |                      0.6680 |     0.0611 |            0.0479 |             0.0744 | promote       |
| icd10_migration      | baseline     |                   0.6682 |                  0.6662 |                      0.6675 |     0.0013 |           -0.0005 |             0.0030 | keep champion |

## End to end

1000 `older_sicker` rows sent through `/predict`; monitor on the inference log: drift share 0.27, drifted = True.
