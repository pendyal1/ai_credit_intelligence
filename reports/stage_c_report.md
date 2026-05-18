# Stage C XGBoost Model Report

Dataset: `utils/rawassi_synthetic_dataset.csv`
Stage A config: `config/hard_filters.json`
Stage B config: `config/benchmark_features.json`
Stage C config: `config/xgboost_model.json`

## Modeling Approach

The classifier is an XGBoost gradient boosting model trained only on deals that passed Stage A. Inputs are the Stage B benchmark and peer-outlier features. Hyperparameters are tuned with `GridSearchCV` using `TimeSeriesSplit` inside the training window; the final score is evaluated once on the untouched out-of-time validation set.

Class imbalance is handled with `scale_pos_weight`, computed from the Stage-A-passed training split.

- `scale_pos_weight`: `24.3804`
- best CV average precision: `0.1027`
- selected threshold: `0.493120`
- best params: `{"colsample_bytree": 0.8, "learning_rate": 0.08, "max_depth": 2, "min_child_weight": 5, "n_estimators": 200, "reg_lambda": 1.0, "subsample": 0.8}`

The threshold is selected from the configured training review rate. This treats high-risk predictions as a human-review queue, which is appropriate because false negatives are more expensive than false positives in invoice discounting.

## Metrics

| split      |   rows |   defaults |   default_rate |   auc_pr |   auc_roc |       ks |   brier_score |   threshold |   predicted_positive_rate |   precision |   recall |   true_negatives |   false_positives |   false_negatives |   true_positives |
|:-----------|-------:|-----------:|---------------:|---------:|----------:|---------:|--------------:|------------:|--------------------------:|------------:|---------:|-----------------:|------------------:|------------------:|-----------------:|
| train      |   2335 |         92 |      0.0394004 | 0.456045 |  0.957828 | 0.834534 |      0.104398 |     0.49312 |                  0.2      |   0.190578  | 0.967391 |             1865 |               378 |                 3 |               89 |
| validation |    323 |         10 |      0.0309598 | 0.106821 |  0.742173 | 0.516613 |      0.211988 |     0.49312 |                  0.383901 |   0.0645161 | 0.8      |              197 |               116 |                 2 |                8 |

## Calibration Table - Validation

|   probability_bin |   rows |   avg_predicted_probability |   observed_default_rate |   defaults |   min_predicted_probability |   max_predicted_probability |
|------------------:|-------:|----------------------------:|------------------------:|-----------:|----------------------------:|----------------------------:|
|                 0 |     65 |                   0.0547683 |               0         |          0 |                   0.0129044 |                    0.110438 |
|                 1 |     64 |                   0.175968  |               0         |          0 |                   0.112687  |                    0.262742 |
|                 2 |     65 |                   0.358738  |               0.0153846 |          1 |                   0.263992  |                    0.465385 |
|                 3 |     64 |                   0.570159  |               0.09375   |          6 |                   0.467009  |                    0.644592 |
|                 4 |     65 |                   0.767794  |               0.0461538 |          3 |                   0.644601  |                    0.889898 |

## Feature Importance

| feature                                            |    gain |   weight |   gain_share |
|:---------------------------------------------------|--------:|---------:|-------------:|
| stage_b_outstanding_to_revenue_est_peer_z          | 89.3606 |       73 |    0.188844  |
| stage_b_outstanding_to_revenue_est_risk_percentile | 64.4495 |       17 |    0.136199  |
| outstanding_to_revenue_est                         | 43.5582 |       78 |    0.0920505 |
| stage_b_benchmark_score                            | 40.912  |       48 |    0.0864583 |
| stage_b_ticket_to_revenue_est_risk_percentile      | 38.9229 |       21 |    0.0822549 |
| stage_b_peer_outlier_score                         | 36.649  |       82 |    0.0774494 |
| advance_tenor_exposure                             | 34.8289 |       59 |    0.073603  |
| stage_b_ticket_to_revenue_est_peer_z               | 32.0443 |       48 |    0.0677184 |
| stage_b_advance_tenor_exposure_risk_percentile     | 31.2215 |        8 |    0.0659796 |
| stage_b_advance_tenor_exposure_peer_z              | 30.8126 |       55 |    0.0651156 |
| ticket_to_revenue_est                              | 30.4396 |       73 |    0.0643273 |
| stage_b_peer_outlier_count                         |  0      |        0 |    0         |
| stage_b_peer_high_risk_outlier_count               |  0      |        0 |    0         |

## Generated Artifacts

- `reports/stage_c_metrics.csv`
- `reports/stage_c_validation_predictions.csv`
- `reports/stage_c_open_predictions.csv`
- `reports/stage_c_gridsearch_results.csv`
- `reports/stage_c_calibration_table.csv`
- `reports/stage_c_calibration.png`
- `reports/stage_c_precision_recall.png`
- `reports/stage_c_confusion_matrix.png`
- `reports/stage_c_feature_importance.csv`
- `reports/stage_c_feature_importance.png`
- `models/stage_c_xgboost_model.joblib`