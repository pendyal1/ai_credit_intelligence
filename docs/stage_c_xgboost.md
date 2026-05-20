# Stage C - XGBoost Classifier

Stage C is the default-probability model. It runs after:

1. Stage A hard filters
2. Stage B benchmark and peer outlier features

The current model intentionally trains only on Stage-A-passed resolved deals. Rejected deals are removed before model fitting because Stage A represents non-negotiable policy rules rather than modelable credit risk.

## Model

The model is an `XGBClassifier` with:

- objective: `binary:logistic`
- metric during fitting: `aucpr`
- class imbalance handling: `scale_pos_weight = non_defaults / defaults`
- hyperparameter tuning: `GridSearchCV`
- CV splitter: `TimeSeriesSplit`

Using `TimeSeriesSplit` keeps tuning time-aware. The final validation set remains the case-study out-of-time split and is not used for grid search.

## Features

The feature set is defined in `config/xgboost_model.json`. It uses Stage B outputs only:

- raw cross-sector benchmark features
- global risk percentiles
- within-sector peer z-scores for continuous peer-comparable features
- aggregate Stage B benchmark and outlier scores

## Threshold

The probability cutoff is configured in `config/xgboost_model.json`. The current cutoff is deliberately recall-first: Stage C is meant to catch likely defaults for review, so false negatives are treated as more expensive than false positives. `reports/stage_c_threshold_analysis.csv` shows how false positives, false negatives, precision, and recall move across candidate thresholds, including stricter high-precision alternatives.

## Run

```bash
python scripts/run_stage_c_model.py
```

Generated files:

- `reports/stage_c_report.md`
- `reports/stage_c_metrics.csv`
- `reports/stage_c_validation_predictions.csv`
- `reports/layer1_validation_metrics.csv`
- `reports/layer1_validation_predictions.csv`
- `reports/stage_c_open_predictions.csv`
- `reports/stage_c_gridsearch_results.csv`
- `reports/stage_c_threshold_analysis.csv`
- `reports/stage_c_calibration_table.csv`
- `reports/stage_c_calibration.png`
- `reports/stage_c_precision_recall.png`
- `reports/stage_c_confusion_matrix.png`
- `reports/stage_c_feature_importance.csv`
- `reports/stage_c_feature_importance.png`
- `models/stage_c_xgboost_model.joblib`
