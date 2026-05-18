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

- the three raw cross-sector benchmark features
- their global risk percentiles
- their within-sector peer z-scores
- aggregate Stage B benchmark and outlier scores

## Run

```bash
python scripts/run_stage_c_model.py
```

Generated files:

- `reports/stage_c_report.md`
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
