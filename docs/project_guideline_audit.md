# Project Guideline Audit

This audit compares the current repository against `utils/DS_Intern_Case_Study.pdf` and `utils/Rawasi_Data_Dictionary.pdf`.

## Data Dictionary Compliance

| Requirement | Status | Evidence |
|---|---|---|
| Use only resolved rows for training/validation | Pass | `src/rawasi_credit/data.py` filters `defaulted.notna()` before the time split. |
| Use out-of-time validation from `2025-10-01` onward | Pass | `DEFAULT_SPLIT_DATE = 2025-10-01`; train rows are before that date, validation rows are on/after it. |
| Hold open rows out from validation | Pass | `split_resolved_open` returns open rows separately; open rows are scored only as holdout/sanity checks. |
| Avoid leaky columns | Pass | `config/benchmark_features.json` and `config/xgboost_model.json` do not include `actual_resolution_date`, `days_to_resolution`, `status`, `defaulted`, `expected_payment_date`, or `buy_buyer_baseline_default_prob`. |
| Submit saved model artifact | Pass | `models/stage_c_xgboost_model.joblib`. |
| Provide script/notebook to score one transaction | Pass | `scripts/score_one_transaction.py`. |

## Case Study Compliance

| Area | Status | Evidence |
|---|---|---|
| Layer 1A hard filters are configurable | Pass | `config/hard_filters.json`, `src/rawasi_credit/filters.py`. |
| Stage A rejects before model scoring | Pass | Stage B/C are fit only on Stage-A-passed rows. |
| Layer 1B has at least three cross-sector features | Pass | `config/benchmark_features.json` contains exposure/revenue, tenor, liquidity, buyer-risk, trade-history, and peer-outlier features. |
| Peer-group outlier detection | Pass | Stage B computes robust z-score outliers within `goods_category`, with global fallback for sparse groups. |
| New-sector handling | Pass | Documented in Stage B notebook/docs via global fallback and Layer 3 cold-start design. |
| Gradient boosting model choice | Pass | Stage C uses `XGBClassifier`, appropriate for tabular credit data. |
| Class imbalance handled | Pass | Stage C uses `scale_pos_weight` from training class balance. |
| No random K-fold validation | Pass | Outer validation is out-of-time; GridSearchCV uses `TimeSeriesSplit` inside the training window. |
| AUC-PR or KS reported | Pass | `reports/stage_c_metrics.csv` includes AUC-PR and KS. |
| Calibration plot/table included | Pass | `reports/stage_c_calibration.png` and `reports/stage_c_calibration_table.csv`. |
| Feature importance included | Pass | `reports/stage_c_shap_feature_importance.csv/png`, `reports/stage_c_validation_shap_explanations.csv`, and legacy XGBoost gain importance. |
| Confusion matrix and threshold rationale | Pass | `reports/stage_c_confusion_matrix.png`, `reports/stage_c_threshold_analysis.csv`, and Stage C report/notebook. |
| Layer 2 trend/anomaly detection | Pass | `src/rawasi_credit/intelligence.py`, `reports/layer2_trend_clusters.csv`, `reports/layer2_distribution_shift.csv`, `reports/layer2_transaction_anomalies.csv`. |
| Layer 2 LLM deal summary | Pass | `src/rawasi_credit/llm_summary.py` and `notebooks/02_layer2_intelligence.ipynb`; prompt uses decision-time context and model/anomaly signals. |
| Layer 3 blind spots with evidence | Pass | `notebooks/03_layer3_blind_spot_analysis.ipynb`. |
| README two-page summary | Pass | `README.md` covers architecture, decisions, next week, limitations, and time spent. |

## Corrections Made During Audit

- Removed outcome `status` from the LLM deal-summary context so the Task 2B prompt does not leak post-decision outcome information.
- Added `scripts/score_one_transaction.py` to explicitly satisfy the data-dictionary submission requirement for loading the saved model and scoring one transaction.
- Re-executed `notebooks/02_layer2_intelligence.ipynb` without an OpenRouter key so saved notebook outputs do not contain API secrets or outcome status in the LLM context.

## Remaining Caveats

- SHAP values are computed for Stage-C-scored validation rows and are expressed in model margin/log-odds contribution space. Stage-A-rejected rows do not receive SHAP explanations because they are stopped by policy before model scoring.
- Layer 2 cluster alerts are small-sample early warnings, not automatic reject rules.
- The dataset is synthetic, so all model governance recommendations should be revisited after real closed deals accumulate.
