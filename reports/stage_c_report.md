# Stage C XGBoost Model Report

Dataset: `utils/rawassi_synthetic_dataset.csv`
Stage A config: `config/hard_filters.json`
Stage B config: `config/benchmark_features.json`
Stage C config: `config/xgboost_model.json`

## Modeling Approach

The classifier is an XGBoost gradient boosting model trained only on deals that passed Stage A. Inputs are the Stage B benchmark and peer-outlier features. Hyperparameters are tuned with `GridSearchCV` using `TimeSeriesSplit` inside the training window; the final score is evaluated once on the untouched out-of-time validation set.

Class imbalance is handled with `scale_pos_weight`, computed from the Stage-A-passed training split.

- `scale_pos_weight`: `22.4462`
- best CV average precision: `0.2966`
- selected threshold: `0.400000`
- best params: `{"colsample_bytree": 0.7, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 10, "n_estimators": 250, "reg_lambda": 1.0, "subsample": 1.0}`

The threshold is loaded from `config/xgboost_model.json` and should be treated as the operating point for the high-risk review queue. The threshold analysis below shows the precision-recall tradeoff on the out-of-time validation set so the cutoff can be defended explicitly.
Threshold rationale: Recall-first operating threshold for default detection after Stage B feature expansion. On the out-of-time validation split this catches 13 of 14 Stage-A-passed defaults while reviewing about 30% of passed deals; full Layer 1 catches 19 of 20 defaults when Stage A rejects are included.

## Metrics

| split      |   rows |   defaults |   default_rate |   auc_pr |   auc_roc |       ks |   brier_score |   threshold |   predicted_positive_rate |   precision |   recall |   true_negatives |   false_positives |   false_negatives |   true_positives |
|:-----------|-------:|-----------:|---------------:|---------:|----------:|---------:|--------------:|------------:|--------------------------:|------------:|---------:|-----------------:|------------------:|------------------:|-----------------:|
| train      |   3048 |        130 |      0.0426509 | 0.724608 |  0.983904 | 0.915084 |     0.0623096 |         0.4 |                  0.151247 |    0.281996 | 1        |             2587 |               331 |                 0 |              130 |
| validation |    431 |         14 |      0.0324826 | 0.651733 |  0.929599 | 0.70555  |     0.117974  |         0.4 |                  0.25058  |    0.12037  | 0.928571 |              322 |                95 |                 1 |               13 |

## Full Layer 1 Validation Metrics

These metrics treat Stage A rejects and Stage C high-risk predictions as deals requiring rejection or manual review.

| scope                  |   rows |   defaults |   flagged_rows |   flagged_rate |   precision |   recall |   true_negatives |   false_positives |   false_negatives |   true_positives |
|:-----------------------|-------:|-----------:|---------------:|---------------:|------------:|---------:|-----------------:|------------------:|------------------:|-----------------:|
| full_layer1_validation |    573 |         20 |            250 |         0.4363 |       0.076 |     0.95 |              322 |               231 |                 1 |               19 |

## Calibration Table - Validation

|   probability_bin |   rows |   avg_predicted_probability |   observed_default_rate |   defaults |   min_predicted_probability |   max_predicted_probability |
|------------------:|-------:|----------------------------:|------------------------:|-----------:|----------------------------:|----------------------------:|
|                 0 |     87 |                   0.0283911 |               0         |          0 |                   0.0089143 |                    0.046834 |
|                 1 |     86 |                   0.069188  |               0         |          0 |                   0.047638  |                    0.09332  |
|                 2 |     86 |                   0.149076  |               0         |          0 |                   0.0940943 |                    0.214302 |
|                 3 |     86 |                   0.333301  |               0.0348837 |          3 |                   0.214377  |                    0.488979 |
|                 4 |     86 |                   0.718921  |               0.127907  |         11 |                   0.491103  |                    0.980649 |

## Threshold Analysis - Validation

|   threshold |   flagged_rows |   flagged_rate |   true_negatives |   false_positives |   false_negatives |   true_positives |   precision |   recall |        f1 |
|------------:|---------------:|---------------:|-----------------:|------------------:|------------------:|-----------------:|------------:|---------:|----------:|
|       0.05  |            336 |      0.779582  |               95 |               322 |                 0 |               14 |   0.0416667 | 1        | 0.08      |
|       0.066 |            305 |      0.707657  |              126 |               291 |                 0 |               14 |   0.0459016 | 1        | 0.0877743 |
|       0.1   |            251 |      0.582367  |              180 |               237 |                 0 |               14 |   0.0557769 | 1        | 0.10566   |
|       0.17  |            201 |      0.466357  |              230 |               187 |                 0 |               14 |   0.0696517 | 1        | 0.130233  |
|       0.2   |            181 |      0.419954  |              250 |               167 |                 0 |               14 |   0.0773481 | 1        | 0.14359   |
|       0.3   |            140 |      0.324826  |              290 |               127 |                 1 |               13 |   0.0928571 | 0.928571 | 0.168831  |
|       0.4   |            108 |      0.25058   |              322 |                95 |                 1 |               13 |   0.12037   | 0.928571 | 0.213115  |
|       0.5   |             84 |      0.194896  |              344 |                73 |                 3 |               11 |   0.130952  | 0.785714 | 0.22449   |
|       0.55  |             75 |      0.174014  |              353 |                64 |                 3 |               11 |   0.146667  | 0.785714 | 0.247191  |
|       0.6   |             67 |      0.155452  |              361 |                56 |                 3 |               11 |   0.164179  | 0.785714 | 0.271605  |
|       0.65  |             57 |      0.132251  |              370 |                47 |                 4 |               10 |   0.175439  | 0.714286 | 0.28169   |
|       0.7   |             43 |      0.099768  |              384 |                33 |                 4 |               10 |   0.232558  | 0.714286 | 0.350877  |
|       0.75  |             34 |      0.0788863 |              392 |                25 |                 5 |                9 |   0.264706  | 0.642857 | 0.375     |
|       0.8   |             27 |      0.062645  |              399 |                18 |                 5 |                9 |   0.333333  | 0.642857 | 0.439024  |
|       0.85  |             19 |      0.0440835 |              406 |                11 |                 6 |                8 |   0.421053  | 0.571429 | 0.484848  |
|       0.9   |             12 |      0.0278422 |              413 |                 4 |                 6 |                8 |   0.666667  | 0.571429 | 0.615385  |

## SHAP Feature Importance - Validation

SHAP values are computed on the out-of-time validation rows. They explain the model margin/log-odds contribution, so positive values push a deal toward higher default risk and negative values push it toward lower default risk.

| feature                                                |   mean_abs_shap |   mean_shap |     gain |   gain_share |
|:-------------------------------------------------------|----------------:|------------:|---------:|-------------:|
| log_cumulative_outstanding_usd                         |       0.540196  | -0.120486   | 248.337  |   0.0375499  |
| stage_b_exp_debt_to_equity_peer_z                      |       0.380535  | -0.169801   |  79.2273 |   0.0119796  |
| stage_b_benchmark_score                                |       0.222053  | -0.148877   |  50.3401 |   0.00761168 |
| stage_b_log_cumulative_outstanding_usd_risk_percentile |       0.177816  | -0.0402108  | 230.413  |   0.0348396  |
| stage_b_exposure_quality_pressure_peer_z               |       0.126187  |  0.0178245  | 120.932  |   0.0182855  |
| stage_b_concentration_adjusted_total_exposure_peer_z   |       0.106367  | -0.0428139  |  32.7089 |   0.00494575 |
| ticket_to_revenue_est                                  |       0.105654  | -0.0075848  |  53.6052 |   0.00810538 |
| stage_b_exp_leverage_liquidity_stress_peer_z           |       0.102623  | -0.0567187  |  30.9843 |   0.00468498 |
| stage_b_uninsured_funded_amount_usd_peer_z             |       0.100589  | -0.0458777  |  63.3028 |   0.00957171 |
| stage_b_funded_amount_usd_peer_z                       |       0.0926462 | -0.0402897  |  29.3836 |   0.00444294 |
| buyer_terms_uninsured_pressure                         |       0.0874894 | -0.0397969  |  95.3273 |   0.014414   |
| stage_b_total_exposure_tenor_to_revenue_days_peer_z    |       0.0825568 | -0.0273438  |  47.3317 |   0.0071568  |
| stage_b_uninsured_funded_to_revenue_est_peer_z         |       0.0779058 | -0.0348419  | 118.34   |   0.0178936  |
| exp_quick_ratio_shortfall                              |       0.0701795 | -0.0288051  |  62.434  |   0.00944034 |
| log_annual_revenue_usd_est                             |       0.0700303 |  0.00306206 |  48.4951 |   0.00733271 |
| buyer_terms_pressure                                   |       0.0679969 | -0.0315433  |  43.4259 |   0.00656621 |
| exp_dispute_to_history_pressure                        |       0.0629374 | -0.0235251  |  36.8621 |   0.00557374 |
| stage_b_exp_fx_loss_x_tenor_peer_z                     |       0.0590816 | -0.0331627  |  38.2686 |   0.00578641 |
| exp_iscore_risk                                        |       0.059046  | -0.0424541  |  34.4535 |   0.00520955 |
| exp_young_firm_risk                                    |       0.0561509 | -0.0219566  |  48.3762 |   0.00731473 |
| exp_debt_to_equity                                     |       0.0554681 | -0.0245341  |  83.6322 |   0.0126456  |
| buyer_risk_pressure                                    |       0.054726  | -0.0254415  |  50.7954 |   0.00768053 |
| exp_history_depth_risk                                 |       0.0517175 | -0.0407569  |  55.6837 |   0.00841966 |
| concentration_adjusted_total_exposure                  |       0.0492415 |  0.00246177 |  45.0138 |   0.00680632 |
| exp_low_history_flag                                   |       0.0490323 | -0.0377971  |  82.6657 |   0.0124995  |

## XGBoost Gain Feature Importance

| feature                                                          |      gain |   weight |   gain_share |
|:-----------------------------------------------------------------|----------:|---------:|-------------:|
| log_cumulative_outstanding_usd                                   | 248.337   |      116 |  0.0375499   |
| stage_b_log_cumulative_outstanding_usd_risk_percentile           | 230.413   |       40 |  0.0348396   |
| stage_b_liquidity_pressure_risk_percentile                       | 145.396   |        2 |  0.0219846   |
| buy_creditsafe_rating_score                                      | 129.673   |        1 |  0.0196071   |
| stage_b_exposure_quality_pressure_peer_z                         | 120.932   |       19 |  0.0182855   |
| stage_b_buyer_terms_uninsured_pressure_risk_percentile           | 119.597   |        5 |  0.0180836   |
| stage_b_uninsured_funded_to_revenue_est_peer_z                   | 118.34    |       16 |  0.0178936   |
| stage_b_total_exposure_tenor_to_revenue_days_risk_percentile     | 110.93    |        3 |  0.0167732   |
| buyer_terms_uninsured_pressure                                   |  95.3273  |       24 |  0.014414    |
| exp_debt_to_equity                                               |  83.6322  |       12 |  0.0126456   |
| exp_low_history_flag                                             |  82.6657  |        5 |  0.0124995   |
| buy_buyer_country_risk_score                                     |  82.5442  |        4 |  0.0124811   |
| stage_b_exp_low_history_flag_risk_percentile                     |  81.583   |        1 |  0.0123358   |
| stage_b_exposure_quality_pressure_risk_percentile                |  80.8649  |        5 |  0.0122272   |
| stage_b_buy_creditsafe_rating_score_risk_percentile              |  79.6658  |        1 |  0.0120459   |
| stage_b_exp_debt_to_equity_peer_z                                |  79.2273  |      100 |  0.0119796   |
| exp_fx_loss_x_non_usd                                            |  76.3442  |        2 |  0.0115436   |
| stage_b_buy_buyer_country_risk_score_risk_percentile             |  75.0866  |        2 |  0.0113535   |
| total_exposure_tenor_to_revenue_days                             |  73.0362  |        6 |  0.0110434   |
| exposure_quality_pressure                                        |  72.864   |       13 |  0.0110174   |
| stage_b_exp_debt_to_equity_risk_percentile                       |  71.6966  |        4 |  0.0108409   |
| stage_b_exp_on_time_gap_x_low_history_peer_z                     |  70.1136  |        7 |  0.0106015   |
| liquidity_pressure                                               |  68.5665  |        5 |  0.0103676   |
| stage_b_exp_current_ratio_shortfall_risk_percentile              |  65.724   |        2 |  0.0099378   |
| stage_b_exp_operating_margin_shortfall_risk_percentile           |  64.8002  |        1 |  0.00979812  |
| stage_b_uninsured_funded_amount_usd_peer_z                       |  63.3028  |       23 |  0.00957171  |
| exp_quick_ratio_shortfall                                        |  62.434   |       15 |  0.00944034  |
| stage_b_exp_quick_ratio_shortfall_peer_z                         |  59.6854  |        7 |  0.00902474  |
| exp_dscr_shortfall                                               |  59.3748  |       15 |  0.00897777  |
| stage_b_loss_given_default_proxy_risk_percentile                 |  59.0092  |        3 |  0.00892249  |
| total_exposure_to_revenue_est                                    |  56.9412  |        1 |  0.00860979  |
| exp_history_depth_risk                                           |  55.6837  |        8 |  0.00841966  |
| stage_b_velocity_exposure_pressure_peer_z                        |  55.1466  |        7 |  0.00833844  |
| stage_b_liquidity_pressure_peer_z                                |  55.0787  |        4 |  0.00832817  |
| stage_b_exp_iscore_risk_risk_percentile                          |  54.8392  |        3 |  0.00829197  |
| ticket_to_revenue_est                                            |  53.6052  |       23 |  0.00810538  |
| stage_b_buyer_risk_pressure_risk_percentile                      |  53.2161  |        7 |  0.00804655  |
| velocity_exposure_pressure                                       |  52.3392  |        3 |  0.00791396  |
| uninsured_funded_amount_usd                                      |  52.0852  |        5 |  0.00787554  |
| log_total_exposure_to_revenue_est                                |  51.7874  |        1 |  0.00783052  |
| stage_b_trade_history_pressure_peer_z                            |  51.1845  |        4 |  0.00773936  |
| exp_current_ratio_shortfall                                      |  51.1593  |        4 |  0.00773554  |
| buyer_risk_pressure                                              |  50.7954  |       32 |  0.00768053  |
| stage_b_exp_quick_ratio_shortfall_risk_percentile                |  50.3501  |        2 |  0.00761319  |
| buy_buyer_payment_history_dso_avg                                |  50.3466  |        4 |  0.00761267  |
| stage_b_benchmark_score                                          |  50.3401  |       49 |  0.00761168  |
| stage_b_concentration_adjusted_ticket_to_revenue_risk_percentile |  50.0444  |        5 |  0.00756697  |
| exp_banking_depth_risk                                           |  49.5647  |        5 |  0.00749444  |
| log_annual_revenue_usd_est                                       |  48.4951  |       16 |  0.00733271  |
| exp_young_firm_risk                                              |  48.3762  |       12 |  0.00731473  |
| stage_b_log_ticket_to_revenue_est_risk_percentile                |  47.8685  |        1 |  0.00723796  |
| stage_b_total_exposure_tenor_to_revenue_days_peer_z              |  47.3317  |       28 |  0.0071568   |
| stage_b_exp_dscr_shortfall_risk_percentile                       |  46.4468  |        2 |  0.00702299  |
| exp_concentration_top1_buyer_pct                                 |  46.4276  |        2 |  0.00702008  |
| stage_b_ticket_to_revenue_est_risk_percentile                    |  46.4086  |        2 |  0.00701721  |
| concentration_adjusted_total_exposure                            |  45.0138  |       19 |  0.00680632  |
| buyer_terms_pressure                                             |  43.4259  |       13 |  0.00656621  |
| stage_b_loss_given_default_proxy_peer_z                          |  43.0085  |        7 |  0.00650311  |
| advance_tenor_exposure                                           |  42.7412  |        9 |  0.00646268  |
| log_invoice_amount_usd                                           |  41.7694  |       22 |  0.00631574  |
| stage_b_log_annual_revenue_usd_est_risk_percentile               |  41.5681  |        5 |  0.00628531  |
| buyer_dso_to_terms                                               |  40.9149  |        6 |  0.00618654  |
| stage_b_exp_young_firm_risk_risk_percentile                      |  40.5911  |        2 |  0.00613759  |
| stage_b_trade_history_pressure_risk_percentile                   |  40.405   |        2 |  0.00610945  |
| country_risk_x_tenor                                             |  40.0044  |        2 |  0.00604887  |
| stage_b_exp_fx_loss_x_tenor_peer_z                               |  38.2686  |       18 |  0.00578641  |
| exp_fx_loss_x_tenor                                              |  38.1592  |        9 |  0.00576986  |
| stage_b_funded_tenor_usd_days_log_risk_percentile                |  38.113   |        9 |  0.00576288  |
| stage_b_exp_management_tenure_risk_risk_percentile               |  38.006   |        1 |  0.0057467   |
| stage_b_ticket_to_revenue_est_peer_z                             |  37.3019  |       11 |  0.00564023  |
| exp_dispute_to_history_pressure                                  |  36.8621  |       15 |  0.00557374  |
| stage_b_buyer_dso_gap_days_risk_percentile                       |  36.2734  |        1 |  0.00548472  |
| stage_b_exp_dispute_to_history_pressure_risk_percentile          |  36.2666  |        4 |  0.00548369  |
| funded_tenor_to_revenue_days                                     |  35.8857  |        1 |  0.0054261   |
| loss_given_default_proxy                                         |  35.8788  |        9 |  0.00542506  |
| exp_iscore_risk                                                  |  34.4535  |       31 |  0.00520955  |
| exp_operating_margin_shortfall                                   |  34.3902  |        1 |  0.00519997  |
| stage_b_buyer_dso_to_terms_risk_percentile                       |  34.3337  |        2 |  0.00519143  |
| uninsured_funded_to_revenue_est                                  |  34.0842  |        2 |  0.0051537   |
| log_outstanding_to_revenue_est                                   |  33.9993  |        3 |  0.00514087  |
| stage_b_exp_fx_loss_abs_pct_peer_z                               |  33.6917  |        8 |  0.00509435  |
| stage_b_exp_history_depth_risk_risk_percentile                   |  33.5567  |        5 |  0.00507395  |
| stage_b_exp_historical_dispute_rate_peer_z                       |  33.5288  |        5 |  0.00506973  |
| stage_b_profitability_pressure_risk_percentile                   |  33.2267  |        1 |  0.00502404  |
| stage_b_concentration_adjusted_total_exposure_peer_z             |  32.7089  |       55 |  0.00494575  |
| exp_sales_decline_magnitude_pct                                  |  32.6709  |        1 |  0.00494     |
| stage_b_concentration_adjusted_ticket_to_revenue_peer_z          |  32.4915  |       18 |  0.00491288  |
| stage_b_buyer_dso_to_terms_peer_z                                |  32.2969  |        3 |  0.00488346  |
| stage_b_funded_to_revenue_est_peer_z                             |  32.0731  |        3 |  0.00484961  |
| stage_b_buyer_terms_pressure_risk_percentile                     |  31.9028  |        5 |  0.00482387  |
| stage_b_buy_buyer_payment_history_dso_avg_peer_z                 |  31.4574  |       11 |  0.00475652  |
| stage_b_exp_leverage_liquidity_stress_peer_z                     |  30.9843  |       43 |  0.00468498  |
| buyer_dso_gap_days                                               |  30.9834  |        1 |  0.00468485  |
| stage_b_profitability_pressure_peer_z                            |  30.9823  |        1 |  0.00468468  |
| profitability_pressure                                           |  30.821   |        1 |  0.00466029  |
| exp_iscore_risk_squared                                          |  30.6921  |       10 |  0.0046408   |
| stage_b_uninsured_funded_to_revenue_est_risk_percentile          |  30.5925  |        1 |  0.00462574  |
| log_ticket_to_revenue_est                                        |  30.1621  |        6 |  0.00456066  |
| stage_b_exp_fx_loss_x_non_usd_peer_z                             |  29.5837  |        3 |  0.0044732   |
| stage_b_funded_amount_usd_peer_z                                 |  29.3836  |       26 |  0.00444294  |
| stage_b_exp_historical_on_time_gap_peer_z                        |  29.3378  |       10 |  0.00443603  |
| outstanding_to_revenue_est                                       |  29.0345  |        1 |  0.00439017  |
| stage_b_concentration_adjusted_total_exposure_risk_percentile    |  29.0128  |        2 |  0.00438688  |
| concentration_adjusted_ticket_to_revenue                         |  28.9144  |       16 |  0.00437201  |
| stage_b_advance_tenor_exposure_risk_percentile                   |  27.9653  |        8 |  0.00422849  |
| funded_amount_usd                                                |  27.7355  |        5 |  0.00419375  |
| exp_leverage_liquidity_stress                                    |  27.5079  |        9 |  0.00415934  |
| exp_on_time_gap_x_low_history                                    |  27.4006  |        8 |  0.0041431   |
| log_funded_amount_usd                                            |  27.2201  |        4 |  0.00411582  |
| stage_b_exp_operating_margin_shortfall_peer_z                    |  26.7     |        5 |  0.00403717  |
| stage_b_funded_tenor_to_revenue_days_peer_z                      |  26.6244  |        9 |  0.00402574  |
| stage_b_buyer_dso_gap_days_peer_z                                |  26.4748  |        1 |  0.00400312  |
| stage_b_exp_concentration_top1_buyer_pct_peer_z                  |  26.4549  |       17 |  0.00400012  |
| funded_tenor_usd_days_log                                        |  26.2128  |       22 |  0.0039635   |
| funded_to_revenue_est                                            |  25.1126  |        3 |  0.00379716  |
| stage_b_creditsafe_risk_x_tenor_peer_z                           |  25.0461  |        8 |  0.00378709  |
| stage_b_peer_outlier_score                                       |  24.565   |        3 |  0.00371435  |
| stage_b_log_invoice_amount_usd_risk_percentile                   |  24.5355  |        4 |  0.0037099   |
| stage_b_exp_on_time_gap_x_low_history_risk_percentile            |  23.5203  |        1 |  0.00355638  |
| exp_auditor_tier_risk_score                                      |  23.5082  |        2 |  0.00355457  |
| exp_gross_margin_shortfall                                       |  22.9585  |        1 |  0.00347144  |
| stage_b_buyer_terms_pressure_peer_z                              |  22.6848  |       14 |  0.00343006  |
| stage_b_exp_gross_margin_shortfall_risk_percentile               |  21.9799  |        1 |  0.00332347  |
| stage_b_advance_tenor_exposure_peer_z                            |  21.3504  |       11 |  0.00322829  |
| stage_b_exp_fx_loss_x_tenor_risk_percentile                      |  21.2093  |        5 |  0.00320696  |
| stage_b_payment_terms_to_buyer_dso_peer_z                        |  20.5256  |       11 |  0.00310358  |
| stage_b_exp_leverage_liquidity_stress_risk_percentile            |  20.3321  |        4 |  0.00307432  |
| stage_b_country_risk_x_tenor_peer_z                              |  19.6712  |       11 |  0.00297438  |
| exp_fx_loss_abs_pct                                              |  19.4596  |        3 |  0.00294239  |
| stage_b_exp_dscr_shortfall_peer_z                                |  18.9045  |       12 |  0.00285846  |
| trade_history_pressure                                           |  18.8792  |       10 |  0.00285462  |
| stage_b_exp_sales_decline_magnitude_pct_peer_z                   |  17.0624  |        3 |  0.00257992  |
| stage_b_log_funded_amount_usd_risk_percentile                    |  15.6957  |        1 |  0.00237326  |
| payment_terms_to_buyer_dso                                       |  15.2594  |        1 |  0.0023073   |
| stage_b_total_exposure_to_revenue_est_peer_z                     |  15.0938  |        3 |  0.00228226  |
| stage_b_exp_fx_loss_abs_pct_risk_percentile                      |  13.5362  |        1 |  0.00204674  |
| stage_b_exp_dispute_to_history_pressure_peer_z                   |  12.6477  |        7 |  0.0019124   |
| stage_b_outstanding_to_revenue_est_peer_z                        |  12.6061  |        2 |  0.00190611  |
| exp_historical_dispute_rate                                      |  11.443   |        3 |  0.00173024  |
| stage_b_exp_current_ratio_shortfall_peer_z                       |   7.72658 |        8 |  0.0011683   |
| stage_b_peer_outlier_count                                       |   7.55114 |        5 |  0.00114177  |
| stage_b_peer_high_risk_outlier_count                             |   7.48049 |        2 |  0.00113109  |
| creditsafe_risk_x_tenor                                          |   6.1137  |        7 |  0.000924422 |
| exp_management_tenure_risk                                       |   4.31791 |        1 |  0.00065289  |
| stage_b_recent_velocity_to_history_peer_z                        |   2.31953 |        5 |  0.000350725 |
| stage_b_creditsafe_risk_x_tenor_risk_percentile                  |   1.30541 |        1 |  0.000197385 |
| log_funded_to_revenue_est                                        |   0       |        0 |  0           |
| exp_sales_decline_flag                                           |   0       |        0 |  0           |
| exp_negative_operating_margin_flag                               |   0       |        0 |  0           |
| exp_historical_on_time_gap                                       |   0       |        0 |  0           |
| recent_velocity_to_history                                       |   0       |        0 |  0           |
| log_uninsured_funded_amount_usd                                  |   0       |        0 |  0           |
| stage_b_exp_gross_margin_shortfall_peer_z                        |   0       |        0 |  0           |
| country_risk_x_uninsured                                         |   0       |        0 |  0           |
| insurance_not_covered_flag                                       |   0       |        0 |  0           |
| stage_b_recent_velocity_to_history_risk_percentile               |   0       |        0 |  0           |
| country_terms_uninsured_pressure                                 |   0       |        0 |  0           |
| stage_b_funded_amount_usd_risk_percentile                        |   0       |        0 |  0           |
| exp_segment_sme_flag                                             |   0       |        0 |  0           |
| stage_b_funded_tenor_to_revenue_days_risk_percentile             |   0       |        0 |  0           |
| stage_b_outstanding_to_revenue_est_risk_percentile               |   0       |        0 |  0           |
| stage_b_log_total_exposure_to_revenue_est_risk_percentile        |   0       |        0 |  0           |
| stage_b_total_exposure_to_revenue_est_risk_percentile            |   0       |        0 |  0           |
| stage_b_log_funded_to_revenue_est_risk_percentile                |   0       |        0 |  0           |
| stage_b_funded_to_revenue_est_risk_percentile                    |   0       |        0 |  0           |
| stage_b_log_outstanding_to_revenue_est_risk_percentile           |   0       |        0 |  0           |
| stage_b_exp_segment_sme_flag_risk_percentile                     |   0       |        0 |  0           |
| stage_b_insurance_not_covered_flag_risk_percentile               |   0       |        0 |  0           |
| stage_b_exp_negative_operating_margin_flag_risk_percentile       |   0       |        0 |  0           |
| stage_b_country_risk_x_uninsured_risk_percentile                 |   0       |        0 |  0           |
| stage_b_exp_sales_decline_flag_risk_percentile                   |   0       |        0 |  0           |
| stage_b_exp_historical_on_time_gap_risk_percentile               |   0       |        0 |  0           |
| stage_b_buy_buyer_payment_history_dso_avg_risk_percentile        |   0       |        0 |  0           |
| stage_b_exp_historical_dispute_rate_risk_percentile              |   0       |        0 |  0           |
| stage_b_uninsured_funded_amount_usd_risk_percentile              |   0       |        0 |  0           |
| stage_b_log_uninsured_funded_amount_usd_risk_percentile          |   0       |        0 |  0           |
| exp_uses_offshore_tradeco_flag                                   |   0       |        0 |  0           |
| exp_audit_qualified_flag                                         |   0       |        0 |  0           |
| stage_b_exp_uses_offshore_tradeco_flag_risk_percentile           |   0       |        0 |  0           |
| stage_b_exp_audit_qualified_flag_risk_percentile                 |   0       |        0 |  0           |
| stage_b_exp_auditor_tier_risk_score_risk_percentile              |   0       |        0 |  0           |
| stage_b_exp_concentration_top1_buyer_pct_risk_percentile         |   0       |        0 |  0           |
| stage_b_velocity_exposure_pressure_risk_percentile               |   0       |        0 |  0           |
| exp_mgmt_succession_gap_flag                                     |   0       |        0 |  0           |
| stage_b_exp_fx_loss_x_non_usd_risk_percentile                    |   0       |        0 |  0           |
| stage_b_non_usd_invoice_flag_risk_percentile                     |   0       |        0 |  0           |
| stage_b_exp_sales_decline_magnitude_pct_risk_percentile          |   0       |        0 |  0           |
| non_usd_invoice_flag                                             |   0       |        0 |  0           |
| stage_b_exp_mgmt_succession_gap_flag_risk_percentile             |   0       |        0 |  0           |
| stage_b_exp_iscore_risk_squared_risk_percentile                  |   0       |        0 |  0           |
| stage_b_country_terms_uninsured_pressure_risk_percentile         |   0       |        0 |  0           |
| stage_b_country_risk_x_tenor_risk_percentile                     |   0       |        0 |  0           |
| stage_b_payment_terms_to_buyer_dso_risk_percentile               |   0       |        0 |  0           |
| stage_b_exp_banking_depth_risk_risk_percentile                   |   0       |        0 |  0           |

## Generated Artifacts

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
- `reports/stage_c_shap_feature_importance.csv`
- `reports/stage_c_shap_feature_importance.png`
- `reports/stage_c_validation_shap_explanations.csv`
- `models/stage_c_xgboost_model.joblib`