# Stage A Hard Filter Report

Dataset: `utils/rawassi_synthetic_dataset.csv`
Config: `config/hard_filters.json`

## Purpose

Stage A removes deals that fail non-negotiable credit policy or data-quality criteria before benchmark scoring or statistical modeling.

## Overall Decision Summary

| stage_a_decision   |   rows |   row_pct |   defaults |   default_rate |
|:-------------------|-------:|----------:|-----------:|---------------:|
| PASS_STAGE_A       |   2863 |    0.5726 |        102 |      0.0383747 |
| REJECT_STAGE_A     |   2137 |    0.4274 |        130 |      0.0650651 |

## Train Decision Summary

| stage_a_decision   |   rows |   row_pct |   defaults |   default_rate |
|:-------------------|-------:|----------:|-----------:|---------------:|
| PASS_STAGE_A       |   2335 |  0.571883 |         92 |      0.0394004 |
| REJECT_STAGE_A     |   1748 |  0.428117 |        120 |      0.0686499 |

## Validation Decision Summary

| stage_a_decision   |   rows |   row_pct |   defaults |   default_rate |
|:-------------------|-------:|----------:|-----------:|---------------:|
| PASS_STAGE_A       |    323 |    0.5637 |         10 |      0.0309598 |
| REJECT_STAGE_A     |    250 |    0.4363 |         10 |      0.04      |

## Open Holdout Decision Summary

| stage_a_decision   |   rows |   row_pct |
|:-------------------|-------:|----------:|
| PASS_STAGE_A       |    205 |   0.59593 |
| REJECT_STAGE_A     |    139 |   0.40407 |

## Rule Hit Summary

| rule                       | description                                                                  | reject_reason                                   |   hit_rows |   hit_rate |   resolved_hit_rows |   defaults_captured |   default_rate_if_hit |
|:---------------------------|:-----------------------------------------------------------------------------|:------------------------------------------------|-----------:|-----------:|--------------------:|--------------------:|----------------------:|
| acceptable_auditor_tier    | Reject exporters whose auditor tier is below the accepted quality bar.       | Auditor tier is not accepted                    |        973 |     0.1946 |                 919 |                  52 |             0.0565832 |
| buyer_country_risk_ceiling | Reject deals where buyer country risk is above the committee ceiling.        | Buyer country risk score above 8                |        622 |     0.1244 |                 578 |                  35 |             0.0605536 |
| missing_mandatory_docs     | Reject when required onboarding and transaction documentation is incomplete. | Missing mandatory documentation                 |        372 |     0.0744 |                 345 |                  19 |             0.0550725 |
| historical_dispute_ceiling | Reject exporters with extreme prior dispute rates.                           | Exporter historical dispute rate at or above 8% |        335 |     0.067  |                 311 |                  41 |             0.131833  |
| minimum_iscore             | Reject exporters below the minimum acceptable Egyptian credit bureau score.  | Exporter i-score below 550                      |        186 |     0.0372 |                 176 |                   5 |             0.0284091 |
| creditsafe_floor           | Reject buyers with ratings considered default-imminent.                      | Buyer CreditSafe rating is E                    |         72 |     0.0144 |                  65 |                   4 |             0.0615385 |

## Notes

- All thresholds and accepted values live in `config/hard_filters.json`.
- The filter layer only uses fields known at the financing decision date.
- `status`, `defaulted`, `actual_resolution_date`, `days_to_resolution`, and buyer baseline default probability are not used by Stage A.