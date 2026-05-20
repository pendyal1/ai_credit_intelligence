# Stage A Hard Filter Report

Dataset: `utils/rawassi_synthetic_dataset.csv`
Config: `config/hard_filters.json`

## Purpose

Stage A removes deals that fail non-negotiable credit policy or data-quality criteria before benchmark scoring or statistical modeling.

## Overall Decision Summary

| stage_a_decision   |   rows |   row_pct |   defaults |   default_rate |
|:-------------------|-------:|----------:|-----------:|---------------:|
| PASS_STAGE_A       |   3732 |    0.7464 |        144 |      0.0413912 |
| REJECT_STAGE_A     |   1268 |    0.2536 |         88 |      0.0747664 |

## Train Decision Summary

| stage_a_decision   |   rows |   row_pct |   defaults |   default_rate |
|:-------------------|-------:|----------:|-----------:|---------------:|
| PASS_STAGE_A       |   3048 |   0.74651 |        130 |      0.0426509 |
| REJECT_STAGE_A     |   1035 |   0.25349 |         82 |      0.0792271 |

## Validation Decision Summary

| stage_a_decision   |   rows |   row_pct |   defaults |   default_rate |
|:-------------------|-------:|----------:|-----------:|---------------:|
| PASS_STAGE_A       |    431 |  0.752182 |         14 |      0.0324826 |
| REJECT_STAGE_A     |    142 |  0.247818 |          6 |      0.0422535 |

## Open Holdout Decision Summary

| stage_a_decision   |   rows |   row_pct |
|:-------------------|-------:|----------:|
| PASS_STAGE_A       |    253 |  0.735465 |
| REJECT_STAGE_A     |     91 |  0.264535 |

## Rule Hit Summary

| rule                       | description                                                                  | reject_reason                                   |   hit_rows |   hit_rate |   resolved_hit_rows |   defaults_captured |   default_rate_if_hit |
|:---------------------------|:-----------------------------------------------------------------------------|:------------------------------------------------|-----------:|-----------:|--------------------:|--------------------:|----------------------:|
| buyer_country_risk_ceiling | Reject deals where buyer country risk is above the committee ceiling.        | Buyer country risk score above 8                |        622 |     0.1244 |                 578 |                  35 |             0.0605536 |
| missing_mandatory_docs     | Reject when required onboarding and transaction documentation is incomplete. | Missing mandatory documentation                 |        372 |     0.0744 |                 345 |                  19 |             0.0550725 |
| historical_dispute_ceiling | Reject exporters with extreme prior dispute rates.                           | Exporter historical dispute rate at or above 8% |        335 |     0.067  |                 311 |                  41 |             0.131833  |
| creditsafe_floor           | Reject buyers with ratings considered default-imminent.                      | Buyer CreditSafe rating is E                    |         72 |     0.0144 |                  65 |                   4 |             0.0615385 |

## Notes

- All thresholds and accepted values live in `config/hard_filters.json`.
- The filter layer only uses fields known at the financing decision date.
- `status`, `defaulted`, `actual_resolution_date`, `days_to_resolution`, and buyer baseline default probability are not used by Stage A.