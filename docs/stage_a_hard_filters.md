# Stage A - Hard Filters

Stage A is the automatic-reject layer of the Rawasi credit intelligence pipeline.
It runs before peer benchmarking and before the statistical credit model.

## Design Goal

The model should not score every deal. Some applications fail basic policy, documentation, or counterparty-risk requirements and should be removed before any probability model is used.

The implementation is intentionally configurable:

- rule thresholds and accepted values live in `config/hard_filters.json`
- credit committee changes do not require Python code edits
- the Python layer only interprets the configured rules and appends decision fields

## Default Automatic-Reject Rules

| Rule | Default policy | Reason |
| --- | --- | --- |
| Missing mandatory documentation | `docs_complete == false` | Missing docs are a fraud, enforceability, and operational-risk issue. |
| Minimum i-score | `exp_iscore < 550` | Severe bureau weakness should not pass automatically into model scoring. |
| Acceptable auditor tier | auditor tier not in `Big4`, `International`, `CBE-approved` | Audit quality is a trust and financial-statement reliability control. |
| Buyer country risk ceiling | `buy_buyer_country_risk_score > 8` | Very high sovereign/currency/enforcement risk can overwhelm deal-level mitigants. |
| CreditSafe floor | `buy_creditsafe_rating == E` | The data dictionary describes E as essentially default-imminent. |
| Historical dispute ceiling | `exp_historical_dispute_rate >= 0.08` | Extreme dispute history directly undermines invoice collectability. |

## Output Columns

`apply_hard_filters` appends:

- `stage_a_rule_<rule_name>` for each enabled rule
- `stage_a_rejected`
- `stage_a_decision`
- `stage_a_reject_reasons`
- `stage_a_reject_rule_count`

## Run

```bash
python scripts/run_stage_a_filters.py
```

Generated files:

- `reports/stage_a_report.md`
- `reports/stage_a_rule_summary.csv`
- `reports/stage_a_scored_transactions.csv`

## Leakage Position

Stage A must only use decision-time fields. It does not use:

- `defaulted`
- `status`
- `actual_resolution_date`
- `days_to_resolution`
- `buy_buyer_baseline_default_prob`
