# Stage B - Sector-Agnostic Benchmarks

Stage B runs after Stage A hard filters. It does not train a default model. Its job is to convert raw transaction fields into interpretable benchmark features and peer outlier diagnostics.

## Features

This implementation intentionally uses only three cross-sector features:

| Feature | Meaning | Why cross-sector |
| --- | --- | --- |
| `ticket_to_revenue_est` | Invoice size relative to exporter revenue | A large ticket is only risky relative to exporter scale. |
| `outstanding_to_revenue_est` | Existing Rawasi exposure relative to exporter revenue | Measures concentration and overextension across sectors. |
| `advance_tenor_exposure` | `advance_pct * payment_terms_days` | Higher advance and longer tenor increase capital-at-risk. |

Exporter annual revenue is provided in EGP. The estimated USD denominator uses the configurable assumption in `config/benchmark_features.json`:

```text
annual_revenue_usd_est = exp_annual_revenue_egp_millions * 1,000,000 / egp_per_usd_assumption
```

## Benchmark Scores

For each feature, Stage B fits a training-only percentile distribution on deals that:

1. are resolved training rows, and
2. pass Stage A.

Each transformed deal receives:

- `stage_b_ticket_to_revenue_est_risk_percentile`
- `stage_b_outstanding_to_revenue_est_risk_percentile`
- `stage_b_advance_tenor_exposure_risk_percentile`
- `stage_b_benchmark_score`
- `stage_b_benchmark_band`

The benchmark score is an equal-weight average of the three risk percentiles.

## Peer Outliers

Peer outlier detection uses the same three features and compares each deal to training peers in the same `goods_category`.

For each feature:

```text
robust_z = 0.6745 * (value - peer_median) / peer_MAD
```

Defaults:

- peer group: `goods_category`
- minimum peer group size: `50`
- outlier threshold: `abs(robust_z) >= 2.5`

If a peer group has fewer than 50 training rows, Stage B falls back to global training statistics and flags the source as `global_fallback`.

## Run

```bash
python scripts/run_stage_b_benchmarks.py
```

Generated files:

- `reports/stage_b_report.md`
- `reports/stage_b_scored_transactions.csv`
- `reports/stage_b_feature_summary.csv`
- `reports/stage_b_peer_outliers.csv`
- `reports/stage_b_training_stats.json`
