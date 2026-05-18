# AI Credit Intelligence

Rawasi case-study implementation for an interpretable SMB invoice discounting credit intelligence system.

## Current Status

- Step 1 data audit: `docs/data_audit.md`
- EDA notebook: `notebooks/01_data_understanding_eda.ipynb`
- Layer 1 / Stage A hard filters: `config/hard_filters.json`, `src/rawasi_credit/filters.py`
- Layer 1 / Stage B benchmarks: `config/benchmark_features.json`, `src/rawasi_credit/benchmarks.py`

## Run Stage A Hard Filters

```bash
python scripts/run_stage_a_filters.py
```

Generated reports are written to `reports/`.

## Run Stage B Benchmarks

```bash
python scripts/run_stage_b_benchmarks.py
```

Stage B fits benchmark statistics on Stage-A-passed training rows and writes generated reports to `reports/`.
