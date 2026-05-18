# AI Credit Intelligence

Rawasi case-study implementation for an interpretable SMB invoice discounting credit intelligence system.

## Current Status

- Step 1 data audit: `docs/data_audit.md`
- EDA notebook: `notebooks/01_data_understanding_eda.ipynb`
- Layer 1 / Stage A hard filters: `config/hard_filters.json`, `src/rawasi_credit/filters.py`

## Run Stage A Hard Filters

```bash
python scripts/run_stage_a_filters.py
```

Generated reports are written to `reports/`.
