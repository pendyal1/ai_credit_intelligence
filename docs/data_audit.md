# Step 1 Data Audit

Dataset path: `utils/rawassi_synthetic_dataset.csv`

## File Shape

- Rows: 5,000
- Columns: 50
- Unique exporters: 300
- Unique buyers: 500
- Duplicate transaction IDs: 0
- Invoice date range: 2023-03-14 to 2026-03-31

## Outcome Distribution

- Paid: 4,424
- Defaulted: 232
- Open / unresolved: 344
- Resolved rows: 4,656
- Resolved default rate: 4.98%

## Out-Of-Time Split

The data dictionary recommends training only on resolved deals and splitting by decision date.

- Train: `invoice_date < 2025-10-01`
  - Rows: 4,083
  - Default rate: 5.19%
- Validation: `invoice_date >= 2025-10-01`
  - Rows: 573
  - Default rate: 3.49%
- Open rows are held out for sanity scoring only.

## Leakage Contract

The model feature set must exclude fields that would not be known at decision time.

Exclude:

- `defaulted`
- `status`
- `actual_resolution_date`
- `days_to_resolution`
- `buy_buyer_baseline_default_prob`

Treat carefully:

- `expected_payment_date` is known at decision time but redundant with `invoice_date + payment_terms_days`; prefer dropping it from the model feature set.

Sanity check:

- `days_to_resolution` alone has ROC-AUC around 0.978 on resolved rows, confirming it is severe leakage.
- `actual_resolution_date` and `days_to_resolution` are populated even for rows marked `open`, so open rows must not be used for model validation.

## Observations For Next Step

- The dataset aligns with the case brief on row count, entity counts, date range, open rows, and roughly 5% default rate.
- The PDF says 47 columns, but the CSV has 50 columns. The extra fields appear consistent with the documented outcome and metadata fields, so this is not blocking.
- Boolean columns are already parseable as true/false values.
- Only `defaulted` has missing values, corresponding to open transactions.

## Step 2 Target

Build the first reusable project layer:

- a data loading module,
- a feature/leakage contract,
- a configurable hard-filter config,
- and a small script that reports train/validation/open split health.
