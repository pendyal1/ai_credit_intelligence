# AI Credit Intelligence

Rawasi case-study implementation for an interpretable SMB invoice discounting credit intelligence system.

## Architecture

```text
Raw transaction data
        |
        v
Layer 1A: JSON-configurable hard-filter rule engine
operators: equals / in / greater_than / greater_equal; config: config/hard_filters.json
gates: docs, buyer country risk, CreditSafe rating, dispute history
        |
        v
Layer 1B: StageBBenchmarkTransformer
sector-agnostic ratios + robust peer benchmarking by goods_category
features: exposure/revenue, tenor exposure, buyer terms, liquidity, profitability, trade history
outliers: median/MAD peer z-scores with global fallback for sparse or new sectors
        |
        v
Layer 1C: XGBoost default classifier
model: xgboost.XGBClassifier(binary:logistic)
tuning: GridSearchCV with TimeSeriesSplit inside training window
imbalance: scale_pos_weight; threshold: recall-first 0.40
outputs: default probability, confusion matrix, AUC-PR, KS, calibration, SHAP reasons
        |
        v
Layer 2: Intelligence layer
trend clusters: recent default-rate lift vs training baseline
drift: Population Stability Index
anomalies: percentile-based transaction anomaly score
LLM: OpenRouter chat-completion prompt for reviewer summary
        |
        v
Layer 3: Model evolution
blind-spot audit, retraining triggers, cold-start design, monitoring governance
        |
        v
Bonus: Document intelligence
OCR: ImageMagick page rendering + Tesseract/script-Arabic
extraction: label-anchored parsing + Arabic OCR numeric normalization
API comparison: Google Document AI, AWS Textract, Azure Document Intelligence
```

Main notebooks:

- `notebooks/01_data_understanding_eda.ipynb` - Layer 1 multi-stage filtering pipeline.
- `notebooks/02_layer2_intelligence.ipynb` - Layer 2 trend/anomaly detection and OpenRouter LLM summary.
- `notebooks/03_layer3_blind_spot_analysis.ipynb` - Layer 3 blind-spot analysis and evolution design.
- `notebooks/04_document_intelligence_bonus.ipynb` - Bonus document OCR extraction and API comparison.

Key scripts:

```bash
python scripts/run_stage_a_filters.py
python scripts/run_stage_b_benchmarks.py
python scripts/run_stage_c_model.py
python scripts/run_layer2_intelligence.py
python scripts/score_one_transaction.py
python scripts/run_document_intelligence.py
```

Layer 2 Task 2B uses OpenRouter if a key is available:

```bash
export OPENROUTER_API_KEY="your_key_here"
export OPENROUTER_MODEL="openai/gpt-4o-mini"
```

## Three Key Decisions

1. **Keep hard filters narrow and explainable.**
   Stage A is reserved for non-negotiable policy and data-quality gates. Variables such as exporter quality, leverage, and liquidity are better used as model features than as automatic reject rules, because rigid cutoffs create high false-positive volume.

2. **Use sector-agnostic derived features before modeling.**
   The most useful Stage B features normalize risk across sectors: invoice-to-revenue, outstanding-to-revenue, total exposure-to-revenue, tenor exposure, uninsured exposure, buyer DSO versus terms, liquidity pressure, profitability pressure, and peer outlier scores. This makes a $500k invoice comparable across small and large exporters.

3. **Optimize Stage C for catching defaults, not maximizing precision.**
   Rawasi’s business risk is asymmetric: missing a default is worse than sending a borderline deal to review. The XGBoost threshold is therefore recall-first and evaluated out of time using AUC-PR, recall, confusion matrix, calibration, threshold analysis, and SHAP explanations.

## Current Results

Layer 1 full validation flags 270 of 573 validation deals for review, catching 19 of 20 defaults. That is 95% recall with low precision. Behaves like conservative credit-screening system.

Stage C also produces global SHAP importance and per-deal SHAP reason fields, so model interpretation is available alongside Stage B benchmark reasons and Layer 2 anomaly evidence.

Layer 2 adds portfolio intelligence:

- 15 recent underperforming cluster alerts.
- 5 significant PSI shifts, mainly in exposure-pressure features.
- 220 transaction anomaly alerts across validation and open-holdout deals.
- Structured OpenRouter prompt for a reviewer-ready deal summary with verdict, top factors, unusual flags, and recommended actions.

Layer 3 identifies four blind spots: exposure overextension, blunt hard-filter over-rejection, emerging cluster drift, and medium-score local anomalies that can still default.

Bonus Document Intelligence extracts the English and Arabic sample annual report values exactly after installing Arabic Tesseract language data: revenue EGP 487.35m, net income EGP 64.53m, and total assets EGP 721.35m. The Arabic result requires OCR-confusable numeric normalization, which is documented in `reports/document_intelligence/document_intelligence_report.md` with API cost comparisons.

## What I Would Build Next

With one additional week, I would focus on production readiness and model governance:

- Add a reviewer UI exposing SHAP drivers, benchmark reasons, anomaly flags, and LLM narrative in one deal view.
- Add champion/challenger model tracking and automated drift alerts.
- Add external macro and country-risk features such as FX moves, inflation, interest-rate shocks, and country downgrade events.

## Limitations

- The dataset is synthetic, so model performance should be treated as directional rather than production-grade.
- Default labels are sparse and imbalanced; precision is low because the system is intentionally recall-first.
- Some Layer 2 cluster alerts have small recent sample sizes, so they should trigger human investigation rather than automatic rejection.
- LLM summaries depend on prompt quality and should not override policy or model evidence without human review.

## Time Spent

Around 6 hours.
