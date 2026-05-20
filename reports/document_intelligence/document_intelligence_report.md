# Bonus Task 4 - Document Intelligence Option A

## Prototype

The prototype renders each PDF page with ImageMagick and runs local Tesseract OCR. It then uses label-anchored extraction for the three requested 2024 financial values.

- Tesseract: `tesseract 5.5.0`
- Available Tesseract languages: `afr, amh, ara, asm, aze, aze_cyrl, bel, ben, bod, bos, bre, bul, cat, ceb, ces, chi_sim, chi_sim_vert, chi_tra, chi_tra_vert, chr, cos, cym, dan, deu, div, dzo, ell, eng, enm, epo, equ, est, eus, fao, fas, fil, fin, fra, frk, frm, fry, gla, gle, glg, grc, guj, hat, heb, hin, hrv, hun, hye, iku, ind, isl, ita, ita_old, jav, jpn, jpn_vert, kan, kat, kat_old, kaz, khm, kir, kmr, kor, kor_vert, lao, lat, lav, lit, ltz, mal, mar, mkd, mlt, mon, mri, msa, mya, nep, nld, nor, oci, ori, osd, pan, pol, por, pus, que, ron, rus, san, script/Arabic, script/Armenian, script/Bengali, script/Canadian_Aboriginal, script/Cherokee, script/Cyrillic, script/Devanagari, script/Ethiopic, script/Fraktur, script/Georgian, script/Greek, script/Gujarati, script/Gurmukhi, script/HanS, script/HanS_vert, script/HanT, script/HanT_vert, script/Hangul, script/Hangul_vert, script/Hebrew, script/Japanese, script/Japanese_vert, script/Kannada, script/Khmer, script/Lao, script/Latin, script/Malayalam, script/Myanmar, script/Oriya, script/Sinhala, script/Syriac, script/Tamil, script/Telugu, script/Thaana, script/Thai, script/Tibetan, script/Vietnamese, sin, slk, slv, snd, snum, spa, spa_old, sqi, srp, srp_latn, sun, swa, swe, syr, tam, tat, tel, tgk, tha, tir, ton, tur, uig, ukr, urd, uzb, uzb_cyrl, vie, yid, yor`
- Arabic language pack available: `True`
- English pages: `12`
- Arabic pages: `12`

## Extraction Results

| document   | display_name       |   extracted_thousands_egp |   ground_truth_thousands_egp |   absolute_error_thousands_egp | match   | source_line                              |
|:-----------|:-------------------|--------------------------:|-----------------------------:|-------------------------------:|:--------|:-----------------------------------------|
| english    | Total Revenue 2024 |                    487350 |                       487350 |                              0 | True    | Total Revenue 487,350 412,890            |
| english    | Net Income 2024    |                     64530 |                        64530 |                              0 | True    | Net Income (thousands EGP) 64,530 53,860 |
| english    | Total Assets 2024  |                    721350 |                       721350 |                              0 | True    | Total Assets 721,350 634,130             |
| arabic     | Total Revenue 2024 |                    487350 |                       487350 |                              0 | True    | إجمالي الإيرادات ‎E,8: EAV,F0-‏            |
| arabic     | Net Income 2024    |                     64530 |                        64530 |                              0 | True    | صافي الدخل ‎OF,81: 1€,0F«‏                 |
| arabic     | Total Assets 2024  |                    721350 |                       721350 |                              0 | True    | إجمالي الأصول - ‎1FE,1r- VY1,¥30‏          |

The English sample extracts all three target values exactly in thousands of EGP. The Arabic extraction depends on OCR-confusable numeric normalization because local Tesseract still distorts several table digits.

## Arabic Failure Mode

The local Tesseract install has Arabic traineddata. The prototype uses `script/Arabic` when available and applies Arabic-Indic digit plus OCR-confusable numeric normalization.

Observed failure modes:

- Arabic labels are now mostly readable with Arabic traineddata, but table spacing and right-to-left ordering are still unstable.
- Numeric strings are partly preserved but often distorted, for example `487,350` can appear as noisy OCR such as `EAV,F0-`.
- The report has mixed table layout and right-to-left text, so line order is unreliable without an Arabic-aware OCR/layout model.

## API Comparison

| Approach | What was tested / assumed | Accuracy on this sample | Cost estimate | Operational notes |
| --- | --- | --- | --- | --- |
| Local Tesseract OCR | Actually run locally; Arabic uses `script/Arabic` when installed plus numeric normalization | English: 3/3 exact. Arabic can be recovered on this sample, but confidence is lower because values require OCR-confusable cleanup | Software cost $0; compute/ops only | Good for a prototype and regression baseline. Needs preprocessing, table parsing, and QA gates before production. |
| Google Document AI Enterprise OCR | Public OCR processor pricing; not called in this prototype | Expected better Arabic/layout handling than local English-only OCR, but still needs field-level validation | $1.50 / 1,000 pages, about $0.018 per 12-page document | Strong pilot candidate because it is managed, page-based, and supports document OCR workflows. |
| AWS Textract Detect Document Text | Public text-detection pricing; not called in this prototype | Expected better scan handling than local OCR, but financial-table key-value extraction may need custom parsing | $0.0015/page for first 1M pages, about $0.018 per 12-page document | Good managed fallback; evaluate Arabic/table quality before committing. |

Pricing references: Google Document AI pricing: https://cloud.google.com/document-ai/pricing. AWS Textract pricing: https://aws.amazon.com/textract/pricing/.

## Recommendation

For the pilot stage under 50 documents/month, use Google Document AI OCR or AWS Textract as the primary OCR engine, with this Tesseract pipeline as a cheap regression baseline. At 12 pages per document, OCR-only API cost is roughly `$0.90` per month for 50 documents, so engineering reliability matters more than OCR unit cost.

At 500 documents/month, the same OCR-only page price is still only about `$9.00` per month. I would move to a managed OCR API with a validation layer rather than scaling local OCR manually. The validation layer should check accounting identities, compare extracted values to expected statement sections, require confidence thresholds, and route exceptions to human review.

## Generated Artifacts

- `reports/document_intelligence/extraction_results.csv`
- `reports/document_intelligence/extraction_results.json`
- `reports/document_intelligence/english_ocr_text.txt`
- `reports/document_intelligence/arabic_ocr_text.txt`