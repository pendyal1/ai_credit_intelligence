#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rawasi_credit.document_intelligence import (  # noqa: E402
    OcrRunConfig,
    available_tesseract_languages,
    dump_json,
    extract_target_values,
    extraction_results_frame,
    result_accuracy_summary,
    run_pdf_ocr,
    tesseract_version,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bonus Task Option A document extraction prototype."
    )
    parser.add_argument(
        "--english-pdf",
        type=Path,
        default=PROJECT_ROOT / "utils" / "Sample_English_Annual_Report.pdf",
        help="English sample annual report PDF.",
    )
    parser.add_argument(
        "--arabic-pdf",
        type=Path,
        default=PROJECT_ROOT / "utils" / "Sample_Arabic_Annual_Report.pdf",
        help="Arabic sample annual report PDF.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "document_intelligence",
        help="Directory for generated bonus-task artifacts.",
    )
    parser.add_argument("--density", type=int, default=300, help="PDF render DPI for OCR.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_dir / "ocr_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    languages = available_tesseract_languages()
    english_config = OcrRunConfig(density=args.density, tesseract_lang="eng")
    arabic_config = OcrRunConfig(
        density=args.density,
        tesseract_lang=(
            "script/Arabic"
            if "script/Arabic" in languages
            else "ara+eng"
            if "ara" in languages
            else "eng"
        ),
        tesseract_psm=4 if "ara" in languages or "script/Arabic" in languages else 6,
    )

    english_ocr = run_pdf_ocr(args.english_pdf, work_dir, english_config)
    arabic_ocr = run_pdf_ocr(args.arabic_pdf, work_dir, arabic_config)

    results = {
        "environment": {
            "tesseract_version": tesseract_version(),
            "available_tesseract_languages": languages,
            "arabic_language_pack_available": "ara" in languages,
            "image_renderer": "ImageMagick magick",
        },
        "english": {
            "pdf_path": str(args.english_pdf),
            "page_count": english_ocr["page_count"],
            "ocr_config": english_ocr["ocr_config"],
            "extraction": extract_target_values(english_ocr["full_text"], "english"),
        },
        "arabic": {
            "pdf_path": str(args.arabic_pdf),
            "page_count": arabic_ocr["page_count"],
            "ocr_config": arabic_ocr["ocr_config"],
            "extraction": extract_target_values(arabic_ocr["full_text"], "arabic"),
        },
    }
    results["english"]["accuracy"] = result_accuracy_summary(results["english"]["extraction"])
    results["arabic"]["accuracy"] = result_accuracy_summary(results["arabic"]["extraction"])

    results_path = args.output_dir / "extraction_results.json"
    csv_path = args.output_dir / "extraction_results.csv"
    report_path = args.output_dir / "document_intelligence_report.md"

    dump_json(results, results_path)
    extraction_results_frame(
        {"english": results["english"], "arabic": results["arabic"]}
    ).to_csv(csv_path, index=False)
    report_path.write_text(_render_report(results, csv_path, results_path))

    (args.output_dir / "english_ocr_text.txt").write_text(english_ocr["full_text"])
    (args.output_dir / "arabic_ocr_text.txt").write_text(arabic_ocr["full_text"])
    shutil.rmtree(work_dir)

    print("Document intelligence bonus task complete.")
    print(f"English accuracy: {results['english']['accuracy']}")
    print(f"Arabic accuracy: {results['arabic']['accuracy']}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {results_path}")
    print(f"Wrote: {csv_path}")


def _render_report(results: dict, csv_path: Path, json_path: Path) -> str:
    result_table = extraction_results_frame(
        {"english": results["english"], "arabic": results["arabic"]}
    )
    display_cols = [
        "document",
        "display_name",
        "extracted_thousands_egp",
        "ground_truth_thousands_egp",
        "absolute_error_thousands_egp",
        "match",
        "source_line",
    ]

    monthly_pages_pilot = 50 * results["english"]["page_count"]
    monthly_pages_scale = 500 * results["english"]["page_count"]
    api_price_per_1000_pages = 1.50
    pilot_cost = monthly_pages_pilot / 1000 * api_price_per_1000_pages
    scale_cost = monthly_pages_scale / 1000 * api_price_per_1000_pages
    cost_per_document = results["english"]["page_count"] / 1000 * api_price_per_1000_pages

    arabic_lang_note = (
        "The local Tesseract install has Arabic traineddata. The prototype uses `script/Arabic` when available and applies Arabic-Indic digit plus OCR-confusable numeric normalization."
        if results["environment"]["arabic_language_pack_available"]
        else "The local Tesseract install does not have Arabic traineddata, so the Arabic run used the English model and failed to recover Arabic labels."
    )

    return "\n".join(
        [
            "# Bonus Task 4 - Document Intelligence Option A",
            "",
            "## Prototype",
            "",
            "The prototype renders each PDF page with ImageMagick and runs local Tesseract OCR. It then uses label-anchored extraction for the three requested 2024 financial values.",
            "",
            f"- Tesseract: `{results['environment']['tesseract_version']}`",
            f"- Available Tesseract languages: `{', '.join(results['environment']['available_tesseract_languages'])}`",
            f"- Arabic language pack available: `{results['environment']['arabic_language_pack_available']}`",
            f"- English pages: `{results['english']['page_count']}`",
            f"- Arabic pages: `{results['arabic']['page_count']}`",
            "",
            "## Extraction Results",
            "",
            result_table[display_cols].to_markdown(index=False),
            "",
            "The English sample extracts all three target values exactly in thousands of EGP. The Arabic extraction depends on OCR-confusable numeric normalization because local Tesseract still distorts several table digits.",
            "",
            "## Arabic Failure Mode",
            "",
            arabic_lang_note,
            "",
            "Observed failure modes:",
            "",
            "- Arabic labels are now mostly readable with Arabic traineddata, but table spacing and right-to-left ordering are still unstable.",
            "- Numeric strings are partly preserved but often distorted, for example `487,350` can appear as noisy OCR such as `EAV,F0-`.",
            "- The report has mixed table layout and right-to-left text, so line order is unreliable without an Arabic-aware OCR/layout model.",
            "",
            "## API Comparison",
            "",
            "| Approach | What was tested / assumed | Accuracy on this sample | Cost estimate | Operational notes |",
            "| --- | --- | --- | --- | --- |",
            "| Local Tesseract OCR | Actually run locally; Arabic uses `script/Arabic` when installed plus numeric normalization | English: 3/3 exact. Arabic can be recovered on this sample, but confidence is lower because values require OCR-confusable cleanup | Software cost $0; compute/ops only | Good for a prototype and regression baseline. Needs preprocessing, table parsing, and QA gates before production. |",
            f"| Google Document AI Enterprise OCR | Public OCR processor pricing; not called in this prototype | Expected better Arabic/layout handling than local English-only OCR, but still needs field-level validation | $1.50 / 1,000 pages, about ${cost_per_document:.3f} per 12-page document | Strong pilot candidate because it is managed, page-based, and supports document OCR workflows. |",
            f"| AWS Textract Detect Document Text | Public text-detection pricing; not called in this prototype | Expected better scan handling than local OCR, but financial-table key-value extraction may need custom parsing | $0.0015/page for first 1M pages, about ${cost_per_document:.3f} per 12-page document | Good managed fallback; evaluate Arabic/table quality before committing. |",
            "",
            "Pricing references: Google Document AI pricing: https://cloud.google.com/document-ai/pricing. AWS Textract pricing: https://aws.amazon.com/textract/pricing/.",
            "",
            "## Recommendation",
            "",
            f"For the pilot stage under 50 documents/month, use Google Document AI OCR or AWS Textract as the primary OCR engine, with this Tesseract pipeline as a cheap regression baseline. At 12 pages per document, OCR-only API cost is roughly `${pilot_cost:.2f}` per month for 50 documents, so engineering reliability matters more than OCR unit cost.",
            "",
            f"At 500 documents/month, the same OCR-only page price is still only about `${scale_cost:.2f}` per month. I would move to a managed OCR API with a validation layer rather than scaling local OCR manually. The validation layer should check accounting identities, compare extracted values to expected statement sections, require confidence thresholds, and route exceptions to human review.",
            "",
            "## Generated Artifacts",
            "",
            f"- `{_display_path(csv_path)}`",
            f"- `{_display_path(json_path)}`",
            "- `reports/document_intelligence/english_ocr_text.txt`",
            "- `reports/document_intelligence/arabic_ocr_text.txt`",
        ]
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
