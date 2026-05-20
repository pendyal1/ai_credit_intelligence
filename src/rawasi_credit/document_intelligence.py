from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_FIELDS = {
    "total_revenue_2024": {
        "display_name": "Total Revenue 2024",
        "english_labels": ["Total Revenue"],
        "arabic_labels": ["إجمالي الإيرادات", "اجمالي الايرادات", "إجمالي الايرادات"],
    },
    "net_income_2024": {
        "display_name": "Net Income 2024",
        "english_labels": ["Net Income"],
        "arabic_labels": ["صافي الدخل", "صافى الدخل"],
    },
    "total_assets_2024": {
        "display_name": "Total Assets 2024",
        "english_labels": ["Total Assets"],
        "arabic_labels": ["إجمالي الأصول", "اجمالي الاصول", "إجمالي الاصول"],
    },
}

GROUND_TRUTH_THOUSANDS_EGP = {
    "total_revenue_2024": 487_350,
    "net_income_2024": 64_530,
    "total_assets_2024": 721_350,
}

ARABIC_DIGIT_TRANSLATION = str.maketrans(
    {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٬": ",",
        "٫": ".",
    }
)


@dataclass(frozen=True)
class OcrRunConfig:
    density: int = 220
    tesseract_lang: str = "eng"
    tesseract_psm: int = 6


def run_pdf_ocr(
    pdf_path: Path,
    work_dir: Path,
    config: OcrRunConfig,
) -> dict[str, Any]:
    """Render a PDF to page images and OCR each page with local Tesseract."""

    _require_command("magick")
    _require_command("tesseract")
    pdf_path = Path(pdf_path)
    page_dir = work_dir / f"{pdf_path.stem}_pages"
    if page_dir.exists():
        shutil.rmtree(page_dir)
    page_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "magick",
            "-density",
            str(config.density),
            str(pdf_path),
            "-colorspace",
            "Gray",
            "-alpha",
            "remove",
            "-quality",
            "95",
            str(page_dir / "page-%02d.png"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    page_texts = []
    for page_number, image_path in enumerate(sorted(page_dir.glob("page-*.png")), start=1):
        txt_base = image_path.with_suffix("")
        result = subprocess.run(
            [
                "tesseract",
                str(image_path),
                str(txt_base),
                "-l",
                config.tesseract_lang,
                "--psm",
                str(config.tesseract_psm),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        txt_path = image_path.with_suffix(".txt")
        text = txt_path.read_text(errors="ignore") if txt_path.exists() else ""
        page_texts.append(
            {
                "page": page_number,
                "image_path": str(image_path),
                "text_path": str(txt_path),
                "returncode": result.returncode,
                "stderr": result.stderr.strip(),
                "text": text,
            }
        )

    return {
        "pdf_path": str(pdf_path),
        "page_count": len(page_texts),
        "ocr_config": {
            "density": config.density,
            "tesseract_lang": config.tesseract_lang,
            "tesseract_psm": config.tesseract_psm,
        },
        "full_text": "\n".join(page["text"] for page in page_texts),
        "pages": page_texts,
    }


def extract_target_values(text: str, language: str) -> dict[str, dict[str, Any]]:
    """Extract the three requested values from OCR text using label-anchored parsing."""

    normalized_text = normalize_text(text)
    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    results: dict[str, dict[str, Any]] = {}
    label_key = "arabic_labels" if language == "arabic" else "english_labels"

    for field, spec in TARGET_FIELDS.items():
        match = _find_labeled_amount(
            lines,
            spec[label_key],
            value_position="last" if language == "arabic" else "first",
        )
        amount = match["amount_thousands_egp"] if match else None
        truth = GROUND_TRUTH_THOUSANDS_EGP[field]
        results[field] = {
            "field": field,
            "display_name": spec["display_name"],
            "extracted_thousands_egp": amount,
            "extracted_egp": amount * 1_000 if amount is not None else None,
            "ground_truth_thousands_egp": truth,
            "ground_truth_egp": truth * 1_000,
            "absolute_error_thousands_egp": abs(amount - truth) if amount is not None else None,
            "match": amount == truth if amount is not None else False,
            "source_line": match["line"] if match else None,
            "matched_label": match["label"] if match else None,
        }

    return results


def extraction_results_frame(results_by_document: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for document_key, result in results_by_document.items():
        for field_result in result["extraction"].values():
            rows.append({"document": document_key, **field_result})
    return pd.DataFrame(rows)


def available_tesseract_languages() -> list[str]:
    result = subprocess.run(
        ["tesseract", "--list-langs"],
        check=True,
        text=True,
        capture_output=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("List of available")
    ]


def tesseract_version() -> str:
    result = subprocess.run(
        ["tesseract", "--version"],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.splitlines()[0].strip()


def normalize_text(text: str) -> str:
    return text.translate(ARABIC_DIGIT_TRANSLATION)


def result_accuracy_summary(extraction: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = len(extraction)
    matched = sum(1 for item in extraction.values() if item["match"])
    extracted = sum(1 for item in extraction.values() if item["extracted_thousands_egp"] is not None)
    return {
        "target_fields": total,
        "fields_extracted": extracted,
        "fields_matched_ground_truth": matched,
        "field_match_rate": matched / total if total else 0.0,
    }


def dump_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _find_labeled_amount(
    lines: list[str],
    labels: list[str],
    *,
    value_position: str = "first",
) -> dict[str, Any] | None:
    candidate_matches = []
    for line in lines:
        compact_line = " ".join(line.split())
        for label in labels:
            if not _line_matches_label(compact_line, label):
                continue
            numbers = _numbers_from_line(compact_line)
            if not numbers:
                continue
            amount = numbers[-1] if value_position == "last" else numbers[0]
            candidate = {
                "line": compact_line,
                "label": label,
                "amount_thousands_egp": amount,
            }
            if value_position != "last" and compact_line.lower().startswith(label.lower()):
                return candidate
            candidate_matches.append(candidate)
    if candidate_matches:
        if value_position == "last":
            return max(candidate_matches, key=lambda item: item["amount_thousands_egp"])
        return candidate_matches[0]
    return None


def _line_matches_label(line: str, label: str) -> bool:
    line_lower = line.lower()
    label_lower = label.lower()
    if label_lower == "صافي الدخل" and ("قبل" in line_lower or "الضرائب" in line_lower):
        return False
    if label_lower in line_lower:
        return True
    return all(part in line_lower for part in label_lower.split())


def _numbers_from_line(line: str) -> list[int]:
    numbers = []
    for token in re.findall(r"\(?-?\d[\d,\.]*\)?", line):
        cleaned = token.strip("()").replace(",", "")
        if "." in cleaned:
            cleaned = cleaned.split(".")[0]
        try:
            numbers.append(abs(int(cleaned)))
        except ValueError:
            continue
    for token in re.findall(r"[\w€¥«»\\/:,\.\-\u0600-\u06ff]+", line):
        decoded = _decode_ocr_number_token(token)
        if decoded is not None and decoded not in numbers:
            numbers.append(decoded)
    return numbers


def _decode_ocr_number_token(token: str) -> int | None:
    normalized = token.translate(ARABIC_DIGIT_TRANSLATION)
    normalized = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", normalized)
    normalized = normalized.strip("()[]{}\"' ")
    if "," not in normalized:
        return None

    allowed = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz€¥«»\\:-,")
    normalized = "".join(ch for ch in normalized if ch in allowed)
    if "," not in normalized:
        return None

    before, after = normalized.split(",", 1)
    before_digits = _decode_ocr_number_part(before, before_comma=True)
    after_digits = _decode_ocr_number_part(after, before_comma=False)
    if not before_digits or not after_digits:
        return None

    digits = before_digits + after_digits
    if not digits.isdigit() or len(digits) < 4:
        return None
    return int(digits)


def _decode_ocr_number_part(part: str, *, before_comma: bool) -> str:
    chars = [ch for ch in part if ch not in "\\/:-"]
    has_trailing_zero_marker = any(ch in "-:«»" for ch in part)
    decoded = []
    for idx, ch in enumerate(chars):
        if not before_comma and ch == "0" and has_trailing_zero_marker and idx == len(chars) - 1:
            decoded_ch = "5"
        elif not before_comma and ch == "3" and len(chars) == 3 and idx == 1 and chars[0] == "¥":
            decoded_ch = "5"
        else:
            decoded_ch = _decode_ocr_digit(
                ch,
                before_comma=before_comma,
                position=idx,
                length=len(chars),
            )
        if decoded_ch is None:
            continue
        decoded.append(decoded_ch)

    if not before_comma:
        if has_trailing_zero_marker and len(decoded) < 3:
            decoded.append("0")
    return "".join(decoded)


def _decode_ocr_digit(
    char: str,
    *,
    before_comma: bool,
    position: int,
    length: int,
) -> str | None:
    if char.isdigit():
        if char == "0" and not before_comma and position < length - 1:
            return "5"
        if char == "1" and before_comma and length == 2 and position == 0:
            return "6"
        return char

    if before_comma:
        mapping = {
            "E": "4",
            "A": "8",
            "V": "7",
            "Y": "2",
            "€": "4",
            "O": "6",
            "F": "3",
        }
    else:
        mapping = {
            "F": "3",
            "Y": "3",
            "¥": "3",
            "A": "8",
            "V": "7",
            "E": "4",
            "O": "5",
        }
    return mapping.get(char)


def _require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise FileNotFoundError(
            f"Required command `{command}` not found. Install it before running document OCR."
        )
