#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rawasi_credit.data import find_dataset_path, load_dataset, split_resolved_open
from rawasi_credit.filters import apply_hard_filters, load_hard_filter_config, summarize_stage_a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rawasi Stage A hard filters.")
    parser.add_argument("--data", type=Path, default=None, help="Path to Rawasi CSV dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "hard_filters.json",
        help="Path to hard-filter JSON config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Directory for generated Stage A reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.data or find_dataset_path(PROJECT_ROOT)
    config = load_hard_filter_config(args.config)
    df = load_dataset(dataset_path)

    scored = apply_hard_filters(df, config)
    train, validation, open_holdout = split_resolved_open(scored)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summary, all_rule_summary = summarize_stage_a(scored, config)
    train_summary, _ = summarize_stage_a(train, config)
    validation_summary, _ = summarize_stage_a(validation, config)
    open_summary, _ = summarize_stage_a(open_holdout, config)

    scored_out = args.output_dir / "stage_a_scored_transactions.csv"
    rule_out = args.output_dir / "stage_a_rule_summary.csv"
    report_out = args.output_dir / "stage_a_report.md"

    scored.to_csv(scored_out, index=False)
    all_rule_summary.to_csv(rule_out, index=False)
    report_out.write_text(
        _render_report(
            dataset_path=_display_path(dataset_path),
            config_path=_display_path(args.config),
            all_summary=all_summary,
            train_summary=train_summary,
            validation_summary=validation_summary,
            open_summary=open_summary,
            rule_summary=all_rule_summary,
        )
    )

    print("Stage A hard filters complete.")
    print(f"Dataset: {dataset_path}")
    print(f"Config: {args.config}")
    print("\nOverall decision summary:")
    print(all_summary.to_string(index=False))
    print("\nRule summary:")
    print(all_rule_summary.to_string(index=False))
    print(f"\nWrote: {report_out}")
    print(f"Wrote: {rule_out}")
    print(f"Wrote: {scored_out}")


def _render_report(
    *,
    dataset_path: str,
    config_path: str,
    all_summary,
    train_summary,
    validation_summary,
    open_summary,
    rule_summary,
) -> str:
    return "\n".join(
        [
            "# Stage A Hard Filter Report",
            "",
            f"Dataset: `{dataset_path}`",
            f"Config: `{config_path}`",
            "",
            "## Purpose",
            "",
            "Stage A removes deals that fail non-negotiable credit policy or data-quality criteria before benchmark scoring or statistical modeling.",
            "",
            "## Overall Decision Summary",
            "",
            all_summary.to_markdown(index=False),
            "",
            "## Train Decision Summary",
            "",
            train_summary.to_markdown(index=False),
            "",
            "## Validation Decision Summary",
            "",
            validation_summary.to_markdown(index=False),
            "",
            "## Open Holdout Decision Summary",
            "",
            open_summary.to_markdown(index=False),
            "",
            "## Rule Hit Summary",
            "",
            rule_summary.to_markdown(index=False),
            "",
            "## Notes",
            "",
            "- All thresholds and accepted values live in `config/hard_filters.json`.",
            "- The filter layer only uses fields known at the financing decision date.",
            "- `status`, `defaulted`, `actual_resolution_date`, `days_to_resolution`, and buyer baseline default probability are not used by Stage A.",
        ]
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
