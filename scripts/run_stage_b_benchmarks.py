#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rawasi_credit.benchmarks import (  # noqa: E402
    BENCHMARK_BAND_COLUMN,
    BENCHMARK_SCORE_COLUMN,
    PEER_OUTLIER_COUNT_COLUMN,
    PEER_OUTLIER_REASONS_COLUMN,
    StageBBenchmarkTransformer,
    load_benchmark_config,
    summarize_stage_b,
)
from rawasi_credit.data import find_dataset_path, load_dataset, split_resolved_open  # noqa: E402
from rawasi_credit.filters import DECISION_COLUMN, apply_hard_filters, load_hard_filter_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rawasi Stage B benchmark features.")
    parser.add_argument("--data", type=Path, default=None, help="Path to Rawasi CSV dataset.")
    parser.add_argument(
        "--hard-filter-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "hard_filters.json",
        help="Path to Stage A hard-filter JSON config.",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "benchmark_features.json",
        help="Path to Stage B benchmark JSON config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Directory for generated Stage B reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.data or find_dataset_path(PROJECT_ROOT)

    hard_filter_config = load_hard_filter_config(args.hard_filter_config)
    benchmark_config = load_benchmark_config(args.benchmark_config)

    df = load_dataset(dataset_path)
    stage_a_scored = apply_hard_filters(df, hard_filter_config)

    train, validation, open_holdout = split_resolved_open(stage_a_scored)
    train_pass = _stage_a_pass(train)
    validation_pass = _stage_a_pass(validation)
    open_pass = _stage_a_pass(open_holdout)

    transformer = StageBBenchmarkTransformer(benchmark_config).fit(train_pass)

    stage_b_train = transformer.transform(train_pass)
    stage_b_train["stage_b_split"] = "train"
    stage_b_validation = transformer.transform(validation_pass)
    stage_b_validation["stage_b_split"] = "validation"
    stage_b_open = transformer.transform(open_pass)
    stage_b_open["stage_b_split"] = "open_holdout"
    stage_b_all_pass = pd.concat([stage_b_train, stage_b_validation, stage_b_open], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scored_out = args.output_dir / "stage_b_scored_transactions.csv"
    stats_out = args.output_dir / "stage_b_feature_summary.csv"
    outliers_out = args.output_dir / "stage_b_peer_outliers.csv"
    transformer_out = args.output_dir / "stage_b_training_stats.json"
    report_out = args.output_dir / "stage_b_report.md"

    stage_b_all_pass.to_csv(scored_out, index=False)
    feature_summary = transformer.training_stats_frame()
    feature_summary.to_csv(stats_out, index=False)
    _peer_outliers(stage_b_all_pass).to_csv(outliers_out, index=False)
    transformer_out.write_text(json.dumps(transformer.to_dict(), indent=2))

    report_out.write_text(
        _render_report(
            dataset_path=_display_path(dataset_path),
            hard_filter_config_path=_display_path(args.hard_filter_config),
            benchmark_config_path=_display_path(args.benchmark_config),
            train_summary=summarize_stage_b(stage_b_train),
            validation_summary=summarize_stage_b(stage_b_validation),
            open_summary=summarize_stage_b(stage_b_open),
            feature_summary=feature_summary,
            top_outliers=_peer_outliers(stage_b_all_pass).head(20),
        )
    )

    print("Stage B benchmarks complete.")
    print(f"Dataset: {dataset_path}")
    print(f"Training rows used for Stage B fit: {len(train_pass):,}")
    print(f"Validation rows transformed: {len(validation_pass):,}")
    print(f"Open rows transformed: {len(open_pass):,}")
    print("\nValidation benchmark summary:")
    print(summarize_stage_b(stage_b_validation).to_string(index=False))
    print(f"\nWrote: {report_out}")
    print(f"Wrote: {scored_out}")
    print(f"Wrote: {stats_out}")
    print(f"Wrote: {outliers_out}")
    print(f"Wrote: {transformer_out}")


def _stage_a_pass(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[DECISION_COLUMN].eq("PASS_STAGE_A")].copy()


def _peer_outliers(scored: pd.DataFrame) -> pd.DataFrame:
    outliers = scored[scored[PEER_OUTLIER_COUNT_COLUMN] > 0].copy()
    cols = [
        "transaction_id",
        "stage_b_split",
        "status",
        "invoice_date",
        "goods_category",
        "defaulted",
        BENCHMARK_SCORE_COLUMN,
        BENCHMARK_BAND_COLUMN,
        PEER_OUTLIER_COUNT_COLUMN,
        PEER_OUTLIER_REASONS_COLUMN,
        "ticket_to_revenue_est",
        "outstanding_to_revenue_est",
        "advance_tenor_exposure",
    ]
    present_cols = [col for col in cols if col in outliers.columns]
    return outliers[present_cols].sort_values(
        [PEER_OUTLIER_COUNT_COLUMN, BENCHMARK_SCORE_COLUMN],
        ascending=[False, False],
    )


def _render_report(
    *,
    dataset_path: str,
    hard_filter_config_path: str,
    benchmark_config_path: str,
    train_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
    open_summary: pd.DataFrame,
    feature_summary: pd.DataFrame,
    top_outliers: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Stage B Benchmark Report",
            "",
            f"Dataset: `{dataset_path}`",
            f"Stage A config: `{hard_filter_config_path}`",
            f"Stage B config: `{benchmark_config_path}`",
            "",
            "## Purpose",
            "",
            "Stage B creates cross-sector benchmark features for deals that pass Stage A. It also identifies within-sector peer outliers using the same three features.",
            "",
            "## Features Used",
            "",
            feature_summary.to_markdown(index=False),
            "",
            "## Train Benchmark Summary",
            "",
            train_summary.to_markdown(index=False),
            "",
            "## Validation Benchmark Summary",
            "",
            validation_summary.to_markdown(index=False),
            "",
            "## Open Holdout Benchmark Summary",
            "",
            open_summary.to_markdown(index=False),
            "",
            "## Top Peer Outliers",
            "",
            top_outliers.to_markdown(index=False) if len(top_outliers) else "No peer outliers found.",
            "",
            "## Notes",
            "",
            "- Stage B is fit on Stage-A-passed training rows only.",
            "- Validation and open rows are transformed using training statistics.",
            "- Percentile scores are global cross-sector benchmarks.",
            "- Peer outlier flags compare each deal to its `goods_category` training peers; sparse sectors fall back to global training statistics.",
        ]
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
