#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rawasi_credit.intelligence import (  # noqa: E402
    batch_distribution_shift_report,
    find_recent_underperforming_clusters,
    transaction_anomaly_report,
)
from rawasi_credit.modeling import stage_c_feature_frame  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rawasi Layer 2 trend and anomaly detection.")
    parser.add_argument(
        "--stage-b-scored",
        type=Path,
        default=REPORTS_DIR / "stage_b_scored_transactions.csv",
        help="Stage B scored transaction CSV with train/validation/open_holdout rows.",
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=MODELS_DIR / "stage_c_xgboost_model.joblib",
        help="Saved Stage C model artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for generated Layer 2 reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_exists(args.stage_b_scored, "Stage B scored transactions")
    _ensure_exists(args.model_artifact, "Stage C model artifact")

    stage_b = pd.read_csv(args.stage_b_scored, parse_dates=["invoice_date"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*serialized model.*", category=UserWarning)
        model_artifact = joblib.load(args.model_artifact)
    monitoring_frame = _score_monitoring_frame(stage_b, model_artifact)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    monitoring_frame.to_csv(args.output_dir / "layer2_scored_monitoring_frame.csv", index=False)

    clusters = _build_cluster_report(monitoring_frame)
    clusters.to_csv(args.output_dir / "layer2_trend_clusters.csv", index=False)

    drift_report = _build_drift_report(monitoring_frame, model_artifact)
    drift_report.to_csv(args.output_dir / "layer2_distribution_shift.csv", index=False)

    anomalies = transaction_anomaly_report(monitoring_frame)
    anomalies.to_csv(args.output_dir / "layer2_transaction_anomalies.csv", index=False)

    report = _render_report(
        clusters=clusters,
        drift_report=drift_report,
        anomalies=anomalies,
        stage_b_scored=args.stage_b_scored,
        model_artifact=args.model_artifact,
    )
    (args.output_dir / "layer2_intelligence_report.md").write_text(report)

    print("Layer 2 intelligence complete.")
    print(f"Monitoring rows: {len(monitoring_frame):,}")
    print(f"Trend clusters: {len(clusters):,}; alerts: {int(clusters['alert_flag'].sum()) if len(clusters) else 0:,}")
    print(
        "Distribution shifts: "
        f"{int(drift_report['human_review_required'].sum()) if len(drift_report) else 0:,} significant"
    )
    print(
        "Transaction anomalies: "
        f"{int(anomalies['layer2_anomaly_alert'].sum()) if len(anomalies) else 0:,} alerts"
    )
    print(f"Wrote: {args.output_dir / 'layer2_intelligence_report.md'}")


def _score_monitoring_frame(stage_b: pd.DataFrame, model_artifact: dict) -> pd.DataFrame:
    feature_columns = model_artifact["feature_columns"]
    model = model_artifact["model"]
    threshold = model_artifact["threshold"]

    scored = stage_b.copy()
    x_all = stage_c_feature_frame(scored, feature_columns)
    scored["stage_c_default_probability"] = model.predict_proba(x_all)[:, 1]
    scored["stage_c_predicted_default"] = (
        scored["stage_c_default_probability"] >= threshold
    ).astype(int)
    scored = _add_shap_reason_columns(scored, model, x_all)
    return scored


def _add_shap_reason_columns(
    scored: pd.DataFrame,
    model,
    feature_frame: pd.DataFrame,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for Layer 2 model-driver context. Install it with "
            "`python -m pip install shap==0.44.1` in this environment."
        ) from exc

    out = scored.copy()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_frame)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = np.asarray(expected_value).ravel()[-1]
    expected_value = float(expected_value)

    out["stage_c_shap_margin"] = expected_value + shap_values.sum(axis=1)
    out["stage_c_top_positive_shap_reasons"] = [
        _format_shap_reasons(shap_values[idx], feature_frame.iloc[idx], largest_positive=True)
        for idx in range(len(out))
    ]
    out["stage_c_top_negative_shap_reasons"] = [
        _format_shap_reasons(shap_values[idx], feature_frame.iloc[idx], largest_positive=False)
        for idx in range(len(out))
    ]
    return out


def _format_shap_reasons(
    shap_row: np.ndarray,
    feature_values: pd.Series,
    *,
    largest_positive: bool,
    limit: int = 5,
) -> str:
    values = pd.Series(shap_row, index=feature_values.index)
    values = values[values.gt(0)] if largest_positive else values[values.lt(0)]
    if values.empty:
        return ""
    values = values.sort_values(ascending=not largest_positive).head(limit)
    return "; ".join(
        f"{feature}={_format_feature_value(feature_values[feature])} ({shap_value:+.4f})"
        for feature, shap_value in values.items()
    )


def _format_feature_value(value: object) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _build_cluster_report(monitoring_frame: pd.DataFrame) -> pd.DataFrame:
    cluster_specs = [
        ["buy_buyer_country", "goods_category"],
        ["buy_buyer_country", "invoice_currency"],
        ["goods_category", "insurance_covered"],
        ["buy_buyer_country", "exp_segment"],
        ["buy_buyer_sector", "goods_category"],
        ["exp_industry", "goods_category"],
    ]

    all_clusters = []
    for group_cols in cluster_specs:
        cluster_df = find_recent_underperforming_clusters(
            df=monitoring_frame,
            group_cols=group_cols,
            probability_col="stage_c_default_probability",
            min_recent_rows=5,
            min_recent_default_rate=0.08,
            min_default_rate_lift=1.75,
        )
        if cluster_df.empty:
            continue
        cluster_df.insert(0, "cluster_definition", " + ".join(group_cols))
        cluster_df.insert(1, "cluster_key", _cluster_key(cluster_df, group_cols))
        all_clusters.append(cluster_df)

    if not all_clusters:
        return pd.DataFrame()
    return pd.concat(all_clusters, ignore_index=True)


def _build_drift_report(monitoring_frame: pd.DataFrame, model_artifact: dict) -> pd.DataFrame:
    priority_features = [
        "stage_c_default_probability",
        "stage_c_predicted_default",
        "stage_b_benchmark_score",
        "stage_b_peer_outlier_score",
        "stage_b_peer_high_risk_outlier_count",
        "log_cumulative_outstanding_usd",
        "total_exposure_to_revenue_est",
        "exposure_quality_pressure",
        "buyer_terms_uninsured_pressure",
        "buyer_dso_to_terms",
        "exp_leverage_liquidity_stress",
        "profitability_pressure",
        "trade_history_pressure",
        "liquidity_pressure",
    ]

    top_model_features = [
        feature
        for feature in model_artifact.get("feature_columns", [])
        if feature in monitoring_frame.columns
    ][:30]
    drift_features = _dedupe(
        [feature for feature in priority_features + top_model_features if feature in monitoring_frame.columns]
    )

    train_df = monitoring_frame[monitoring_frame["stage_b_split"].eq("train")].copy()
    validation_df = monitoring_frame[monitoring_frame["stage_b_split"].eq("validation")].copy()
    return batch_distribution_shift_report(
        baseline_df=train_df,
        incoming_df=validation_df,
        feature_cols=drift_features,
    )


def _render_report(
    *,
    clusters: pd.DataFrame,
    drift_report: pd.DataFrame,
    anomalies: pd.DataFrame,
    stage_b_scored: Path,
    model_artifact: Path,
) -> str:
    alert_clusters = clusters[clusters["alert_flag"]].head(20) if len(clusters) else pd.DataFrame()
    significant_shift = (
        drift_report[drift_report["human_review_required"]].head(20)
        if len(drift_report)
        else pd.DataFrame()
    )
    anomaly_alerts = (
        anomalies[anomalies["layer2_anomaly_alert"]].head(20)
        if len(anomalies)
        else pd.DataFrame()
    )

    return "\n".join(
        [
            "# Layer 2 Trend and Anomaly Detection Report",
            "",
            f"Stage B scored input: `{_display_path(stage_b_scored)}`",
            f"Stage C model artifact: `{_display_path(model_artifact)}`",
            "",
            "## Method",
            "",
            "- Scores Stage-B-passed train, validation, and open-holdout rows with the saved Stage C model.",
            "- Adds per-transaction SHAP reason fields so analyst and LLM summaries can explain model drivers.",
            "- Detects recent underperforming clusters by comparing validation default rates with training default-rate baselines.",
            "- Detects feature drift with Population Stability Index between train and validation.",
            "- Ranks validation and open transactions with an anomaly score based on probability, benchmark score, and peer outlier severity.",
            "",
            "## Alert Clusters",
            "",
            _for_markdown(alert_clusters, _cluster_report_cols())
            if len(alert_clusters)
            else "No alert clusters found.",
            "",
            "## Significant Distribution Shifts",
            "",
            _for_markdown(significant_shift, _drift_report_cols())
            if len(significant_shift)
            else "No significant PSI shifts found.",
            "",
            "## Top Transaction Anomalies",
            "",
            _for_markdown(anomaly_alerts, _anomaly_report_cols())
            if len(anomaly_alerts)
            else "No transaction anomaly alerts found.",
            "",
            "## Generated Artifacts",
            "",
            "- `reports/layer2_scored_monitoring_frame.csv`",
            "- `reports/layer2_trend_clusters.csv`",
            "- `reports/layer2_distribution_shift.csv`",
            "- `reports/layer2_transaction_anomalies.csv`",
            "- `reports/layer2_intelligence_report.md`",
        ]
    )


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}. Run Stage B and Stage C first.")


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _cluster_key(frame: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    return frame[group_cols].astype(str).agg(" / ".join, axis=1)


def _for_markdown(frame: pd.DataFrame, preferred_cols: list[str]) -> str:
    cols = [col for col in preferred_cols if col in frame.columns]
    return frame[cols].to_markdown(index=False)


def _cluster_report_cols() -> list[str]:
    return [
        "cluster_definition",
        "cluster_key",
        "recent_rows",
        "recent_defaults",
        "recent_default_rate",
        "train_rows",
        "train_defaults",
        "train_default_rate",
        "default_rate_lift",
        "recent_avg_predicted_probability",
        "model_underestimated_recent_risk",
        "recent_default_minus_predicted",
    ]


def _drift_report_cols() -> list[str]:
    return [
        "feature",
        "psi",
        "severity",
        "human_review_required",
        "baseline_mean",
        "incoming_mean",
        "baseline_missing_rate",
        "incoming_missing_rate",
    ]


def _anomaly_report_cols() -> list[str]:
    return [
        "transaction_id",
        "invoice_date",
        "stage_b_split",
        "status",
        "defaulted",
        "buy_buyer_country",
        "goods_category",
        "stage_c_default_probability",
        "stage_b_benchmark_score",
        "stage_b_peer_high_risk_outlier_count",
        "layer2_anomaly_score",
        "layer2_anomaly_reasons",
    ]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
