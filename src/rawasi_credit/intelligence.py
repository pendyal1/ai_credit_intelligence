from __future__ import annotations

import numpy as np
import pandas as pd


def require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    """Fail fast with a readable message when an input report is stale or incomplete."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{context} missing required columns: {missing}")


def find_recent_underperforming_clusters(
    df: pd.DataFrame,
    group_cols: list[str],
    split_col: str = "stage_b_split",
    target_col: str = "defaulted",
    probability_col: str = "stage_c_default_probability",
    train_label: str = "train",
    recent_label: str = "validation",
    min_recent_rows: int = 5,
    min_recent_default_rate: float = 0.08,
    min_default_rate_lift: float = 2.0,
    smoothing_defaults: float = 0.5,
    smoothing_rows: float = 10.0,
) -> pd.DataFrame:
    """
    Find segments where recent realized defaults are meaningfully worse
    than the historical/training baseline.

    This is used for Layer 2 trend detection.
    """

    require_columns(
        df,
        group_cols + [split_col, target_col, probability_col],
        "recent underperforming cluster input",
    )

    data = df[df[target_col].notna()].copy()
    data[target_col] = data[target_col].astype(int)

    train = data[data[split_col].eq(train_label)].copy()
    recent = data[data[split_col].eq(recent_label)].copy()
    if train.empty or recent.empty:
        return pd.DataFrame()

    train_summary = (
        train.groupby(group_cols, dropna=False)
        .agg(
            train_rows=(target_col, "size"),
            train_defaults=(target_col, "sum"),
            train_default_rate=(target_col, "mean"),
            train_avg_predicted_probability=(probability_col, "mean"),
        )
        .reset_index()
    )

    recent_summary = (
        recent.groupby(group_cols, dropna=False)
        .agg(
            recent_rows=(target_col, "size"),
            recent_defaults=(target_col, "sum"),
            recent_default_rate=(target_col, "mean"),
            recent_avg_predicted_probability=(probability_col, "mean"),
        )
        .reset_index()
    )

    clusters = recent_summary.merge(train_summary, on=group_cols, how="left")

    # If a segment did not exist in training, treat it as a cold-start monitoring issue.
    clusters["is_new_segment"] = clusters["train_rows"].isna()

    clusters["train_default_rate"] = clusters["train_default_rate"].fillna(0.0)
    clusters["train_avg_predicted_probability"] = clusters[
        "train_avg_predicted_probability"
    ].fillna(0.0)
    clusters["train_rows"] = clusters["train_rows"].fillna(0).astype(int)
    clusters["train_defaults"] = clusters["train_defaults"].fillna(0).astype(int)

    clusters["train_smoothed_default_rate"] = (
        clusters["train_defaults"] + smoothing_defaults
    ) / (clusters["train_rows"] + smoothing_rows)
    clusters["recent_smoothed_default_rate"] = (
        clusters["recent_defaults"] + smoothing_defaults
    ) / (clusters["recent_rows"] + smoothing_rows)
    clusters["default_rate_lift"] = (
        clusters["recent_smoothed_default_rate"] / clusters["train_smoothed_default_rate"]
    )

    clusters["model_underestimated_recent_risk"] = (
        clusters["recent_default_rate"] > clusters["recent_avg_predicted_probability"]
    )
    clusters["recent_default_minus_predicted"] = (
        clusters["recent_default_rate"] - clusters["recent_avg_predicted_probability"]
    )

    clusters["alert_flag"] = (
        (clusters["recent_rows"] >= min_recent_rows)
        & (clusters["recent_default_rate"] >= min_recent_default_rate)
        & (
            (clusters["default_rate_lift"] >= min_default_rate_lift)
            | (clusters["is_new_segment"])
        )
    )

    return clusters.sort_values(
        [
            "alert_flag",
            "recent_default_rate",
            "default_rate_lift",
            "recent_default_minus_predicted",
            "recent_rows",
        ],
        ascending=[False, False, False, False, False],
    )


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate Population Stability Index between a baseline distribution
    and a new/incoming distribution.
    """

    expected = expected.replace([np.inf, -np.inf], np.nan).dropna()
    actual = actual.replace([np.inf, -np.inf], np.nan).dropna()

    if expected.empty or actual.empty:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(expected.quantile(quantiles).to_numpy())

    if len(breakpoints) < 3:
        return np.nan

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_bins = pd.cut(expected, bins=breakpoints, include_lowest=True)
    actual_bins = pd.cut(actual, bins=breakpoints, include_lowest=True)

    expected_dist = expected_bins.value_counts(normalize=True, sort=False).clip(lower=epsilon)
    actual_dist = actual_bins.value_counts(normalize=True, sort=False).clip(lower=epsilon)

    actual_dist = actual_dist.reindex(expected_dist.index, fill_value=epsilon)

    psi = ((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum()

    return float(psi)


def batch_distribution_shift_report(
    baseline_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Compare an incoming batch to the training distribution using PSI.
    """

    require_columns(baseline_df, feature_cols, "distribution shift baseline")
    require_columns(incoming_df, feature_cols, "distribution shift incoming")

    rows = []
    for col in feature_cols:
        psi = calculate_psi(baseline_df[col], incoming_df[col])

        if pd.isna(psi):
            severity = "not_available"
        elif psi < 0.10:
            severity = "stable"
        elif psi < 0.25:
            severity = "moderate_shift"
        else:
            severity = "significant_shift"

        rows.append(
            {
                "feature": col,
                "psi": psi,
                "severity": severity,
                "human_review_required": severity == "significant_shift",
                "baseline_rows": int(baseline_df[col].notna().sum()),
                "incoming_rows": int(incoming_df[col].notna().sum()),
                "baseline_missing_rate": float(baseline_df[col].isna().mean()),
                "incoming_missing_rate": float(incoming_df[col].isna().mean()),
                "baseline_mean": _safe_mean(baseline_df[col]),
                "incoming_mean": _safe_mean(incoming_df[col]),
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")


def transaction_anomaly_report(
    df: pd.DataFrame,
    baseline_label: str = "train",
    incoming_labels: tuple[str, ...] = ("validation", "open_holdout"),
    split_col: str = "stage_b_split",
    probability_col: str = "stage_c_default_probability",
    benchmark_col: str = "stage_b_benchmark_score",
    peer_count_col: str = "stage_b_peer_high_risk_outlier_count",
    peer_score_col: str = "stage_b_peer_outlier_score",
    threshold_col: str = "stage_c_predicted_default",
    min_anomaly_score: float = 85.0,
) -> pd.DataFrame:
    """Rank validation/open transactions that look unusual versus training behavior."""

    require_columns(
        df,
        [
            "transaction_id",
            "invoice_date",
            "status",
            split_col,
            probability_col,
            benchmark_col,
            peer_count_col,
            peer_score_col,
            threshold_col,
        ],
        "transaction anomaly input",
    )

    baseline = df[df[split_col].eq(baseline_label)].copy()
    incoming = df[df[split_col].isin(incoming_labels)].copy()
    if baseline.empty or incoming.empty:
        return pd.DataFrame()

    incoming["probability_percentile_vs_train"] = _percentile_vs_baseline(
        incoming[probability_col],
        baseline[probability_col],
    )
    incoming["benchmark_percentile_vs_train"] = _percentile_vs_baseline(
        incoming[benchmark_col],
        baseline[benchmark_col],
    )
    incoming["peer_score_percentile_vs_train"] = _percentile_vs_baseline(
        incoming[peer_score_col],
        baseline[peer_score_col],
    )
    incoming["peer_count_percentile_vs_train"] = _percentile_vs_baseline(
        incoming[peer_count_col],
        baseline[peer_count_col],
    )

    incoming["layer2_anomaly_score"] = (
        0.45 * incoming["probability_percentile_vs_train"]
        + 0.25 * incoming["benchmark_percentile_vs_train"]
        + 0.20 * incoming["peer_score_percentile_vs_train"]
        + 0.10 * incoming["peer_count_percentile_vs_train"]
    )
    incoming["layer2_anomaly_alert"] = (
        (incoming["layer2_anomaly_score"] >= min_anomaly_score)
        | (incoming["probability_percentile_vs_train"] >= 95)
        | (
            (incoming[peer_count_col] >= 10)
            & (incoming["peer_score_percentile_vs_train"] >= 95)
        )
    )
    incoming["layer2_anomaly_reasons"] = incoming.apply(_anomaly_reasons, axis=1)

    preferred_cols = [
        "transaction_id",
        "invoice_date",
        "stage_b_split",
        "status",
        "defaulted",
        "buy_buyer_country",
        "goods_category",
        "invoice_currency",
        "stage_b_benchmark_band",
        probability_col,
        threshold_col,
        benchmark_col,
        peer_count_col,
        peer_score_col,
        "probability_percentile_vs_train",
        "benchmark_percentile_vs_train",
        "peer_score_percentile_vs_train",
        "peer_count_percentile_vs_train",
        "layer2_anomaly_score",
        "layer2_anomaly_alert",
        "stage_b_peer_outlier_reasons",
        "layer2_anomaly_reasons",
    ]
    cols = [col for col in preferred_cols if col in incoming.columns]
    return incoming[cols].sort_values(
        ["layer2_anomaly_alert", "layer2_anomaly_score"],
        ascending=[False, False],
    )


def _percentile_vs_baseline(values: pd.Series, baseline: pd.Series) -> pd.Series:
    clean_baseline = baseline.replace([np.inf, -np.inf], np.nan).dropna().sort_values()
    if clean_baseline.empty:
        return pd.Series(np.nan, index=values.index)
    ranks = np.searchsorted(clean_baseline.to_numpy(), values.to_numpy(), side="right")
    return pd.Series(ranks / len(clean_baseline) * 100, index=values.index)


def _anomaly_reasons(row: pd.Series) -> str:
    reasons = []
    if row.get("stage_c_predicted_default", 0) == 1:
        reasons.append("above Stage C review threshold")
    if row.get("probability_percentile_vs_train", 0) >= 95:
        reasons.append("probability above 95th percentile of training")
    if row.get("benchmark_percentile_vs_train", 0) >= 90:
        reasons.append("benchmark score above 90th percentile of training")
    if row.get("peer_score_percentile_vs_train", 0) >= 90:
        reasons.append("peer outlier score above 90th percentile of training")
    if row.get("stage_b_peer_high_risk_outlier_count", 0) >= 5:
        reasons.append("many high-risk peer outliers")
    return "; ".join(reasons)


def _safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else np.nan
