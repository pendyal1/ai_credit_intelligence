from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BENCHMARK_SCORE_COLUMN = "stage_b_benchmark_score"
BENCHMARK_BAND_COLUMN = "stage_b_benchmark_band"
PEER_GROUP_SOURCE_COLUMN = "stage_b_peer_group_source"
PEER_OUTLIER_COUNT_COLUMN = "stage_b_peer_outlier_count"
PEER_HIGH_RISK_OUTLIER_COUNT_COLUMN = "stage_b_peer_high_risk_outlier_count"
PEER_OUTLIER_SCORE_COLUMN = "stage_b_peer_outlier_score"
PEER_OUTLIER_REASONS_COLUMN = "stage_b_peer_outlier_reasons"


def load_benchmark_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        config = json.load(f)
    _validate_config(config)
    return config


@dataclass
class StageBBenchmarkTransformer:
    """Fit decision-time benchmark statistics and transform Rawasi transactions."""

    config: dict[str, Any]

    def __post_init__(self) -> None:
        _validate_config(self.config)
        self.feature_names = [feature["name"] for feature in self.config["features"]]
        self.peer_feature_names = [
            feature["name"]
            for feature in self.config["features"]
            if feature.get("include_in_peer_outliers", True)
        ]
        self.weights = np.array([feature.get("weight", 1.0) for feature in self.config["features"]])
        self.weights = self.weights / self.weights.sum()
        self.global_feature_values_: dict[str, np.ndarray] = {}
        self.global_peer_stats_: dict[str, dict[str, float]] = {}
        self.group_peer_stats_: dict[str, dict[str, dict[str, float]]] = {}
        self.fitted_ = False

    def fit(self, df: pd.DataFrame) -> "StageBBenchmarkTransformer":
        featured = self.add_cross_sector_features(df)

        for feature_name in self.feature_names:
            values = featured[feature_name].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(values) == 0:
                raise ValueError(f"No valid training values for benchmark feature: {feature_name}")
            self.global_feature_values_[feature_name] = np.sort(values)
            self.global_peer_stats_[feature_name] = _robust_stats(values)

        group_col = self.config["peer_outliers"]["group_column"]
        min_group_size = self.config["peer_outliers"]["minimum_peer_group_size"]

        self.group_peer_stats_ = {}
        for group_value, group_df in featured.groupby(group_col, dropna=False):
            if len(group_df) < min_group_size:
                continue
            group_key = _group_key(group_value)
            self.group_peer_stats_[group_key] = {
                feature_name: _robust_stats(group_df[feature_name].dropna().to_numpy())
                for feature_name in self.peer_feature_names
            }

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("StageBBenchmarkTransformer must be fitted before transform().")

        result = self.add_cross_sector_features(df)
        percentile_data = {}

        for feature in self.config["features"]:
            feature_name = feature["name"]
            percentile_col = f"stage_b_{feature_name}_risk_percentile"
            percentile = _percentile_rank(
                result[feature_name],
                self.global_feature_values_[feature_name],
            )
            if not feature.get("higher_is_riskier", True):
                percentile = 1 - percentile
            percentile_data[percentile_col] = percentile * 100

        percentile_frame = pd.DataFrame(percentile_data, index=result.index)
        percentile_cols = list(percentile_frame.columns)
        result = pd.concat([result, percentile_frame], axis=1).copy()
        result[BENCHMARK_SCORE_COLUMN] = np.average(
            result[percentile_cols].to_numpy(),
            axis=1,
            weights=self.weights,
        )
        result[BENCHMARK_BAND_COLUMN] = result[BENCHMARK_SCORE_COLUMN].apply(
            lambda score: _score_band(score, self.config)
        )

        self._append_peer_outlier_columns(result)
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def add_cross_sector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        egp_per_usd = self.config["egp_per_usd_assumption"]

        annual_revenue_usd_est = (
            result["exp_annual_revenue_egp_millions"] * 1_000_000 / egp_per_usd
        ).replace(0, np.nan)

        result["annual_revenue_usd_est"] = annual_revenue_usd_est
        result["log_annual_revenue_usd_est"] = _log1p_nonnegative(annual_revenue_usd_est)
        result["log_invoice_amount_usd"] = _log1p_nonnegative(result["invoice_amount_usd"])
        result["log_cumulative_outstanding_usd"] = _log1p_nonnegative(
            result["cumulative_outstanding_usd"]
        )
        result["ticket_to_revenue_est"] = result["invoice_amount_usd"] / annual_revenue_usd_est
        result["outstanding_to_revenue_est"] = (
            result["cumulative_outstanding_usd"] / annual_revenue_usd_est
        )
        result["log_ticket_to_revenue_est"] = _log1p_nonnegative(
            result["ticket_to_revenue_est"]
        )
        result["log_outstanding_to_revenue_est"] = _log1p_nonnegative(
            result["outstanding_to_revenue_est"]
        )

        result["funded_amount_usd"] = result["invoice_amount_usd"] * result["advance_pct"]
        result["log_funded_amount_usd"] = _log1p_nonnegative(result["funded_amount_usd"])
        result["funded_to_revenue_est"] = result["funded_amount_usd"] / annual_revenue_usd_est
        result["log_funded_to_revenue_est"] = _log1p_nonnegative(
            result["funded_to_revenue_est"]
        )
        result["total_exposure_after_advance_usd"] = (
            result["cumulative_outstanding_usd"] + result["funded_amount_usd"]
        )
        result["log_total_exposure_after_advance_usd"] = _log1p_nonnegative(
            result["total_exposure_after_advance_usd"]
        )
        result["total_exposure_to_revenue_est"] = (
            result["total_exposure_after_advance_usd"] / annual_revenue_usd_est
        )
        result["log_total_exposure_to_revenue_est"] = _log1p_nonnegative(
            result["total_exposure_to_revenue_est"]
        )
        result["advance_tenor_exposure"] = result["advance_pct"] * result["payment_terms_days"]
        result["log_advance_tenor_exposure"] = _log1p_nonnegative(
            result["advance_tenor_exposure"]
        )
        result["insurance_not_covered_flag"] = (~result["insurance_covered"]).astype(int)
        result["uninsured_funded_amount_usd"] = (
            result["funded_amount_usd"] * result["insurance_not_covered_flag"]
        )
        result["log_uninsured_funded_amount_usd"] = _log1p_nonnegative(
            result["uninsured_funded_amount_usd"]
        )
        result["uninsured_funded_to_revenue_est"] = (
            result["uninsured_funded_amount_usd"] / annual_revenue_usd_est
        )
        result["funded_tenor_usd_days_log"] = _log1p_nonnegative(
            result["funded_amount_usd"] * result["payment_terms_days"]
        )
        result["funded_tenor_to_revenue_days"] = (
            result["funded_to_revenue_est"] * result["payment_terms_days"]
        )
        result["total_exposure_tenor_to_revenue_days"] = (
            result["total_exposure_to_revenue_est"] * result["payment_terms_days"]
        )

        result["exp_iscore_risk"] = 850 - result["exp_iscore"]
        result["exp_iscore_risk_squared"] = result["exp_iscore_risk"] ** 2
        result["exp_young_firm_risk"] = 1 / (result["exp_years_in_business"] + 1)
        result["exp_management_tenure_risk"] = 1 / (result["exp_mgmt_tenure_years"] + 1)
        result["exp_mgmt_succession_gap_flag"] = (
            ~result["exp_mgmt_succession_in_place"]
        ).astype(int)
        result["exp_current_ratio_shortfall"] = (1.5 - result["exp_current_ratio"]).clip(lower=0)
        result["exp_quick_ratio_shortfall"] = (1.0 - result["exp_quick_ratio"]).clip(lower=0)
        result["exp_dscr_shortfall"] = (1.5 - result["exp_dscr"]).clip(lower=0)
        result["exp_leverage_liquidity_stress"] = result["exp_debt_to_equity"] / (
            result["exp_current_ratio"] + result["exp_quick_ratio"] + result["exp_dscr"] + 0.1
        )
        result["exp_operating_margin_shortfall"] = (
            10 - result["exp_operating_margin_pct"]
        ).clip(lower=0) / 100
        result["exp_gross_margin_shortfall"] = (
            20 - result["exp_gross_profit_margin_pct"]
        ).clip(lower=0) / 100
        result["exp_negative_operating_margin_flag"] = (
            result["exp_operating_margin_pct"] < 0
        ).astype(int)
        result["exp_sales_decline_flag"] = (result["exp_sales_growth_pct"] < 0).astype(int)
        result["exp_sales_decline_magnitude_pct"] = (
            -result["exp_sales_growth_pct"]
        ).clip(lower=0)
        result["exp_fx_loss_abs_pct"] = result["exp_fx_loss_to_npat_pct"].abs()
        result["non_usd_invoice_flag"] = result["invoice_currency"].ne("USD").astype(int)
        result["exp_fx_loss_x_non_usd"] = (
            result["exp_fx_loss_abs_pct"] * result["non_usd_invoice_flag"]
        )
        result["exp_fx_loss_x_tenor"] = (
            result["exp_fx_loss_abs_pct"] * result["payment_terms_days"] / 180
        )
        result["exp_historical_on_time_gap"] = 1 - result["exp_historical_on_time_payment_rate"]
        result["exp_low_history_flag"] = (result["exp_historical_invoices_count"] < 25).astype(int)
        result["exp_history_depth_risk"] = 1 / (result["exp_historical_invoices_count"] + 1)
        result["exp_on_time_gap_x_low_history"] = (
            result["exp_historical_on_time_gap"] * (1 + result["exp_history_depth_risk"])
        )
        result["exp_dispute_to_history_pressure"] = (
            result["exp_historical_dispute_rate"]
            * np.log1p(result["exp_historical_invoices_count"])
        )
        result["recent_velocity_to_history"] = (
            result["recent_velocity_count"] / (result["exp_historical_invoices_count"] + 1)
        )
        result["velocity_exposure_pressure"] = (
            result["recent_velocity_to_history"] * result["total_exposure_to_revenue_est"]
        )
        result["concentration_adjusted_ticket_to_revenue"] = (
            result["ticket_to_revenue_est"] * result["exp_concentration_top1_buyer_pct"]
        )
        result["concentration_adjusted_total_exposure"] = (
            result["total_exposure_to_revenue_est"] * result["exp_concentration_top1_buyer_pct"]
        )
        result["exp_banking_depth_risk"] = (
            1 / (result["exp_n_banking_relationships"] + 1)
            + 1 / (result["exp_banking_relationship_years"] + 1)
        )
        result["exp_auditor_tier_risk_score"] = result["exp_auditor_tier"].map(
            {"Big4": 0.0, "International": 0.25, "CBE-approved": 0.5, "Non-CBE": 1.0}
        )
        result["exp_audit_qualified_flag"] = result["exp_audit_qualified"].astype(int)
        result["exp_uses_offshore_tradeco_flag"] = (
            result["exp_uses_offshore_tradeco"].astype(int)
        )
        result["exp_free_zone_flag"] = result["exp_free_zone"].astype(int)
        result["exp_segment_sme_flag"] = result["exp_segment"].eq("SME").astype(int)
        result["exp_segment_lc_flag"] = result["exp_segment"].eq("LC").astype(int)

        result["buy_creditsafe_rating_score"] = (
            result["buy_creditsafe_rating"].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5})
        )
        result["buyer_dso_gap_days"] = (
            result["buy_buyer_payment_history_dso_avg"] - result["payment_terms_days"]
        ).clip(lower=0)
        result["buyer_dso_to_terms"] = (
            result["buy_buyer_payment_history_dso_avg"] / (result["payment_terms_days"] + 1)
        )
        result["payment_terms_to_buyer_dso"] = (
            result["payment_terms_days"] / (result["buy_buyer_payment_history_dso_avg"] + 1)
        )
        result["country_risk_x_tenor"] = (
            result["buy_buyer_country_risk_score"] * result["payment_terms_days"] / 180
        )
        result["creditsafe_risk_x_tenor"] = (
            result["buy_creditsafe_rating_score"] * result["payment_terms_days"] / 180
        )

        dso_scaled = result["buy_buyer_payment_history_dso_avg"].clip(lower=0, upper=180) / 180
        result["buyer_risk_pressure"] = (
            (result["buy_buyer_country_risk_score"] / 10)
            + (result["buy_creditsafe_rating_score"] / 5)
            + dso_scaled
        ) / 3
        result["trade_history_pressure"] = (
            result["exp_historical_on_time_gap"]
            + result["exp_historical_dispute_rate"]
            + result["exp_concentration_top1_buyer_pct"]
        )
        result["profitability_pressure"] = (
            result["exp_operating_margin_shortfall"]
            + result["exp_gross_margin_shortfall"]
            + result["exp_sales_decline_magnitude_pct"] / 100
        )
        result["liquidity_pressure"] = (
            result["exp_current_ratio_shortfall"]
            + result["exp_quick_ratio_shortfall"]
            + result["exp_dscr_shortfall"]
            + (result["exp_debt_to_equity"] / 5)
        )
        result["buyer_terms_pressure"] = (
            result["buyer_risk_pressure"]
            * (result["payment_terms_days"] / 180)
            * (1 + result["insurance_not_covered_flag"])
        )
        result["country_risk_x_uninsured"] = (
            result["buy_buyer_country_risk_score"] * result["insurance_not_covered_flag"]
        )
        result["country_terms_uninsured_pressure"] = (
            result["country_risk_x_uninsured"] * result["payment_terms_days"] / 180
        )
        result["buyer_terms_uninsured_pressure"] = (
            result["buyer_risk_pressure"]
            * result["insurance_not_covered_flag"]
            * result["payment_terms_days"]
            / 180
        )
        result["exposure_quality_pressure"] = (
            result["total_exposure_to_revenue_est"]
            * (1 + result["buyer_risk_pressure"])
            * (1 + result["trade_history_pressure"])
        )
        result["loss_given_default_proxy"] = (
            result["funded_to_revenue_est"] * (1 + result["insurance_not_covered_flag"])
        )

        return result

    def training_stats_frame(self) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("StageBBenchmarkTransformer must be fitted before stats export.")

        rows = []
        for feature in self.config["features"]:
            name = feature["name"]
            values = self.global_feature_values_[name]
            stats = self.global_peer_stats_[name]
            rows.append(
                {
                    "feature": name,
                    "description": feature.get("description", ""),
                    "why_cross_sector": feature.get("why_cross_sector", ""),
                    "training_rows": len(values),
                    "min": float(np.min(values)),
                    "p25": float(np.quantile(values, 0.25)),
                    "median": stats["median"],
                    "p75": float(np.quantile(values, 0.75)),
                    "max": float(np.max(values)),
                    "mad": stats["mad"],
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        if not self.fitted_:
            raise RuntimeError("StageBBenchmarkTransformer must be fitted before export.")
        return {
            "config": self.config,
            "global_peer_stats": self.global_peer_stats_,
            "group_peer_stats": self.group_peer_stats_,
            "global_feature_values": {
                feature: values.tolist()
                for feature, values in self.global_feature_values_.items()
            },
        }

    def _append_peer_outlier_columns(self, result: pd.DataFrame) -> None:
        group_col = self.config["peer_outliers"]["group_column"]
        threshold = self.config["peer_outliers"]["robust_z_threshold"]

        sources = []
        outlier_counts = []
        high_risk_counts = []
        scores = []
        reasons = []

        for _, row in result.iterrows():
            group_key = _group_key(row[group_col])
            group_stats = self.group_peer_stats_.get(group_key)
            if group_stats is None:
                group_stats = self.global_peer_stats_
                source = "global_fallback"
                peer_name = "global training population"
            else:
                source = f"{group_col}"
                peer_name = f"{group_col}={row[group_col]}"

            row_reasons = []
            row_abs_z_scores = []
            row_high_risk_count = 0

            for feature_name in self.peer_feature_names:
                stats = group_stats[feature_name]
                value = row[feature_name]
                z_score = _robust_z_score(value, stats["median"], stats["mad"])
                z_col = f"stage_b_{feature_name}_peer_z"
                result.at[row.name, z_col] = z_score

                if pd.isna(z_score):
                    continue

                abs_z = abs(float(z_score))
                row_abs_z_scores.append(abs_z)
                if abs_z >= threshold:
                    direction = "high" if z_score > 0 else "low"
                    row_reasons.append(f"{feature_name} {direction} vs {peer_name}")
                    if z_score > 0:
                        row_high_risk_count += 1

            sources.append(source)
            outlier_counts.append(len(row_reasons))
            high_risk_counts.append(row_high_risk_count)
            scores.append(min(max(row_abs_z_scores, default=0.0) / threshold, 2.0) * 50)
            reasons.append("; ".join(row_reasons))

        result[PEER_GROUP_SOURCE_COLUMN] = sources
        result[PEER_OUTLIER_COUNT_COLUMN] = outlier_counts
        result[PEER_HIGH_RISK_OUTLIER_COUNT_COLUMN] = high_risk_counts
        result[PEER_OUTLIER_SCORE_COLUMN] = scores
        result[PEER_OUTLIER_REASONS_COLUMN] = reasons


def summarize_stage_b(scored: pd.DataFrame, target_col: str = "defaulted") -> pd.DataFrame:
    summary = (
        scored.groupby(BENCHMARK_BAND_COLUMN, dropna=False)
        .agg(
            rows=("transaction_id", "size"),
            avg_benchmark_score=(BENCHMARK_SCORE_COLUMN, "mean"),
            avg_peer_outlier_score=(PEER_OUTLIER_SCORE_COLUMN, "mean"),
            peer_outlier_rate=(PEER_OUTLIER_COUNT_COLUMN, lambda s: (s > 0).mean()),
            high_risk_peer_outlier_rate=(
                PEER_HIGH_RISK_OUTLIER_COUNT_COLUMN,
                lambda s: (s > 0).mean(),
            ),
        )
        .reset_index()
    )
    summary["row_pct"] = summary["rows"] / len(scored) if len(scored) else np.nan

    if target_col in scored.columns and scored[target_col].notna().any():
        resolved = scored[scored[target_col].notna()].copy()
        resolved[target_col] = resolved[target_col].astype(int)
        outcome = (
            resolved.groupby(BENCHMARK_BAND_COLUMN, dropna=False)
            .agg(defaults=(target_col, "sum"), default_rate=(target_col, "mean"))
            .reset_index()
        )
        summary = summary.merge(outcome, on=BENCHMARK_BAND_COLUMN, how="left")

    order = {"Low": 0, "Medium": 1, "High": 2}
    return (
        summary.assign(_order=summary[BENCHMARK_BAND_COLUMN].map(order).fillna(99))
        .sort_values("_order")
        .drop(columns="_order")
    )


def _validate_config(config: dict[str, Any]) -> None:
    if "egp_per_usd_assumption" not in config:
        raise ValueError("Benchmark config must include egp_per_usd_assumption.")
    if "features" not in config or not isinstance(config["features"], list):
        raise ValueError("Benchmark config must contain a list under 'features'.")
    if not config["features"]:
        raise ValueError("At least one benchmark feature is required.")
    names = [feature.get("name") for feature in config["features"]]
    if any(not name for name in names):
        raise ValueError("Every benchmark feature must have a name.")
    if len(names) != len(set(names)):
        raise ValueError("Benchmark feature names must be unique.")
    if "peer_outliers" not in config:
        raise ValueError("Benchmark config must include peer_outliers settings.")
    for key in ["group_column", "minimum_peer_group_size", "robust_z_threshold"]:
        if key not in config["peer_outliers"]:
            raise ValueError(f"Benchmark config peer_outliers missing key: {key}")


def _percentile_rank(values: pd.Series, sorted_training_values: np.ndarray) -> pd.Series:
    clean = values.replace([np.inf, -np.inf], np.nan)
    ranks = np.searchsorted(sorted_training_values, clean.to_numpy(), side="right")
    percentiles = ranks / len(sorted_training_values)
    return pd.Series(percentiles, index=values.index).where(clean.notna(), np.nan)


def _robust_stats(values: np.ndarray) -> dict[str, float]:
    clean = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    if mad == 0:
        mad = float(np.std(clean))
    if mad == 0:
        mad = 1.0
    return {"median": median, "mad": mad}


def _robust_z_score(value: float, median: float, mad: float) -> float:
    if pd.isna(value):
        return np.nan
    return 0.6745 * (float(value) - median) / mad


def _log1p_nonnegative(series: pd.Series) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    return np.log1p(clean.clip(lower=0))


def _score_band(score: float, config: dict[str, Any]) -> str:
    if pd.isna(score):
        return "Unknown"
    low_max = config["benchmark_score"]["low_risk_max"]
    medium_max = config["benchmark_score"]["medium_risk_max"]
    if score <= low_max:
        return "Low"
    if score <= medium_max:
        return "Medium"
    return "High"


def _group_key(value: Any) -> str:
    return "<NA>" if pd.isna(value) else str(value)
