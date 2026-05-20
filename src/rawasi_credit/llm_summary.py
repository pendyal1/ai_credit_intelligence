from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd
import requests


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_deal_context(
    transaction_id: str,
    monitoring_frame: pd.DataFrame,
    anomalies: pd.DataFrame | None = None,
    clusters: pd.DataFrame | None = None,
    drift_report: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Create a compact, reviewer-oriented context payload for one deal."""

    if "transaction_id" not in monitoring_frame.columns:
        raise KeyError("monitoring_frame missing required column: transaction_id")

    matches = monitoring_frame[monitoring_frame["transaction_id"].eq(transaction_id)]
    if matches.empty:
        raise ValueError(f"Transaction not found in monitoring frame: {transaction_id}")

    row = matches.iloc[0]
    anomaly_row = _first_match(anomalies, transaction_id)
    relevant_clusters = _matching_alert_clusters(row, clusters)
    significant_shifts = _significant_shifts(drift_report)

    return {
        "transaction_id": transaction_id,
        "deal_snapshot": _pick(
            row,
            [
                "invoice_date",
                "stage_b_split",
                "invoice_amount_usd",
                "funded_amount_usd",
                "invoice_currency",
                "payment_terms_days",
                "advance_pct",
                "insurance_covered",
                "goods_category",
                "exporter_id",
                "buyer_id",
                "buy_buyer_country",
                "buy_buyer_sector",
                "buy_creditsafe_rating",
                "buy_buyer_country_risk_score",
                "buy_buyer_payment_history_dso_avg",
                "exp_industry",
                "exp_segment",
                "exp_years_in_business",
                "exp_iscore",
                "exp_current_ratio",
                "exp_quick_ratio",
                "exp_debt_to_equity",
                "exp_operating_margin_pct",
                "exp_historical_on_time_payment_rate",
                "exp_historical_dispute_rate",
                "exp_concentration_top1_buyer_pct",
                "cumulative_outstanding_usd",
                "recent_velocity_count",
            ],
        ),
        "layer1_signals": _pick(
            row,
            [
                "stage_a_decision",
                "stage_b_benchmark_score",
                "stage_b_benchmark_band",
                "stage_b_peer_high_risk_outlier_count",
                "stage_b_peer_outlier_score",
                "stage_b_peer_outlier_reasons",
                "stage_c_default_probability",
                "stage_c_predicted_default",
                "stage_c_top_positive_shap_reasons",
                "stage_c_top_negative_shap_reasons",
                "stage_c_shap_margin",
            ],
        ),
        "stage_b_derived_risk_features": _pick(
            row,
            [
                "ticket_to_revenue_est",
                "outstanding_to_revenue_est",
                "funded_to_revenue_est",
                "total_exposure_to_revenue_est",
                "advance_tenor_exposure",
                "buyer_dso_to_terms",
                "buyer_terms_uninsured_pressure",
                "exposure_quality_pressure",
                "loss_given_default_proxy",
                "trade_history_pressure",
                "profitability_pressure",
                "liquidity_pressure",
            ],
        ),
        "layer2_anomaly_signals": _pick(
            anomaly_row,
            [
                "layer2_anomaly_alert",
                "layer2_anomaly_score",
                "probability_percentile_vs_train",
                "benchmark_percentile_vs_train",
                "peer_score_percentile_vs_train",
                "peer_count_percentile_vs_train",
                "layer2_anomaly_reasons",
            ],
        ),
        "matching_alert_clusters": relevant_clusters,
        "significant_distribution_shifts": significant_shifts,
    }


def build_deal_summary_messages(deal_context: dict[str, Any]) -> list[dict[str, str]]:
    """Build the exact prompt messages used for Task 2B."""

    system_prompt = (
        "You are a senior credit risk analyst supporting Rawasi, an SMB invoice "
        "discounting platform. Use only the provided structured data. Do not invent "
        "facts. Write for a human credit officer, not a data scientist. Return strict "
        "JSON only."
    )
    user_prompt = f"""
Create a structured credit summary for the deal below.

Required JSON schema:
{{
  "risk_verdict": "Approve | Refer | Reject",
  "confidence": "Low | Medium | High",
  "one_sentence_verdict": "One concise sentence with the verdict and why.",
  "top_three_factors": [
    {{"factor": "Plain-English factor", "evidence": "Specific value or signal", "credit_implication": "Why it matters"}},
    {{"factor": "Plain-English factor", "evidence": "Specific value or signal", "credit_implication": "Why it matters"}},
    {{"factor": "Plain-English factor", "evidence": "Specific value or signal", "credit_implication": "Why it matters"}}
  ],
  "unusual_flags_not_fully_captured_by_model": [
    {{"flag": "Unusual pattern", "why_it_needs_human_review": "Short explanation"}}
  ],
  "recommended_reviewer_actions": ["Action 1", "Action 2", "Action 3"]
}}

Decision guidance:
- Approve only when Layer 1 and Layer 2 signals are both low risk.
- Refer when the deal is not an automatic reject but needs human review.
- Reject when risk is extreme, policy-breaking, or strongly default-like.
- Prioritize catching default risk over minimizing false positives.
- If Stage C predicted default is 1, the verdict should usually be Refer or Reject.
- Include one unusual flag that adds value beyond restating the model probability.

Deal context:
{json.dumps(_json_ready(deal_context), indent=2)}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_openrouter_deal_summary(
    deal_context: dict[str, Any],
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 900,
    timeout: int = 60,
) -> dict[str, Any]:
    """
    Call OpenRouter's OpenAI-compatible chat completions endpoint.

    Set OPENROUTER_API_KEY in the environment before calling this function.
    Optional environment variables:
    - OPENROUTER_MODEL
    - OPENROUTER_SITE_URL
    - OPENROUTER_APP_TITLE
    """

    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY before calling OpenRouter.")

    model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    site_url = os.getenv("OPENROUTER_SITE_URL")
    app_title = os.getenv("OPENROUTER_APP_TITLE", "Rawasi Credit Intelligence")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_title:
        headers["X-Title"] = app_title

    payload = {
        "model": model,
        "messages": build_deal_summary_messages(deal_context),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        OPENROUTER_CHAT_COMPLETIONS_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]

    return {
        "model": body.get("model", model),
        "usage": body.get("usage", {}),
        "raw_content": content,
        "parsed_summary": parse_llm_json(content),
    }


def parse_llm_json(content: str) -> dict[str, Any]:
    """Parse JSON returned by the LLM, including fenced-code fallbacks."""

    cleaned = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    return json.loads(cleaned)


def _first_match(frame: pd.DataFrame | None, transaction_id: str) -> pd.Series | None:
    if frame is None or frame.empty or "transaction_id" not in frame.columns:
        return None
    matches = frame[frame["transaction_id"].eq(transaction_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _matching_alert_clusters(row: pd.Series, clusters: pd.DataFrame | None) -> list[dict[str, Any]]:
    if clusters is None or clusters.empty or "alert_flag" not in clusters.columns:
        return []

    alerts = clusters[clusters["alert_flag"].astype(bool)].copy()
    matches = []
    for _, cluster in alerts.iterrows():
        group_cols = str(cluster.get("cluster_definition", "")).split(" + ")
        if not group_cols or any(col not in row.index for col in group_cols):
            continue
        if all(str(row.get(col)) == str(cluster.get(col)) for col in group_cols):
            matches.append(
                _json_ready(
                    {
                        "cluster_definition": cluster.get("cluster_definition"),
                        "cluster_key": cluster.get("cluster_key"),
                        "recent_rows": cluster.get("recent_rows"),
                        "recent_defaults": cluster.get("recent_defaults"),
                        "recent_default_rate": cluster.get("recent_default_rate"),
                        "train_default_rate": cluster.get("train_default_rate"),
                        "default_rate_lift": cluster.get("default_rate_lift"),
                        "model_underestimated_recent_risk": cluster.get(
                            "model_underestimated_recent_risk"
                        ),
                    }
                )
            )
    return matches[:5]


def _significant_shifts(drift_report: pd.DataFrame | None) -> list[dict[str, Any]]:
    if drift_report is None or drift_report.empty or "human_review_required" not in drift_report.columns:
        return []
    shifts = drift_report[drift_report["human_review_required"].astype(bool)].head(8)
    return _json_ready(
        shifts[
            [
                col
                for col in [
                    "feature",
                    "psi",
                    "severity",
                    "baseline_mean",
                    "incoming_mean",
                ]
                if col in shifts.columns
            ]
        ].to_dict(orient="records")
    )


def _pick(row: pd.Series | None, columns: list[str]) -> dict[str, Any]:
    if row is None:
        return {}
    return _json_ready({col: row[col] for col in columns if col in row.index})


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
