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

from rawasi_credit.modeling import stage_c_feature_frame  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the saved Stage C model and score one Stage-B-engineered transaction. "
            "Run Stage B/C first if reports or model artifacts are missing."
        )
    )
    parser.add_argument(
        "--transaction-id",
        default=None,
        help="Transaction ID to score. Defaults to the highest-risk open-holdout deal.",
    )
    parser.add_argument(
        "--stage-b-scored",
        type=Path,
        default=REPORTS_DIR / "stage_b_scored_transactions.csv",
        help="Stage B scored transactions CSV.",
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=MODELS_DIR / "stage_c_xgboost_model.joblib",
        help="Saved Stage C model artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_exists(args.stage_b_scored, "Stage B scored transactions")
    _ensure_exists(args.model_artifact, "Stage C model artifact")

    stage_b = pd.read_csv(args.stage_b_scored, parse_dates=["invoice_date"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*serialized model.*", category=UserWarning)
        artifact = joblib.load(args.model_artifact)

    feature_columns = artifact["feature_columns"]
    model = artifact["model"]
    threshold = artifact["threshold"]

    if args.transaction_id:
        candidates = stage_b[stage_b["transaction_id"].eq(args.transaction_id)].copy()
        if candidates.empty:
            raise ValueError(f"Transaction not found: {args.transaction_id}")
    else:
        candidates = stage_b[stage_b["stage_b_split"].eq("open_holdout")].copy()
        if candidates.empty:
            candidates = stage_b.copy()

    x = stage_c_feature_frame(candidates, feature_columns)
    probabilities = model.predict_proba(x)[:, 1]
    scored = candidates.copy()
    scored["stage_c_default_probability"] = probabilities
    scored["stage_c_predicted_default"] = (
        scored["stage_c_default_probability"] >= threshold
    ).astype(int)
    scored = _add_shap_reason_columns(scored, model, x)
    scored = scored.sort_values("stage_c_default_probability", ascending=False)

    output_cols = [
        "transaction_id",
        "invoice_date",
        "stage_b_split",
        "invoice_amount_usd",
        "invoice_currency",
        "goods_category",
        "buy_buyer_country",
        "stage_b_benchmark_score",
        "stage_b_benchmark_band",
        "stage_b_peer_high_risk_outlier_count",
        "stage_b_peer_outlier_reasons",
        "stage_c_default_probability",
        "stage_c_predicted_default",
        "stage_c_top_positive_shap_reasons",
        "stage_c_top_negative_shap_reasons",
    ]
    output_cols = [col for col in output_cols if col in scored.columns]

    print(scored[output_cols].head(1).to_string(index=False))


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _add_shap_reason_columns(scored: pd.DataFrame, model, feature_frame: pd.DataFrame) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for one-transaction explanations. Install it with "
            "`python -m pip install shap==0.44.1` in this environment."
        ) from exc

    out = scored.copy()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_frame)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
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


if __name__ == "__main__":
    main()
