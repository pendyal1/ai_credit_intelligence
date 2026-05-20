from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_model_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        config = json.load(f)
    _validate_model_config(config)
    return config


def stage_c_feature_frame(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Stage C feature columns missing from dataframe: {missing}")
    return df[feature_columns].replace([np.inf, -np.inf], np.nan)


def ks_statistic(y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> float:
    frame = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)}).sort_values(
        "score", ascending=False
    )
    positives = frame["y"].sum()
    negatives = len(frame) - positives
    if positives == 0 or negatives == 0:
        return np.nan
    cum_pos = frame["y"].cumsum() / positives
    cum_neg = (1 - frame["y"]).cumsum() / negatives
    return float((cum_pos - cum_neg).abs().max())


def threshold_from_config(train_probabilities: np.ndarray, config: dict[str, Any]) -> float:
    threshold_config = config["threshold"]
    strategy = threshold_config["strategy"]
    if strategy == "top_train_risk_quantile":
        review_rate = threshold_config["review_rate"]
        if not 0 < review_rate < 1:
            raise ValueError("review_rate must be between 0 and 1.")
        return float(np.quantile(train_probabilities, 1 - review_rate))
    if strategy == "fixed_probability":
        return float(threshold_config["value"])
    raise ValueError(f"Unsupported threshold strategy: {strategy}")


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_probability)
    y_pred = (y_prob_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y_true_arr)),
        "defaults": int(y_true_arr.sum()),
        "default_rate": float(y_true_arr.mean()),
        "auc_pr": float(average_precision_score(y_true_arr, y_prob_arr)),
        "auc_roc": float(roc_auc_score(y_true_arr, y_prob_arr)),
        "ks": ks_statistic(y_true_arr, y_prob_arr),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "threshold": float(threshold),
        "predicted_positive_rate": float(y_pred.mean()),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def calibration_table(
    y_true: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray,
    bins: int = 5,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "defaulted": np.asarray(y_true).astype(int),
            "predicted_probability": np.asarray(y_probability),
        }
    )
    frame["probability_bin"] = pd.qcut(
        frame["predicted_probability"].rank(method="first"),
        q=min(bins, len(frame)),
        labels=False,
        duplicates="drop",
    )
    return (
        frame.groupby("probability_bin")
        .agg(
            rows=("defaulted", "size"),
            avg_predicted_probability=("predicted_probability", "mean"),
            observed_default_rate=("defaulted", "mean"),
            defaults=("defaulted", "sum"),
            min_predicted_probability=("predicted_probability", "min"),
            max_predicted_probability=("predicted_probability", "max"),
        )
        .reset_index()
    )


def threshold_analysis_table(
    y_true: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [
            0.05,
            0.066,
            0.1,
            0.17,
            0.2,
            0.3,
            0.4,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
        ]

    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_probability)
    rows = []
    for threshold in thresholds:
        y_pred = (y_prob_arr >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "flagged_rows": int(y_pred.sum()),
                "flagged_rate": float(y_pred.mean()),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def feature_importance_frame(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    weight_scores = booster.get_score(importance_type="weight")
    rows = []
    for feature in feature_columns:
        rows.append(
            {
                "feature": feature,
                "gain": float(gain_scores.get(feature, 0.0)),
                "weight": float(weight_scores.get(feature, 0.0)),
            }
        )
    frame = pd.DataFrame(rows)
    total_gain = frame["gain"].sum()
    frame["gain_share"] = frame["gain"] / total_gain if total_gain else 0.0
    return frame.sort_values("gain", ascending=False)


def _validate_model_config(config: dict[str, Any]) -> None:
    if "features" not in config or not isinstance(config["features"], list) or not config["features"]:
        raise ValueError("Model config must include a non-empty features list.")
    if "grid_search" not in config:
        raise ValueError("Model config must include grid_search settings.")
    for key in ["cv_splits", "scoring", "param_grid"]:
        if key not in config["grid_search"]:
            raise ValueError(f"Model config grid_search missing key: {key}")
    if "threshold" not in config or "strategy" not in config["threshold"]:
        raise ValueError("Model config must include threshold.strategy.")
