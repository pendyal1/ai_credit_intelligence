#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rawasi_credit.benchmarks import StageBBenchmarkTransformer, load_benchmark_config  # noqa: E402
from rawasi_credit.data import find_dataset_path, load_dataset, split_resolved_open  # noqa: E402
from rawasi_credit.filters import DECISION_COLUMN, apply_hard_filters, load_hard_filter_config  # noqa: E402
from rawasi_credit.modeling import (  # noqa: E402
    calibration_table,
    classification_metrics,
    feature_importance_frame,
    load_model_config,
    stage_c_feature_frame,
    threshold_analysis_table,
    threshold_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rawasi Stage C XGBoost classifier.")
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
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "config" / "xgboost_model.json",
        help="Path to Stage C model JSON config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Directory for generated Stage C reports.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory for saved model artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.data or find_dataset_path(PROJECT_ROOT)

    hard_filter_config = load_hard_filter_config(args.hard_filter_config)
    benchmark_config = load_benchmark_config(args.benchmark_config)
    model_config = load_model_config(args.model_config)

    stage_b_train, stage_b_validation, stage_b_open = _build_stage_b_frames(
        dataset_path=dataset_path,
        hard_filter_config=hard_filter_config,
        benchmark_config=benchmark_config,
    )

    feature_columns = model_config["features"]
    train_sorted = stage_b_train.sort_values("invoice_date").copy()
    validation_sorted = stage_b_validation.sort_values("invoice_date").copy()

    x_train = stage_c_feature_frame(train_sorted, feature_columns)
    y_train = train_sorted["defaulted"].astype(int)
    x_validation = stage_c_feature_frame(validation_sorted, feature_columns)
    y_validation = validation_sorted["defaulted"].astype(int)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=model_config.get("random_state", 42),
        scale_pos_weight=scale_pos_weight,
        n_jobs=1,
    )

    cv = TimeSeriesSplit(n_splits=model_config["grid_search"]["cv_splits"])
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=model_config["grid_search"]["param_grid"],
        scoring=model_config["grid_search"]["scoring"],
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    train_prob = best_model.predict_proba(x_train)[:, 1]
    validation_prob = best_model.predict_proba(x_validation)[:, 1]
    threshold = threshold_from_config(train_prob, model_config)

    train_metrics = classification_metrics(y_train, train_prob, threshold)
    validation_metrics = classification_metrics(y_validation, validation_prob, threshold)

    validation_predictions = validation_sorted[
        [
            "transaction_id",
            "invoice_date",
            "status",
            "defaulted",
            "stage_b_benchmark_score",
            "stage_b_benchmark_band",
            "stage_b_peer_outlier_count",
            "stage_b_peer_outlier_reasons",
        ]
    ].copy()
    validation_predictions["stage_c_default_probability"] = validation_prob
    validation_predictions["stage_c_predicted_default"] = (
        validation_predictions["stage_c_default_probability"] >= threshold
    ).astype(int)
    open_predictions = _score_open_holdout(best_model, stage_b_open, feature_columns)
    feature_importance = feature_importance_frame(best_model, feature_columns)
    shap_importance, validation_shap_explanations = _build_shap_outputs(
        model=best_model,
        feature_frame=x_validation,
        source_frame=validation_sorted,
        probabilities=validation_prob,
        predictions=validation_predictions["stage_c_predicted_default"],
        feature_importance=feature_importance,
    )
    validation_predictions = validation_predictions.merge(
        validation_shap_explanations[
            [
                "transaction_id",
                "stage_c_top_positive_shap_reasons",
                "stage_c_top_negative_shap_reasons",
                "stage_c_shap_margin",
            ]
        ],
        on="transaction_id",
        how="left",
    )
    layer1_validation_predictions = _build_layer1_validation_predictions(
        dataset_path,
        hard_filter_config,
        validation_predictions,
    )
    layer1_metrics = _binary_flag_metrics(
        layer1_validation_predictions,
        flag_col="layer1_flagged_for_review",
    )
    cv_results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
    calibration = calibration_table(y_validation, validation_prob, bins=5)
    threshold_analysis = threshold_analysis_table(y_validation, validation_prob)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    validation_predictions.to_csv(args.output_dir / "stage_c_validation_predictions.csv", index=False)
    layer1_validation_predictions.to_csv(
        args.output_dir / "layer1_validation_predictions.csv",
        index=False,
    )
    layer1_metrics.to_csv(args.output_dir / "layer1_validation_metrics.csv", index=False)
    open_predictions.to_csv(args.output_dir / "stage_c_open_predictions.csv", index=False)
    feature_importance.to_csv(args.output_dir / "stage_c_feature_importance.csv", index=False)
    shap_importance.to_csv(args.output_dir / "stage_c_shap_feature_importance.csv", index=False)
    validation_shap_explanations.to_csv(
        args.output_dir / "stage_c_validation_shap_explanations.csv",
        index=False,
    )
    cv_results.to_csv(args.output_dir / "stage_c_gridsearch_results.csv", index=False)
    calibration.to_csv(args.output_dir / "stage_c_calibration_table.csv", index=False)
    threshold_analysis.to_csv(args.output_dir / "stage_c_threshold_analysis.csv", index=False)

    _plot_calibration(calibration, args.output_dir / "stage_c_calibration.png")
    _plot_pr_curve(y_validation, validation_prob, args.output_dir / "stage_c_precision_recall.png")
    _plot_confusion_matrix(
        y_validation,
        validation_predictions["stage_c_predicted_default"],
        args.output_dir / "stage_c_confusion_matrix.png",
    )
    _plot_feature_importance(feature_importance, args.output_dir / "stage_c_feature_importance.png")
    _plot_shap_importance(shap_importance, args.output_dir / "stage_c_shap_feature_importance.png")

    model_artifact = {
        "model": best_model,
        "feature_columns": feature_columns,
        "threshold": threshold,
        "model_config": model_config,
        "benchmark_config": benchmark_config,
        "hard_filter_config": hard_filter_config,
    }
    joblib.dump(model_artifact, args.model_dir / "stage_c_xgboost_model.joblib")

    metrics = pd.DataFrame(
        [
            {"split": "train", **train_metrics},
            {"split": "validation", **validation_metrics},
        ]
    )
    metrics.to_csv(args.output_dir / "stage_c_metrics.csv", index=False)

    report = _render_report(
        dataset_path=_display_path(dataset_path),
        hard_filter_config_path=_display_path(args.hard_filter_config),
        benchmark_config_path=_display_path(args.benchmark_config),
        model_config_path=_display_path(args.model_config),
        scale_pos_weight=scale_pos_weight,
        best_params=grid.best_params_,
        best_cv_score=grid.best_score_,
        threshold=threshold,
        threshold_rationale=model_config["threshold"].get("rationale", ""),
        metrics=metrics,
        layer1_metrics=layer1_metrics,
        calibration=calibration,
        threshold_analysis=threshold_analysis,
        feature_importance=feature_importance,
        shap_importance=shap_importance,
    )
    (args.output_dir / "stage_c_report.md").write_text(report)

    print("Stage C XGBoost model complete.")
    print(f"Training rows: {len(x_train):,}; defaults: {int(y_train.sum()):,}")
    print(f"Validation rows: {len(x_validation):,}; defaults: {int(y_validation.sum()):,}")
    print(f"Best CV {model_config['grid_search']['scoring']}: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")
    print(f"Threshold: {threshold:.6f}")
    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print("\nFull Layer 1 validation metrics:")
    print(layer1_metrics.to_string(index=False))
    print(f"\nWrote: {args.output_dir / 'stage_c_report.md'}")
    print(f"Wrote: {args.model_dir / 'stage_c_xgboost_model.joblib'}")


def _build_stage_b_frames(
    *,
    dataset_path: Path,
    hard_filter_config: dict,
    benchmark_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_dataset(dataset_path)
    stage_a_scored = apply_hard_filters(df, hard_filter_config)
    train, validation, open_holdout = split_resolved_open(stage_a_scored)

    train_pass = train[train[DECISION_COLUMN].eq("PASS_STAGE_A")].copy()
    validation_pass = validation[validation[DECISION_COLUMN].eq("PASS_STAGE_A")].copy()
    open_pass = open_holdout[open_holdout[DECISION_COLUMN].eq("PASS_STAGE_A")].copy()

    transformer = StageBBenchmarkTransformer(benchmark_config).fit(train_pass)
    stage_b_train = transformer.transform(train_pass)
    stage_b_train["stage_b_split"] = "train"
    stage_b_validation = transformer.transform(validation_pass)
    stage_b_validation["stage_b_split"] = "validation"
    stage_b_open = transformer.transform(open_pass)
    stage_b_open["stage_b_split"] = "open_holdout"
    return stage_b_train, stage_b_validation, stage_b_open


def _build_layer1_validation_predictions(
    dataset_path: Path,
    hard_filter_config: dict,
    validation_predictions: pd.DataFrame,
) -> pd.DataFrame:
    stage_a_scored = apply_hard_filters(load_dataset(dataset_path), hard_filter_config)
    _, validation, _ = split_resolved_open(stage_a_scored)
    output = validation[
        [
            "transaction_id",
            "invoice_date",
            "status",
            "defaulted",
            DECISION_COLUMN,
            "stage_a_reject_reasons",
        ]
    ].copy()
    prediction_columns = [
        "transaction_id",
        "stage_b_benchmark_score",
        "stage_b_benchmark_band",
        "stage_b_peer_outlier_count",
        "stage_c_default_probability",
        "stage_c_predicted_default",
    ]
    prediction_columns.extend(
        col
        for col in [
            "stage_c_top_positive_shap_reasons",
            "stage_c_top_negative_shap_reasons",
            "stage_c_shap_margin",
        ]
        if col in validation_predictions.columns
    )
    output = output.merge(
        validation_predictions[prediction_columns],
        on="transaction_id",
        how="left",
    )
    output["stage_c_predicted_default"] = (
        output["stage_c_predicted_default"].fillna(0).astype(int)
    )
    output["layer1_flagged_for_review"] = (
        output[DECISION_COLUMN].eq("REJECT_STAGE_A")
        | output["stage_c_predicted_default"].eq(1)
    ).astype(int)
    output["layer1_route"] = np.select(
        [
            output[DECISION_COLUMN].eq("REJECT_STAGE_A"),
            output["stage_c_predicted_default"].eq(1),
        ],
        ["Stage A reject", "Stage C high-risk review"],
        default="Pass Layer 1",
    )
    return output.sort_values(["layer1_flagged_for_review", "invoice_date"], ascending=[False, True])


def _binary_flag_metrics(frame: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    y_true = frame["defaulted"].astype(int)
    y_pred = frame[flag_col].astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    flagged = int(y_pred.sum())
    return pd.DataFrame(
        [
            {
                "scope": "full_layer1_validation",
                "rows": int(len(frame)),
                "defaults": int(y_true.sum()),
                "flagged_rows": flagged,
                "flagged_rate": float(y_pred.mean()),
                "precision": tp / flagged if flagged else 0.0,
                "recall": tp / int(y_true.sum()) if int(y_true.sum()) else 0.0,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            }
        ]
    )


def _score_open_holdout(
    model: XGBClassifier,
    stage_b_open: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    if len(stage_b_open) == 0:
        return pd.DataFrame()
    x_open = stage_c_feature_frame(stage_b_open, feature_columns)
    probabilities = model.predict_proba(x_open)[:, 1]
    output = stage_b_open[
        [
            "transaction_id",
            "invoice_date",
            "status",
            "stage_b_benchmark_score",
            "stage_b_benchmark_band",
            "stage_b_peer_outlier_count",
            "stage_b_peer_outlier_reasons",
        ]
    ].copy()
    output["stage_c_default_probability"] = probabilities
    return output.sort_values("stage_c_default_probability", ascending=False)


def _plot_calibration(calibration: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(
        calibration["avg_predicted_probability"],
        calibration["observed_default_rate"],
        marker="o",
        linewidth=2,
        label="Validation bins",
    )
    ax.set_title("Stage C Calibration - Validation")
    ax.set_xlabel("Average predicted default probability")
    ax.set_ylabel("Observed default rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_pr_curve(y_true: pd.Series, y_probability: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_true, y_probability, ax=ax)
    ax.set_title("Stage C Precision-Recall Curve - Validation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: pd.Series,
    y_predicted: pd.Series,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_predicted, labels=[0, 1], ax=ax)
    ax.set_title("Stage C Confusion Matrix - Validation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_feature_importance(feature_importance: pd.DataFrame, output_path: Path) -> None:
    top = feature_importance.head(15).sort_values("gain")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["gain"])
    ax.set_title("Stage C XGBoost Feature Importance - Gain")
    ax.set_xlabel("Gain")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_shap_outputs(
    *,
    model: XGBClassifier,
    feature_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: pd.Series,
    feature_importance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for Stage C explainability. Install it with "
            "`python -m pip install shap==0.44.1` in this environment."
        ) from exc

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_frame)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = np.asarray(expected_value).ravel()[-1]
    expected_value = float(expected_value)

    shap_importance = pd.DataFrame(
        {
            "feature": list(feature_frame.columns),
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    )
    shap_importance = shap_importance.merge(
        feature_importance[["feature", "gain", "gain_share"]],
        on="feature",
        how="left",
    ).sort_values("mean_abs_shap", ascending=False)

    explanation_rows: list[dict] = []
    for row_idx, (_, source_row) in enumerate(source_frame.iterrows()):
        row_values = shap_values[row_idx]
        feature_values = feature_frame.iloc[row_idx]
        explanation_rows.append(
            {
                "transaction_id": source_row["transaction_id"],
                "invoice_date": source_row["invoice_date"],
                "status": source_row.get("status"),
                "defaulted": source_row.get("defaulted"),
                "stage_c_default_probability": probabilities[row_idx],
                "stage_c_predicted_default": int(predictions.iloc[row_idx]),
                "stage_c_shap_base_value": expected_value,
                "stage_c_shap_margin": expected_value + float(row_values.sum()),
                "stage_c_top_positive_shap_reasons": _format_shap_reasons(
                    row_values,
                    feature_values,
                    largest_positive=True,
                ),
                "stage_c_top_negative_shap_reasons": _format_shap_reasons(
                    row_values,
                    feature_values,
                    largest_positive=False,
                ),
            }
        )

    return shap_importance, pd.DataFrame(explanation_rows)


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


def _plot_shap_importance(shap_importance: pd.DataFrame, output_path: Path) -> None:
    top = shap_importance.head(20).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_title("Stage C SHAP Feature Importance - Validation")
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_report(
    *,
    dataset_path: str,
    hard_filter_config_path: str,
    benchmark_config_path: str,
    model_config_path: str,
    scale_pos_weight: float,
    best_params: dict,
    best_cv_score: float,
    threshold: float,
    threshold_rationale: str,
    metrics: pd.DataFrame,
    layer1_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    threshold_analysis: pd.DataFrame,
    feature_importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Stage C XGBoost Model Report",
            "",
            f"Dataset: `{dataset_path}`",
            f"Stage A config: `{hard_filter_config_path}`",
            f"Stage B config: `{benchmark_config_path}`",
            f"Stage C config: `{model_config_path}`",
            "",
            "## Modeling Approach",
            "",
            "The classifier is an XGBoost gradient boosting model trained only on deals that passed Stage A. Inputs are the Stage B benchmark and peer-outlier features. Hyperparameters are tuned with `GridSearchCV` using `TimeSeriesSplit` inside the training window; the final score is evaluated once on the untouched out-of-time validation set.",
            "",
            "Class imbalance is handled with `scale_pos_weight`, computed from the Stage-A-passed training split.",
            "",
            f"- `scale_pos_weight`: `{scale_pos_weight:.4f}`",
            f"- best CV average precision: `{best_cv_score:.4f}`",
            f"- selected threshold: `{threshold:.6f}`",
            f"- best params: `{json.dumps(best_params)}`",
            "",
            "The threshold is loaded from `config/xgboost_model.json` and should be treated as the operating point for the high-risk review queue. The threshold analysis below shows the precision-recall tradeoff on the out-of-time validation set so the cutoff can be defended explicitly.",
            f"Threshold rationale: {threshold_rationale}",
            "",
            "## Metrics",
            "",
            metrics.to_markdown(index=False),
            "",
            "## Full Layer 1 Validation Metrics",
            "",
            "These metrics treat Stage A rejects and Stage C high-risk predictions as deals requiring rejection or manual review.",
            "",
            layer1_metrics.to_markdown(index=False),
            "",
            "## Calibration Table - Validation",
            "",
            calibration.to_markdown(index=False),
            "",
            "## Threshold Analysis - Validation",
            "",
            threshold_analysis.to_markdown(index=False),
            "",
            "## SHAP Feature Importance - Validation",
            "",
            "SHAP values are computed on the out-of-time validation rows. They explain the model margin/log-odds contribution, so positive values push a deal toward higher default risk and negative values push it toward lower default risk.",
            "",
            shap_importance.head(25).to_markdown(index=False),
            "",
            "## XGBoost Gain Feature Importance",
            "",
            feature_importance.to_markdown(index=False),
            "",
            "## Generated Artifacts",
            "",
            "- `reports/stage_c_metrics.csv`",
            "- `reports/stage_c_validation_predictions.csv`",
            "- `reports/layer1_validation_metrics.csv`",
            "- `reports/layer1_validation_predictions.csv`",
            "- `reports/stage_c_open_predictions.csv`",
            "- `reports/stage_c_gridsearch_results.csv`",
            "- `reports/stage_c_threshold_analysis.csv`",
            "- `reports/stage_c_calibration_table.csv`",
            "- `reports/stage_c_calibration.png`",
            "- `reports/stage_c_precision_recall.png`",
            "- `reports/stage_c_confusion_matrix.png`",
            "- `reports/stage_c_feature_importance.csv`",
            "- `reports/stage_c_feature_importance.png`",
            "- `reports/stage_c_shap_feature_importance.csv`",
            "- `reports/stage_c_shap_feature_importance.png`",
            "- `reports/stage_c_validation_shap_explanations.csv`",
            "- `models/stage_c_xgboost_model.joblib`",
        ]
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
