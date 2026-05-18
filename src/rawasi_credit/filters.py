from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RULE_FLAG_PREFIX = "stage_a_rule_"
REJECTED_COLUMN = "stage_a_rejected"
DECISION_COLUMN = "stage_a_decision"
REASONS_COLUMN = "stage_a_reject_reasons"
RULE_COUNT_COLUMN = "stage_a_reject_rule_count"


def load_hard_filter_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        config = json.load(f)
    _validate_config(config)
    return config


def apply_hard_filters(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply Stage A automatic-reject rules to a transaction dataframe."""
    _validate_config(config)

    result = df.copy()
    enabled_rules = [rule for rule in config["rules"] if rule.get("enabled", True)]

    reason_lists = pd.Series([[] for _ in range(len(result))], index=result.index, dtype=object)
    rejected = pd.Series(False, index=result.index)

    for rule in enabled_rules:
        mask = _evaluate_rule(result, rule)
        flag_col = f"{RULE_FLAG_PREFIX}{rule['name']}"
        result[flag_col] = mask
        rejected = rejected | mask

        reason = rule.get("reject_reason", rule["name"])
        reason_lists.loc[mask] = reason_lists.loc[mask].apply(lambda reasons: reasons + [reason])

    labels = config.get("decision_labels", {})
    pass_label = labels.get("pass", "PASS_STAGE_A")
    reject_label = labels.get("reject", "REJECT_STAGE_A")

    result[REJECTED_COLUMN] = rejected
    result[DECISION_COLUMN] = np.where(rejected, reject_label, pass_label)
    result[REASONS_COLUMN] = reason_lists.apply("; ".join)
    result[RULE_COUNT_COLUMN] = reason_lists.apply(len)

    return result


def summarize_stage_a(
    scored: pd.DataFrame,
    config: dict[str, Any],
    target_col: str = "defaulted",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return decision-level and rule-level summaries."""
    summary = (
        scored.groupby(DECISION_COLUMN, dropna=False)
        .agg(rows=("transaction_id", "size"))
        .reset_index()
    )
    summary["row_pct"] = summary["rows"] / len(scored)

    if target_col in scored.columns and scored[target_col].notna().any():
        resolved = scored[scored[target_col].notna()].copy()
        resolved[target_col] = resolved[target_col].astype(int)
        outcome = (
            resolved.groupby(DECISION_COLUMN, dropna=False)
            .agg(defaults=(target_col, "sum"), default_rate=(target_col, "mean"))
            .reset_index()
        )
        summary = summary.merge(outcome, on=DECISION_COLUMN, how="left")

    rule_rows = []
    for rule in config["rules"]:
        if not rule.get("enabled", True):
            continue
        flag_col = f"{RULE_FLAG_PREFIX}{rule['name']}"
        if flag_col not in scored.columns:
            continue

        hit = scored[flag_col].fillna(False)
        row = {
            "rule": rule["name"],
            "description": rule.get("description", ""),
            "reject_reason": rule.get("reject_reason", rule["name"]),
            "hit_rows": int(hit.sum()),
            "hit_rate": float(hit.mean()),
        }

        if target_col in scored.columns and scored[target_col].notna().any():
            resolved = scored[scored[target_col].notna()].copy()
            resolved[target_col] = resolved[target_col].astype(int)
            resolved_hit = resolved[flag_col].fillna(False)
            hit_subset = resolved[resolved_hit]
            row["resolved_hit_rows"] = int(resolved_hit.sum())
            row["defaults_captured"] = int(hit_subset[target_col].sum())
            row["default_rate_if_hit"] = (
                float(hit_subset[target_col].mean()) if len(hit_subset) else np.nan
            )

        rule_rows.append(row)

    rule_summary = pd.DataFrame(rule_rows).sort_values("hit_rows", ascending=False)
    return summary, rule_summary


def _validate_config(config: dict[str, Any]) -> None:
    if "rules" not in config or not isinstance(config["rules"], list):
        raise ValueError("Hard-filter config must contain a list under 'rules'.")
    names = [rule.get("name") for rule in config["rules"]]
    if any(not name for name in names):
        raise ValueError("Every hard-filter rule must have a non-empty name.")
    if len(names) != len(set(names)):
        raise ValueError("Hard-filter rule names must be unique.")
    for rule in config["rules"]:
        for required in ["column", "operator", "value"]:
            if required not in rule:
                raise ValueError(f"Rule {rule['name']} missing required key: {required}")


def _evaluate_rule(df: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    column = rule["column"]
    if column not in df.columns:
        raise KeyError(f"Rule {rule['name']} references missing column: {column}")

    series = df[column]
    operator = rule["operator"]
    value = rule["value"]

    if operator == "equals":
        mask = series.eq(value)
    elif operator == "not_equals":
        mask = series.ne(value)
    elif operator == "in":
        mask = series.isin(value)
    elif operator == "not_in":
        mask = ~series.isin(value)
    elif operator == "less_than":
        mask = series < value
    elif operator == "less_equal":
        mask = series <= value
    elif operator == "greater_than":
        mask = series > value
    elif operator == "greater_equal":
        mask = series >= value
    else:
        raise ValueError(f"Unsupported operator for rule {rule['name']}: {operator}")

    if rule.get("missing_values_fail", False):
        mask = mask | series.isna()
    else:
        mask = mask & series.notna()

    return mask.fillna(False)
