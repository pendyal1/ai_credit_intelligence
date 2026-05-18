from __future__ import annotations

from pathlib import Path

import pandas as pd


DATE_COLUMNS = ["invoice_date", "expected_payment_date", "actual_resolution_date"]
DEFAULT_SPLIT_DATE = pd.Timestamp("2025-10-01")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_dataset_path(root: Path | None = None) -> Path:
    """Find the Rawasi synthetic dataset without depending on its exact filename."""
    root = root or project_root()
    candidates = sorted((root / "utils").glob("*synthetic*.csv*"))
    if not candidates:
        candidates = sorted((root / "utils").glob("*.csv*"))
    if not candidates:
        raise FileNotFoundError("No CSV dataset found under utils/.")
    if len(candidates) > 1:
        names = ", ".join(str(path) for path in candidates)
        raise FileExistsError(f"Multiple dataset candidates found; pass --data explicitly: {names}")
    return candidates[0]


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path else find_dataset_path()
    return pd.read_csv(dataset_path, parse_dates=DATE_COLUMNS)


def split_resolved_open(
    df: pd.DataFrame,
    split_date: pd.Timestamp = DEFAULT_SPLIT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and open holdout frames using the case-study split."""
    resolved = df[df["defaulted"].notna()].copy()
    resolved["defaulted"] = resolved["defaulted"].astype(int)

    train = resolved[resolved["invoice_date"] < split_date].copy()
    validation = resolved[resolved["invoice_date"] >= split_date].copy()
    open_holdout = df[df["defaulted"].isna()].copy()

    return train, validation, open_holdout
