from pathlib import Path
from collections import Counter
import json
import shutil

import numpy as np
import pandas as pd


# ============================================================
# FOLDERS
# ============================================================


def get_project_dir() -> Path:
    base = Path.cwd()
    if (base / "raw data").exists():
        return base
    if (base.parent / "raw data").exists():
        return base.parent
    return base


PROJECT_DIR = get_project_dir()
RAW_DIR = PROJECT_DIR / "raw data"

BRONZE_DIR = PROJECT_DIR / "processed/bronze"
SILVER_DIR = PROJECT_DIR / "processed/silver"
REJECTED_DIR = PROJECT_DIR / "processed/rejected"
ANOMALY_DIR = PROJECT_DIR / "processed/anomalies"
SUMMARY_DIR = PROJECT_DIR / "summaries"

for folder in [BRONZE_DIR, SILVER_DIR, REJECTED_DIR, ANOMALY_DIR, SUMMARY_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

NULL_TOKENS = {
    "",
    " ",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "-",
    "--",
    "?",
    "missing"
}

# Sri Lanka rough coordinate boundary
LAT_MIN, LAT_MAX = 5.0, 10.5
LON_MIN, LON_MAX = 79.0, 82.5

EXPECTED_TRANSACTION_YEARS = {2023, 2024, 2025}
EXPECTED_SEASONALITY_YEARS = {2023, 2024, 2025, 2026}

VALID_OUTLET_SIZES = {"Small", "Medium", "Large", "Extra Large"}
VALID_SEASONALITY = {"Favorable", "Moderate", "Un-Favorable"}
VALID_HOLIDAY_TYPES = {"Public", "Bank", "Mercantile", "Poya Day"}

PRICE_MIN_RECORDS_PER_SKU = 20


# ============================================================
# COMMON HELPERS
# ============================================================

def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, engine="python", low_memory=False)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    string_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in string_cols:
        s = df[col].astype("string").str.strip()
        s = s.str.replace(r"\s+", " ", regex=True)

        missing_mask = s.str.lower().isin(NULL_TOKENS)
        df[col] = s.mask(missing_mask, pd.NA)

    return df


def standardize_outlet_size(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )

    cleaned = cleaned.replace({
        "Extra large": "Extra Large",
        "Nan": pd.NA,
        "None": pd.NA,
        "Null": pd.NA,
        "N/A": pd.NA,
        "Na": pd.NA
    })
