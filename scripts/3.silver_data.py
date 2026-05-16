from pathlib import Path
from collections import Counter
import json
import shutil

import numpy as np
import pandas as pd


# ============================================================
# FOLDERS
# ============================================================

RAW_DIR = Path("raw data")

BRONZE_DIR = Path("processed/bronze")
SILVER_DIR = Path("processed/silver")
REJECTED_DIR = Path("processed/rejected")
ANOMALY_DIR = Path("processed/anomalies")
SUMMARY_DIR = Path("summaries")

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

    return cleaned


def standardize_outlet_type(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )

    cleaned = cleaned.replace({
        "Grocry": "Grocery",
        "Grocerry": "Grocery",
        "Grocery ": "Grocery",
        "Bakry": "Bakery",
        "Backery": "Bakery",
        "Pharamacy": "Pharmacy",
        "Pharmcy": "Pharmacy"
    })

    return cleaned


def standardize_seasonality(series: pd.Series) -> pd.Series:
    s = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace("_", "-", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    return s.replace({
        "favorable": "Favorable",
        "favourable": "Favorable",
        "moderate": "Moderate",
        "unfavorable": "Un-Favorable",
        "un-favorable": "Un-Favorable",
        "un favourable": "Un-Favorable",
        "un favorable": "Un-Favorable"
    })


def standardize_holiday_type(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )

    return cleaned.replace({
        "Public Holiday": "Public",
        "Bank Holiday": "Bank",
        "Mercantile Holiday": "Mercantile",
        "Poya": "Poya Day",
        "Poya day": "Poya Day"
    })


def add_rejection_reason(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    mask = mask.fillna(False)

    if "_dq_rejection_reason" not in df.columns:
        df["_dq_rejection_reason"] = ""

    df.loc[mask, "_dq_rejection_reason"] = df.loc[mask, "_dq_rejection_reason"].apply(
        lambda x: reason if x == "" else f"{x} | {reason}"
    )


def is_valid_sri_lanka_coordinate(lat: pd.Series, lon: pd.Series) -> pd.Series:
    return (
        lat.between(LAT_MIN, LAT_MAX) &
        lon.between(LON_MIN, LON_MAX)
    )


def safe_sample_values(series: pd.Series, n: int = 5) -> str:
    try:
        vals = series.dropna().unique()[:n].tolist()
        return json.dumps(vals, default=str)
    except Exception:
        return "[]"


def create_column_summary(file_label: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = len(df)
    output = []

    for col in df.columns:
        s = df[col]

        output.append({
            "file": file_label,
            "column": col,
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_ratio": round(float(s.isna().mean()), 4) if rows else None,
            "unique_count": int(s.nunique(dropna=True)),
            "unique_ratio": round(float(s.nunique(dropna=True) / rows), 4) if rows else None,
            "sample_values": safe_sample_values(s)
        })

    return pd.DataFrame(output)


def create_dataset_summary(file_label: str, df: pd.DataFrame) -> dict:
    return {
        "file": file_label,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1])
    }


def find_existing_file(possible_names: list[str]) -> Path | None:
    for name in possible_names:
        path = RAW_DIR / name
        if path.exists():
            return path
    return None


def save_rejected(df: pd.DataFrame, file_name: str) -> None:
    out_path = REJECTED_DIR / file_name
    df.to_csv(out_path, index=False)


def save_anomaly(df: pd.DataFrame, file_name: str) -> None:
    out_path = ANOMALY_DIR / file_name
    df.to_csv(out_path, index=False)


# ============================================================
# BRONZE COPY
# ============================================================

def create_bronze_copies() -> None:
    for csv_file in RAW_DIR.glob("*.csv"):
        shutil.copy2(csv_file, BRONZE_DIR / csv_file.name)

    print("Bronze layer created:", BRONZE_DIR)


# ============================================================
# OUTLET MASTER SILVER
# ============================================================

def process_outlet_master() -> tuple[pd.DataFrame, dict]:
    path = find_existing_file(["outlet_master.csv", "outlet_master_final.csv"])

    if path is None:
        raise FileNotFoundError("outlet_master.csv not found.")

    raw = read_csv_safely(path)
    df = clean_column_names(raw)
    df = normalize_string_columns(df)

    df["_dq_rejection_reason"] = ""

    if "Outlet_ID" not in df.columns:
        raise ValueError("Outlet_ID column missing in outlet_master.")

    add_rejection_reason(df, df["Outlet_ID"].isna(), "Missing mandatory field: Outlet_ID")

    if "Outlet_Size" in df.columns:
        df["Outlet_Size"] = standardize_outlet_size(df["Outlet_Size"])

        invalid_size = df["Outlet_Size"].notna() & ~df["Outlet_Size"].isin(VALID_OUTLET_SIZES)

        # Do not reject the outlet only because size is wrong.
        # Set invalid size to missing; size imputation will handle it later.
        df["Outlet_Size_Invalid_Flag"] = invalid_size.astype(int)
        df.loc[invalid_size, "Outlet_Size"] = pd.NA
    else:
        df["Outlet_Size"] = pd.NA
        df["Outlet_Size_Invalid_Flag"] = 0

    df["Outlet_Size_Missing_Flag"] = df["Outlet_Size"].isna().astype(int)

    if "Outlet_Type" in df.columns:
        df["Outlet_Type"] = standardize_outlet_type(df["Outlet_Type"])

    if "Cooler_Count" in df.columns:
        before = df["Cooler_Count"]
        df["Cooler_Count"] = pd.to_numeric(df["Cooler_Count"], errors="coerce")

        invalid_cooler = before.notna() & df["Cooler_Count"].isna()
        negative_cooler = df["Cooler_Count"].notna() & (df["Cooler_Count"] < 0)

        add_rejection_reason(df, invalid_cooler, "Invalid numeric value in Cooler_Count")
        add_rejection_reason(df, negative_cooler, "Negative Cooler_Count")
    else:
        df["Cooler_Count"] = np.nan

    # Duplicate Outlet_ID handling
    duplicate_keep = df.duplicated(subset=["Outlet_ID"], keep="first")
    add_rejection_reason(df, duplicate_keep, "Duplicate business key: Outlet_ID")

    rejected = df[df["_dq_rejection_reason"] != ""].copy()
    silver = df[df["_dq_rejection_reason"] == ""].copy()

    rejected.to_csv(REJECTED_DIR / "outlet_master_rejected.csv", index=False)

    silver = silver.drop(columns=["_dq_rejection_reason"], errors="ignore")
    silver.to_csv(SILVER_DIR / "outlet_master_silver.csv", index=False)

    summary = {
        "dataset": "outlet_master",
        "raw_rows": len(raw),
        "silver_rows": len(silver),
        "rejected_rows": len(rejected),
        "missing_outlet_size_rows_kept_for_imputation": int(silver["Outlet_Size"].isna().sum()),
        "invalid_outlet_size_rows_set_to_missing": int(silver["Outlet_Size_Invalid_Flag"].sum())
    }

    return silver, summary


# ============================================================
# OUTLET COORDINATES SILVER WITH LAT/LON SWAP FIX
# ============================================================

def process_outlet_coordinates() -> tuple[pd.DataFrame, dict]:
    path = find_existing_file(["outlet_coordinates.csv", "outlet_coordinates_final.csv"])

    if path is None:
        raise FileNotFoundError("outlet_coordinates.csv not found.")

    raw = read_csv_safely(path)
    df = clean_column_names(raw)
    df = normalize_string_columns(df)

    df["_dq_rejection_reason"] = ""

    required = ["Outlet_ID", "Latitude", "Longitude"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"{col} column missing in outlet_coordinates.")

    add_rejection_reason(df, df["Outlet_ID"].isna(), "Missing mandatory field: Outlet_ID")

    df["Latitude_Original"] = df["Latitude"]
    df["Longitude_Original"] = df["Longitude"]

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    add_rejection_reason(df, df["Latitude"].isna(), "Missing or invalid Latitude")
    add_rejection_reason(df, df["Longitude"].isna(), "Missing or invalid Longitude")

    original_valid = is_valid_sri_lanka_coordinate(df["Latitude"], df["Longitude"])
    swapped_valid = is_valid_sri_lanka_coordinate(df["Longitude"], df["Latitude"])

    needs_swap = (~original_valid) & swapped_valid

    df["coordinate_corrected_flag"] = needs_swap.astype(int)
    df["coordinate_correction_type"] = np.where(
        needs_swap,
        "lat_lon_swapped",
        "none"
    )

    # Correct swapped coordinates
    lat_copy = df.loc[needs_swap, "Latitude"].copy()
    df.loc[needs_swap, "Latitude"] = df.loc[needs_swap, "Longitude"]
    df.loc[needs_swap, "Longitude"] = lat_copy

    final_valid = is_valid_sri_lanka_coordinate(df["Latitude"], df["Longitude"])

    add_rejection_reason(
        df,
        df["Latitude"].notna() & df["Longitude"].notna() & (~final_valid),
        "Invalid coordinates after swap check"
    )

    duplicate_keep = df.duplicated(subset=["Outlet_ID"], keep="first")
    add_rejection_reason(df, duplicate_keep, "Duplicate business key: Outlet_ID")

    rejected = df[df["_dq_rejection_reason"] != ""].copy()
    silver = df[df["_dq_rejection_reason"] == ""].copy()

    rejected.to_csv(REJECTED_DIR / "outlet_coordinates_rejected.csv", index=False)

    correction_log = silver[silver["coordinate_corrected_flag"] == 1].copy()
    correction_log.to_csv(SUMMARY_DIR / "coordinate_corrections_log.csv", index=False)

    silver = silver.drop(columns=["_dq_rejection_reason"], errors="ignore")
    silver.to_csv(SILVER_DIR / "outlet_coordinates_silver.csv", index=False)

    summary = {
        "dataset": "outlet_coordinates",
        "raw_rows": len(raw),
        "silver_rows": len(silver),
        "rejected_rows": len(rejected),
        "lat_lon_swapped_corrected_rows": int(silver["coordinate_corrected_flag"].sum())
    }

    return silver, summary


# ============================================================
# DISTRIBUTOR SEASONALITY SILVER
# ============================================================

def process_distributor_seasonality() -> tuple[pd.DataFrame, dict]:
    path = find_existing_file([
        "distributor_seasonality.csv",
        "distributor_seasonality_details.csv",
        "distributor_seasonality_final.csv"
    ])

    if path is None:
        raise FileNotFoundError("distributor_seasonality file not found.")

    raw = read_csv_safely(path)
    df = clean_column_names(raw)
    df = normalize_string_columns(df)

    df["_dq_rejection_reason"] = ""

    required = ["Distributor_ID", "Year", "Month", "Seasonality_Index"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"{col} column missing in distributor seasonality.")

    for col in required:
        add_rejection_reason(df, df[col].isna(), f"Missing mandatory field: {col}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")

    add_rejection_reason(df, df["Year"].isna(), "Invalid numeric value in Year")
    add_rejection_reason(df, df["Month"].isna(), "Invalid numeric value in Month")
    add_rejection_reason(df, df["Year"].notna() & ~df["Year"].isin(EXPECTED_SEASONALITY_YEARS), "Year outside expected seasonality range")
    add_rejection_reason(df, df["Month"].notna() & ~df["Month"].between(1, 12), "Month outside range 1-12")

    df["Seasonality_Index"] = standardize_seasonality(df["Seasonality_Index"])

    add_rejection_reason(
        df,
        df["Seasonality_Index"].notna() & ~df["Seasonality_Index"].isin(VALID_SEASONALITY),
        "Invalid Seasonality_Index category"
    )

    duplicate_keep = df.duplicated(subset=["Distributor_ID", "Year", "Month"], keep="first")
    add_rejection_reason(df, duplicate_keep, "Duplicate business key: Distributor_ID-Year-Month")

    df["seasonality_score"] = df["Seasonality_Index"].map({
        "Favorable": 1,
        "Moderate": 0,
        "Un-Favorable": -1
    })

    df["seasonality_multiplier"] = df["Seasonality_Index"].map({
        "Favorable": 1.10,
        "Moderate": 1.00,
        "Un-Favorable": 0.90
    })

    rejected = df[df["_dq_rejection_reason"] != ""].copy()
    silver = df[df["_dq_rejection_reason"] == ""].copy()

    rejected.to_csv(REJECTED_DIR / "distributor_seasonality_rejected.csv", index=False)

    silver = silver.drop(columns=["_dq_rejection_reason"], errors="ignore")
    silver.to_csv(SILVER_DIR / "distributor_seasonality_silver.csv", index=False)

    summary = {
        "dataset": "distributor_seasonality",
        "raw_rows": len(raw),
        "silver_rows": len(silver),
        "rejected_rows": len(rejected)
    }

    return silver, summary


# ============================================================
# HOLIDAY SILVER + MONTHLY HOLIDAY FEATURES
# ============================================================

def process_holidays() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = find_existing_file(["holiday_list.csv", "holiday_list_final.csv"])

    if path is None:
        raise FileNotFoundError("holiday_list.csv not found.")

    raw = read_csv_safely(path)
    df = clean_column_names(raw)
    df = normalize_string_columns(df)

    df["_dq_rejection_reason"] = ""

    required = ["Date", "Holiday_Name", "Holiday_Type"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"{col} column missing in holiday_list.")

    for col in required:
        add_rejection_reason(df, df[col].isna(), f"Missing mandatory field: {col}")

    df["Date_Parsed"] = pd.to_datetime(df["Date"], errors="coerce")
    add_rejection_reason(df, df["Date_Parsed"].isna(), "Invalid Date value")

    df["Holiday_Name"] = (
        df["Holiday_Name"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    df["Holiday_Type"] = standardize_holiday_type(df["Holiday_Type"])

    add_rejection_reason(
        df,
        df["Holiday_Type"].notna() & ~df["Holiday_Type"].isin(VALID_HOLIDAY_TYPES),
        "Invalid Holiday_Type category"
    )

    # Exact duplicate only: same Date + Holiday_Name + Holiday_Type
    duplicate_keep = df.duplicated(subset=["Date", "Holiday_Name", "Holiday_Type"], keep="first")
    add_rejection_reason(df, duplicate_keep, "Exact duplicate holiday record")

    rejected = df[df["_dq_rejection_reason"] != ""].copy()
    silver = df[df["_dq_rejection_reason"] == ""].copy()

    rejected.to_csv(REJECTED_DIR / "holiday_list_rejected.csv", index=False)

    silver["Date"] = silver["Date_Parsed"].dt.date
    silver["Year"] = silver["Date_Parsed"].dt.year
    silver["Month"] = silver["Date_Parsed"].dt.month
    silver["Year_Month"] = silver["Date_Parsed"].dt.to_period("M").astype(str)
    silver["Day_Of_Week"] = silver["Date_Parsed"].dt.dayofweek

    silver["is_public_holiday"] = (silver["Holiday_Type"] == "Public").astype(int)
    silver["is_bank_holiday"] = (silver["Holiday_Type"] == "Bank").astype(int)
    silver["is_mercantile_holiday"] = (silver["Holiday_Type"] == "Mercantile").astype(int)

    silver["is_poya_day"] = (
        (silver["Holiday_Type"] == "Poya Day") |
        (silver["Holiday_Name"].str.contains("Poya", case=False, na=False))
    ).astype(int)

    silver["is_weekend_holiday"] = silver["Day_Of_Week"].isin([5, 6]).astype(int)
    silver["is_long_weekend_proxy"] = silver["Day_Of_Week"].isin([0, 4]).astype(int)

    festival_keywords = [
        "New Year",
        "Vesak",
        "Christmas",
        "Pongal",
        "Deepavali",
        "Ramazan",
        "Eid",
        "Labour",
        "National Day"
    ]

    pattern = "|".join(festival_keywords)

    silver["is_festive_holiday"] = (
        silver["Holiday_Name"]
        .str.contains(pattern, case=False, na=False)
        .astype(int)
    )

    monthly_features = (
        silver
        .groupby(["Year", "Month", "Year_Month"])
        .agg(
            holiday_date_count=("Date", "nunique"),
            holiday_record_count=("Holiday_Name", "count"),
            public_holiday_count=("is_public_holiday", "sum"),
            bank_holiday_count=("is_bank_holiday", "sum"),
            mercantile_holiday_count=("is_mercantile_holiday", "sum"),
            poya_day_count=("is_poya_day", "sum"),
            weekend_holiday_count=("is_weekend_holiday", "sum"),
            long_weekend_holiday_count=("is_long_weekend_proxy", "sum"),
            festive_holiday_count=("is_festive_holiday", "sum")
        )
        .reset_index()
    )

    monthly_features["holiday_intensity_score"] = (
        monthly_features["public_holiday_count"] * 1.00 +
        monthly_features["mercantile_holiday_count"] * 0.80 +
        monthly_features["bank_holiday_count"] * 0.50 +
        monthly_features["poya_day_count"] * 0.70 +
        monthly_features["long_weekend_holiday_count"] * 0.60 +
        monthly_features["festive_holiday_count"] * 1.20
    )

    silver = silver.drop(columns=["_dq_rejection_reason", "Date_Parsed"], errors="ignore")

    silver.to_csv(SILVER_DIR / "holiday_list_silver.csv", index=False)
    monthly_features.to_csv(SILVER_DIR / "holiday_monthly_features_silver.csv", index=False)

    summary = {
        "dataset": "holiday_list",
        "raw_rows": len(raw),
        "silver_rows": len(silver),
        "rejected_rows": len(rejected),
        "monthly_feature_rows": len(monthly_features)
    }

    return silver, monthly_features, summary


# ============================================================
# TRANSACTIONS SILVER
# ============================================================

def build_sku_price_benchmark(sales_df: pd.DataFrame) -> pd.DataFrame:
    benchmark = (
        sales_df
        .groupby("SKU_ID")["Price_Per_Liter"]
        .agg(
            sku_price_count="count",
            sku_price_median="median",
            sku_price_q1=lambda x: np.percentile(x.dropna(), 25),
            sku_price_q3=lambda x: np.percentile(x.dropna(), 75)
        )
        .reset_index()
    )

    benchmark["sku_price_iqr"] = benchmark["sku_price_q3"] - benchmark["sku_price_q1"]

    # If IQR is almost zero, use median-based tolerance.
    benchmark["sku_price_lower_bound"] = np.where(
        benchmark["sku_price_iqr"] > 0,
        benchmark["sku_price_q1"] - 3 * benchmark["sku_price_iqr"],
        benchmark["sku_price_median"] * 0.75
    )

    benchmark["sku_price_upper_bound"] = np.where(
        benchmark["sku_price_iqr"] > 0,
        benchmark["sku_price_q3"] + 3 * benchmark["sku_price_iqr"],
        benchmark["sku_price_median"] * 1.25
    )

    benchmark["sku_price_lower_bound"] = benchmark["sku_price_lower_bound"].clip(lower=0)

    return benchmark


def process_transactions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    path = find_existing_file(["transactions_history_final.csv", "transactions_history.csv"])

    if path is None:
        raise FileNotFoundError("transactions_history_final.csv / transactions_history.csv not found.")

    raw = read_csv_safely(path)
    df = clean_column_names(raw)
    df = normalize_string_columns(df)

    df["_dq_rejection_reason"] = ""

    required = [
        "Outlet_ID",
        "Year",
        "Month",
        "Distributor_ID",
        "SKU_ID",
        "Volume_Liters",
        "Total_Bill_Value"
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"{col} column missing in transactions.")

    for col in required:
        add_rejection_reason(df, df[col].isna(), f"Missing mandatory field: {col}")

    # Numeric conversion
    for col in ["Year", "Month", "Volume_Liters", "Total_Bill_Value"]:
        original_not_missing = df[col].notna()
        converted = pd.to_numeric(df[col], errors="coerce")

        add_rejection_reason(
            df,
            original_not_missing & converted.isna(),
            f"Invalid numeric value in {col}"
        )

        df[col] = converted

    add_rejection_reason(df, df["Year"].notna() & ~df["Year"].isin(EXPECTED_TRANSACTION_YEARS), "Year outside expected range 2023-2025")
    add_rejection_reason(df, df["Month"].notna() & ~df["Month"].between(1, 12), "Month outside range 1-12")

    # Exact duplicate rows
    exact_duplicate = df.drop(columns=["_dq_rejection_reason"], errors="ignore").duplicated(keep="first")
    add_rejection_reason(df, exact_duplicate, "Exact duplicate row")

    # First separate truly invalid records
    base_rejected = df[df["_dq_rejection_reason"] != ""].copy()
    valid_base = df[df["_dq_rejection_reason"] == ""].copy()

    # Classify transaction business meaning
    positive_sales_mask = (
        (valid_base["Volume_Liters"] > 0) &
        (valid_base["Total_Bill_Value"] > 0)
    )

    returns_mask = (
        (valid_base["Volume_Liters"] < 0) &
        (valid_base["Total_Bill_Value"] < 0)
    )

    zero_volume_positive_bill_mask = (
        (valid_base["Volume_Liters"] == 0) &
        (valid_base["Total_Bill_Value"] > 0)
    )

    zero_value_positive_volume_mask = (
        (valid_base["Volume_Liters"] > 0) &
        (valid_base["Total_Bill_Value"] == 0)
    )

    zero_zero_mask = (
        (valid_base["Volume_Liters"] == 0) &
        (valid_base["Total_Bill_Value"] == 0)
    )

    sign_mismatch_mask = (
        ((valid_base["Volume_Liters"] < 0) & (valid_base["Total_Bill_Value"] >= 0)) |
        ((valid_base["Volume_Liters"] > 0) & (valid_base["Total_Bill_Value"] < 0))
    )

    sales = valid_base[positive_sales_mask].copy()
    returns = valid_base[returns_mask].copy()

    billing_anomalies = valid_base[
        zero_volume_positive_bill_mask |
        zero_value_positive_volume_mask |
        zero_zero_mask
    ].copy()

    sign_mismatch_rejected = valid_base[sign_mismatch_mask].copy()

    if len(sign_mismatch_rejected) > 0:
        sign_mismatch_rejected["_dq_rejection_reason"] = "Sign mismatch between Volume_Liters and Total_Bill_Value"

    if len(billing_anomalies) > 0:
        billing_anomalies["anomaly_type"] = np.select(
            [
                zero_volume_positive_bill_mask.loc[billing_anomalies.index],
                zero_value_positive_volume_mask.loc[billing_anomalies.index],
                zero_zero_mask.loc[billing_anomalies.index]
            ],
            [
                "zero_volume_positive_bill",
                "positive_volume_zero_bill",
                "zero_volume_zero_bill"
            ],
            default="other_billing_anomaly"
        )

    # Price per liter
    sales["Price_Per_Liter"] = sales["Total_Bill_Value"] / sales["Volume_Liters"]
    returns["Price_Per_Liter"] = returns["Total_Bill_Value"].abs() / returns["Volume_Liters"].abs()

    # SKU-level price benchmark using only positive sales
    price_benchmark = build_sku_price_benchmark(sales)
    price_benchmark.to_csv(SUMMARY_DIR / "sku_price_benchmark.csv", index=False)

    sales = sales.merge(price_benchmark, on="SKU_ID", how="left")

    sales["sku_price_outlier_flag"] = (
        (sales["sku_price_count"] >= PRICE_MIN_RECORDS_PER_SKU) &
        (
            (sales["Price_Per_Liter"] < sales["sku_price_lower_bound"]) |
            (sales["Price_Per_Liter"] > sales["sku_price_upper_bound"])
        )
    ).astype(int)

    price_outliers = sales[sales["sku_price_outlier_flag"] == 1].copy()
    sales_clean = sales[sales["sku_price_outlier_flag"] == 0].copy()

    if len(price_outliers) > 0:
        price_outliers["anomaly_type"] = "sku_level_price_outlier"

    # Possible duplicate transaction key report only, do not remove
    tx_key_cols = ["Outlet_ID", "Year", "Month", "Distributor_ID", "SKU_ID"]

    possible_duplicate_keys = (
        sales_clean
        .groupby(tx_key_cols)
        .size()
        .reset_index(name="row_count")
        .query("row_count > 1")
        .sort_values("row_count", ascending=False)
    )

    possible_duplicate_keys.to_csv(SUMMARY_DIR / "possible_transaction_duplicate_keys.csv", index=False)

    rejected = pd.concat(
        [base_rejected, sign_mismatch_rejected],
        ignore_index=True
    )

    # Save outputs
    sales_clean.to_csv(SILVER_DIR / "transactions_sales_silver.csv", index=False)
    returns.to_csv(SILVER_DIR / "transactions_returns_silver.csv", index=False)

    billing_anomalies.to_csv(ANOMALY_DIR / "transactions_billing_anomalies.csv", index=False)
    price_outliers.to_csv(ANOMALY_DIR / "transactions_price_outliers.csv", index=False)

    rejected.to_csv(REJECTED_DIR / "transactions_rejected.csv", index=False)

    # Return features for later Gold
    return_features = pd.DataFrame()

    if len(returns) > 0:
        returns["return_liters_abs"] = returns["Volume_Liters"].abs()
        returns["return_value_abs"] = returns["Total_Bill_Value"].abs()

        return_features = (
            returns
            .groupby("Outlet_ID")
            .agg(
                return_record_count=("Outlet_ID", "count"),
                return_liters_abs=("return_liters_abs", "sum"),
                return_value_abs=("return_value_abs", "sum"),
                return_month_count=("Month", "nunique"),
                return_sku_count=("SKU_ID", "nunique")
            )
            .reset_index()
        )

    return_features.to_csv(SILVER_DIR / "transaction_return_features_silver.csv", index=False)

    # Billing anomaly features for later Gold
    billing_anomaly_features = pd.DataFrame()

    if len(billing_anomalies) > 0:
        billing_anomaly_features = (
            billing_anomalies
            .groupby("Outlet_ID")
            .agg(
                billing_anomaly_count=("Outlet_ID", "count"),
                zero_volume_positive_bill_count=("anomaly_type", lambda x: (x == "zero_volume_positive_bill").sum()),
                positive_volume_zero_bill_count=("anomaly_type", lambda x: (x == "positive_volume_zero_bill").sum()),
                zero_volume_zero_bill_count=("anomaly_type", lambda x: (x == "zero_volume_zero_bill").sum())
            )
            .reset_index()
        )

    billing_anomaly_features.to_csv(SILVER_DIR / "transaction_billing_anomaly_features_silver.csv", index=False)

    summary = {
        "dataset": "transactions",
        "raw_rows": len(raw),
        "positive_sales_silver_rows": len(sales_clean),
        "returns_silver_rows": len(returns),
        "billing_anomaly_rows": len(billing_anomalies),
        "price_outlier_rows": len(price_outliers),
        "rejected_rows": len(rejected),
        "negative_return_share_of_raw": round(len(returns) / len(raw), 5) if len(raw) else 0,
        "zero_volume_billing_anomaly_share_of_raw": round(len(billing_anomalies) / len(raw), 5) if len(raw) else 0,
        "sku_price_outlier_share_of_positive_sales": round(len(price_outliers) / max(len(sales), 1), 5)
    }

    return sales_clean, returns, billing_anomalies, rejected, summary


# ============================================================
# SUMMARY WRITER
# ============================================================

def write_summaries(outputs: dict, quality_summaries: list[dict]) -> None:
    dataset_summaries = []
    column_summaries = []

    for label, df in outputs.items():
        dataset_summaries.append(create_dataset_summary(label, df))
        column_summaries.append(create_column_summary(label, df))

    pd.DataFrame(dataset_summaries).to_csv(
        SUMMARY_DIR / "silver_dataset_summary.csv",
        index=False
    )

    if column_summaries:
        pd.concat(column_summaries, ignore_index=True).to_csv(
            SUMMARY_DIR / "silver_column_summary.csv",
            index=False
        )

    pd.DataFrame(quality_summaries).to_csv(
        SUMMARY_DIR / "silver_quality_summary.csv",
        index=False
    )

    # Rejection reason summary
    reason_rows = []

    for rejected_file in REJECTED_DIR.glob("*.csv"):
        r = read_csv_safely(rejected_file)

        if "_dq_rejection_reason" in r.columns and len(r) > 0:
            exploded = (
                r["_dq_rejection_reason"]
                .dropna()
                .astype(str)
                .str.split(" | ", regex=False)
                .explode()
            )

            counts = Counter(exploded)

            for reason, count in counts.items():
                reason_rows.append({
                    "file": rejected_file.name,
                    "reason": reason,
                    "count": count
                })

    pd.DataFrame(reason_rows).to_csv(
        SUMMARY_DIR / "silver_rejection_reason_summary.csv",
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting Silver pipeline...")

    create_bronze_copies()

    quality_summaries = []
    outputs = {}

    outlet_master, summary = process_outlet_master()
    quality_summaries.append(summary)
    outputs["outlet_master_silver.csv"] = outlet_master

    outlet_coordinates, summary = process_outlet_coordinates()
    quality_summaries.append(summary)
    outputs["outlet_coordinates_silver.csv"] = outlet_coordinates

    distributor_seasonality, summary = process_distributor_seasonality()
    quality_summaries.append(summary)
    outputs["distributor_seasonality_silver.csv"] = distributor_seasonality

    holiday_list, holiday_monthly_features, summary = process_holidays()
    quality_summaries.append(summary)
    outputs["holiday_list_silver.csv"] = holiday_list
    outputs["holiday_monthly_features_silver.csv"] = holiday_monthly_features

    sales, returns, billing_anomalies, rejected_tx, summary = process_transactions()
    quality_summaries.append(summary)
    outputs["transactions_sales_silver.csv"] = sales
    outputs["transactions_returns_silver.csv"] = returns
    outputs["transactions_billing_anomalies.csv"] = billing_anomalies
    outputs["transactions_rejected.csv"] = rejected_tx

    write_summaries(outputs, quality_summaries)

    print("\nSilver pipeline completed successfully.")
    print("Bronze files:", BRONZE_DIR)
    print("Silver files:", SILVER_DIR)
    print("Rejected files:", REJECTED_DIR)
    print("Anomaly files:", ANOMALY_DIR)
    print("Summary files:", SUMMARY_DIR)

    print("\nMain Silver outputs:")
    print("-", SILVER_DIR / "outlet_master_silver.csv")
    print("-", SILVER_DIR / "outlet_coordinates_silver.csv")
    print("-", SILVER_DIR / "transactions_sales_silver.csv")
    print("-", SILVER_DIR / "transactions_returns_silver.csv")
    print("-", SILVER_DIR / "holiday_monthly_features_silver.csv")
    print("-", SILVER_DIR / "distributor_seasonality_silver.csv")


if __name__ == "__main__":
    main()