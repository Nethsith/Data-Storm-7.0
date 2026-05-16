from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path("raw data")
SILVER_DIR = Path("processed/silver")
SUMMARY_DIR = Path("summaries")
REJECTED_DIR = Path("processed/rejected")

SILVER_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

VALID_HOLIDAY_TYPES = {"Public", "Bank", "Mercantile", "Poya Day"}

FESTIVE_KEYWORDS = [
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

SEASONALITY_DEFAULT_MULTIPLIER = {
    "Favorable": 1.10,
    "Moderate": 1.00,
    "Un-Favorable": 0.90
}

SEASONALITY_SCORE = {
    "Favorable": 1,
    "Moderate": 0,
    "Un-Favorable": -1
}


# ============================================================
# HELPERS
# ============================================================

def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, engine="python", low_memory=False)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def clean_text_series(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def standardize_holiday_type(s: pd.Series) -> pd.Series:
    cleaned = clean_text_series(s).str.title()

    return cleaned.replace({
        "Public Holiday": "Public",
        "Bank Holiday": "Bank",
        "Mercantile Holiday": "Mercantile",
        "Poya": "Poya Day",
        "Poya day": "Poya Day",
        "Poya Day": "Poya Day"
    })


def standardize_seasonality(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype("string")
        .str.strip()
        .str.lower()
        .str.replace("_", "-", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    return cleaned.replace({
        "favorable": "Favorable",
        "favourable": "Favorable",
        "moderate": "Moderate",
        "unfavorable": "Un-Favorable",
        "un-favorable": "Un-Favorable",
        "un favourable": "Un-Favorable",
        "un favorable": "Un-Favorable"
    })


def find_holiday_input_file() -> Path:
    possible_files = [
        RAW_DIR / "holiday_list.csv",
        RAW_DIR / "holiday_list_final.csv",
        SILVER_DIR / "holiday_list_silver.csv"
    ]

    for path in possible_files:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find holiday file. Expected raw data/holiday_list.csv "
        "or processed/silver/holiday_list_silver.csv"
    )


def parse_holiday_date(date_series: pd.Series) -> pd.Series:
    # In your data, values like 1/6/2023 mean Jan 6, 2023.
    # Therefore dayfirst=False is used.
    return pd.to_datetime(date_series, errors="coerce", dayfirst=False)


def unique_join(values) -> str:
    clean_values = (
        pd.Series(values)
        .dropna()
        .astype(str)
        .str.strip()
    )

    unique_values = sorted(clean_values.unique())

    return " | ".join(unique_values)


def cap_multiplier(row) -> float:
    label = row["Seasonality_Index"]
    value = row["seasonality_multiplier_calibrated"]

    if label == "Favorable":
        return float(np.clip(value, 1.00, 1.20))

    if label == "Moderate":
        return 1.00

    if label == "Un-Favorable":
        return float(np.clip(value, 0.80, 1.00))

    return 1.00


# ============================================================
# PART 1 — FIX HOLIDAY SCORE
# ============================================================

def fix_holiday_score():
    holiday_path = find_holiday_input_file()

    print("Reading holiday file:", holiday_path)

    raw = read_csv_safely(holiday_path)
    df = clean_column_names(raw)

    required_cols = ["Date", "Holiday_Name", "Holiday_Type"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Holiday file missing required columns: {missing_cols}")

    df["_dq_rejection_reason"] = ""

    # Basic cleaning
    df["Holiday_Name"] = clean_text_series(df["Holiday_Name"])
    df["Holiday_Type"] = standardize_holiday_type(df["Holiday_Type"])
    df["Date_Parsed"] = parse_holiday_date(df["Date"])

    # Mandatory checks
    df.loc[df["Date_Parsed"].isna(), "_dq_rejection_reason"] += "Invalid Date value"
    df.loc[df["Holiday_Name"].isna(), "_dq_rejection_reason"] += "Missing Holiday_Name"
    df.loc[df["Holiday_Type"].isna(), "_dq_rejection_reason"] += "Missing Holiday_Type"

    invalid_type_mask = (
        df["Holiday_Type"].notna()
        & ~df["Holiday_Type"].isin(VALID_HOLIDAY_TYPES)
    )

    df.loc[invalid_type_mask, "_dq_rejection_reason"] += "Invalid Holiday_Type"

    valid_base = df[df["_dq_rejection_reason"] == ""].copy()
    rejected_base = df[df["_dq_rejection_reason"] != ""].copy()

    # Remove exact duplicates only.
    # Do not remove different classifications on the same date.
    exact_dup_mask = valid_base.duplicated(
        subset=["Date_Parsed", "Holiday_Name", "Holiday_Type"],
        keep="first"
    )

    exact_duplicates = valid_base[exact_dup_mask].copy()
    exact_duplicates["_dq_rejection_reason"] = "Exact duplicate holiday classification record"

    holiday_row_level = valid_base[~exact_dup_mask].copy()

    rejected = pd.concat([rejected_base, exact_duplicates], ignore_index=True)

    # Create row-level Silver holiday file
    holiday_row_level["Date"] = holiday_row_level["Date_Parsed"].dt.date
    holiday_row_level["Year"] = holiday_row_level["Date_Parsed"].dt.year
    holiday_row_level["Month"] = holiday_row_level["Date_Parsed"].dt.month
    holiday_row_level["Year_Month"] = holiday_row_level["Date_Parsed"].dt.to_period("M").astype(str)
    holiday_row_level["Day_Name"] = holiday_row_level["Date_Parsed"].dt.day_name()
    holiday_row_level["Day_Of_Week"] = holiday_row_level["Date_Parsed"].dt.dayofweek

    # Pandas dayofweek:
    # Monday = 0, Friday = 4, Saturday = 5, Sunday = 6
    holiday_row_level["row_has_public_holiday"] = (
        holiday_row_level["Holiday_Type"] == "Public"
    ).astype(int)

    holiday_row_level["row_has_bank_holiday"] = (
        holiday_row_level["Holiday_Type"] == "Bank"
    ).astype(int)

    holiday_row_level["row_has_mercantile_holiday"] = (
        holiday_row_level["Holiday_Type"] == "Mercantile"
    ).astype(int)

    holiday_row_level["row_has_poya_day"] = (
        (holiday_row_level["Holiday_Type"] == "Poya Day")
        | holiday_row_level["Holiday_Name"].str.contains("Poya", case=False, na=False)
    ).astype(int)

    festive_pattern = "|".join(FESTIVE_KEYWORDS)

    holiday_row_level["row_is_festive_holiday"] = (
        holiday_row_level["Holiday_Name"]
        .str.contains(festive_pattern, case=False, na=False)
        .astype(int)
    )

    holiday_row_level["is_weekend_holiday"] = (
        holiday_row_level["Day_Of_Week"].isin([5, 6])
    ).astype(int)

    holiday_row_level["is_long_weekend_proxy"] = (
        holiday_row_level["Day_Of_Week"].isin([0, 4])
    ).astype(int)

    # Save row-level cleaned holiday file
    holiday_list_silver_cols = [
        "Date",
        "Holiday_Name",
        "Holiday_Type",
        "Year",
        "Month",
        "Year_Month",
        "Day_Name",
        "Day_Of_Week",
        "row_has_public_holiday",
        "row_has_bank_holiday",
        "row_has_mercantile_holiday",
        "row_has_poya_day",
        "row_is_festive_holiday",
        "is_weekend_holiday",
        "is_long_weekend_proxy"
    ]

    holiday_row_level[holiday_list_silver_cols].to_csv(
        SILVER_DIR / "holiday_list_silver.csv",
        index=False
    )

    # ========================================================
    # Collapse to unique date level
    # This fixes the overcounting issue.
    # Example: one Poya date with Public/Bank/Mercantile/Poya rows
    # should count as 1 Poya date, not 4 Poya holidays.
    # ========================================================

    date_level = (
        holiday_row_level
        .groupby(
            [
                "Date",
                "Year",
                "Month",
                "Year_Month",
                "Day_Name",
                "Day_Of_Week"
            ],
            as_index=False
        )
        .agg(
            holiday_names_combined=("Holiday_Name", unique_join),
            holiday_types_combined=("Holiday_Type", unique_join),
            holiday_classification_count=("Holiday_Type", "count"),
            has_public_holiday=("row_has_public_holiday", "max"),
            has_bank_holiday=("row_has_bank_holiday", "max"),
            has_mercantile_holiday=("row_has_mercantile_holiday", "max"),
            has_poya_day=("row_has_poya_day", "max"),
            is_weekend_holiday=("is_weekend_holiday", "max"),
            is_long_weekend_proxy=("is_long_weekend_proxy", "max"),
            is_festive_holiday=("row_is_festive_holiday", "max")
        )
    )

    date_level.to_csv(
        SILVER_DIR / "holiday_date_level_silver.csv",
        index=False
    )

    # ========================================================
    # Monthly holiday features from unique holiday dates
    # ========================================================

    monthly_features = (
        date_level
        .groupby(["Year", "Month", "Year_Month"], as_index=False)
        .agg(
            holiday_date_count=("Date", "nunique"),
            holiday_classification_count=("holiday_classification_count", "sum"),
            public_holiday_date_count=("has_public_holiday", "sum"),
            bank_holiday_date_count=("has_bank_holiday", "sum"),
            mercantile_holiday_date_count=("has_mercantile_holiday", "sum"),
            poya_day_date_count=("has_poya_day", "sum"),
            weekend_holiday_date_count=("is_weekend_holiday", "sum"),
            long_weekend_holiday_date_count=("is_long_weekend_proxy", "sum"),
            festive_holiday_date_count=("is_festive_holiday", "sum")
        )
    )

    monthly_features["holiday_intensity_score"] = (
        monthly_features["public_holiday_date_count"] * 1.00
        + monthly_features["mercantile_holiday_date_count"] * 0.80
        + monthly_features["bank_holiday_date_count"] * 0.50
        + monthly_features["poya_day_date_count"] * 0.70
        + monthly_features["long_weekend_holiday_date_count"] * 0.60
        + monthly_features["festive_holiday_date_count"] * 1.20
    )

    # Compatibility aliases.
    # Your current Gold code uses old names like public_holiday_count.
    # These aliases now contain corrected unique-date-level counts.
    monthly_features["holiday_record_count"] = monthly_features["holiday_classification_count"]
    monthly_features["public_holiday_count"] = monthly_features["public_holiday_date_count"]
    monthly_features["bank_holiday_count"] = monthly_features["bank_holiday_date_count"]
    monthly_features["mercantile_holiday_count"] = monthly_features["mercantile_holiday_date_count"]
    monthly_features["poya_day_count"] = monthly_features["poya_day_date_count"]
    monthly_features["weekend_holiday_count"] = monthly_features["weekend_holiday_date_count"]
    monthly_features["long_weekend_holiday_count"] = monthly_features["long_weekend_holiday_date_count"]
    monthly_features["festive_holiday_count"] = monthly_features["festive_holiday_date_count"]

    # Overwrite existing Silver holiday monthly feature file
    monthly_features.to_csv(
        SILVER_DIR / "holiday_monthly_features_silver.csv",
        index=False
    )

    rejected.to_csv(
        REJECTED_DIR / "holiday_list_rejected.csv",
        index=False
    )

    # Validation for January 2023
    jan_2023_date_level = date_level[
        (date_level["Year"] == 2023)
        & (date_level["Month"] == 1)
    ].copy()

    jan_2023_monthly = monthly_features[
        (monthly_features["Year"] == 2023)
        & (monthly_features["Month"] == 1)
    ].copy()

    jan_2023_date_level.to_csv(
        SUMMARY_DIR / "holiday_jan_2023_date_level_check.csv",
        index=False
    )

    jan_2023_monthly.to_csv(
        SUMMARY_DIR / "holiday_jan_2023_monthly_check.csv",
        index=False
    )

    summary = {
        "raw_rows": len(raw),
        "cleaned_row_level_rows_after_exact_duplicate_removal": len(holiday_row_level),
        "rejected_or_exact_duplicate_rows": len(rejected),
        "unique_holiday_date_rows": len(date_level),
        "monthly_feature_rows": len(monthly_features),
        "jan_2023_unique_holiday_dates": int(jan_2023_monthly["holiday_date_count"].iloc[0]) if len(jan_2023_monthly) else 0,
        "jan_2023_holiday_classification_count": int(jan_2023_monthly["holiday_classification_count"].iloc[0]) if len(jan_2023_monthly) else 0,
        "jan_2023_poya_day_date_count": int(jan_2023_monthly["poya_day_date_count"].iloc[0]) if len(jan_2023_monthly) else 0,
        "jan_2023_holiday_intensity_score": float(jan_2023_monthly["holiday_intensity_score"].iloc[0]) if len(jan_2023_monthly) else 0
    }

    pd.DataFrame([summary]).to_csv(
        SUMMARY_DIR / "holiday_monthly_feature_summary.csv",
        index=False
    )

    print("\nHoliday score fixed and files overwritten.")
    print("Saved:")
    print("-", SILVER_DIR / "holiday_list_silver.csv")
    print("-", SILVER_DIR / "holiday_date_level_silver.csv")
    print("-", SILVER_DIR / "holiday_monthly_features_silver.csv")
    print("-", SUMMARY_DIR / "holiday_monthly_feature_summary.csv")

    print("\nJanuary 2023 check:")
    print(jan_2023_monthly[[
        "Year",
        "Month",
        "holiday_date_count",
        "holiday_classification_count",
        "public_holiday_date_count",
        "bank_holiday_date_count",
        "mercantile_holiday_date_count",
        "poya_day_date_count",
        "long_weekend_holiday_date_count",
        "festive_holiday_date_count",
        "holiday_intensity_score"
    ]] if len(jan_2023_monthly) else "No January 2023 row found.")

    return monthly_features


# ============================================================
# PART 2 — CREATE DISTRIBUTOR-MONTH DEMAND CONTEXT SCORE
# ============================================================

def create_distributor_month_context_score():
    holiday_file = SILVER_DIR / "holiday_monthly_features_silver.csv"
    seasonality_file = SILVER_DIR / "distributor_seasonality_silver.csv"
    sales_file = SILVER_DIR / "transactions_sales_silver.csv"

    if not holiday_file.exists():
        raise FileNotFoundError("Run fix_holiday_score() first. holiday_monthly_features_silver.csv not found.")

    if not seasonality_file.exists():
        print("\nNo distributor_seasonality_silver.csv found. Skipping distributor-month context score.")
        return None

    holidays = read_csv_safely(holiday_file)
    seasonality = read_csv_safely(seasonality_file)

    seasonality.columns = (
        seasonality.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    required_cols = ["Distributor_ID", "Year", "Month", "Seasonality_Index"]
    missing_cols = [c for c in required_cols if c not in seasonality.columns]

    if missing_cols:
        raise ValueError(f"Seasonality file missing required columns: {missing_cols}")

    seasonality["Year"] = pd.to_numeric(seasonality["Year"], errors="coerce").astype("Int64")
    seasonality["Month"] = pd.to_numeric(seasonality["Month"], errors="coerce").astype("Int64")
    seasonality["Seasonality_Index"] = standardize_seasonality(seasonality["Seasonality_Index"])

    seasonality["seasonality_score"] = seasonality["Seasonality_Index"].map(SEASONALITY_SCORE).fillna(0)
    seasonality["seasonality_multiplier_default"] = (
        seasonality["Seasonality_Index"]
        .map(SEASONALITY_DEFAULT_MULTIPLIER)
        .fillna(1.00)
    )

    holidays["Year"] = pd.to_numeric(holidays["Year"], errors="coerce").astype("Int64")
    holidays["Month"] = pd.to_numeric(holidays["Month"], errors="coerce").astype("Int64")

    context = seasonality.merge(
        holidays,
        on=["Year", "Month"],
        how="left"
    )

    holiday_numeric_cols = [
        "holiday_date_count",
        "holiday_classification_count",
        "public_holiday_date_count",
        "bank_holiday_date_count",
        "mercantile_holiday_date_count",
        "poya_day_date_count",
        "weekend_holiday_date_count",
        "long_weekend_holiday_date_count",
        "festive_holiday_date_count",
        "holiday_intensity_score"
    ]

    for col in holiday_numeric_cols:
        if col in context.columns:
            context[col] = pd.to_numeric(context[col], errors="coerce").fillna(0)

    # Default values first
    context["seasonality_multiplier_calibrated"] = context["seasonality_multiplier_default"]
    context["seasonality_multiplier_source"] = "fallback_default"

    # Optional calibration using clean Silver sales
    if sales_file.exists():
        sales = read_csv_safely(sales_file)

        needed_sales_cols = ["Distributor_ID", "Year", "Month", "Volume_Liters"]

        if all(c in sales.columns for c in needed_sales_cols):
            sales["Year"] = pd.to_numeric(sales["Year"], errors="coerce").astype("Int64")
            sales["Month"] = pd.to_numeric(sales["Month"], errors="coerce").astype("Int64")
            sales["Volume_Liters"] = pd.to_numeric(sales["Volume_Liters"], errors="coerce")

            monthly_dist_sales = (
                sales
                .groupby(["Distributor_ID", "Year", "Month"], as_index=False)
                .agg(monthly_distributor_liters=("Volume_Liters", "sum"))
            )

            sales_with_seasonality = monthly_dist_sales.merge(
                seasonality[["Distributor_ID", "Year", "Month", "Seasonality_Index"]],
                on=["Distributor_ID", "Year", "Month"],
                how="left"
            )

            dist_label_avg = (
                sales_with_seasonality
                .dropna(subset=["Seasonality_Index"])
                .groupby(["Distributor_ID", "Seasonality_Index"], as_index=False)
                .agg(avg_liters_by_label=("monthly_distributor_liters", "mean"))
            )

            dist_moderate = (
                dist_label_avg[dist_label_avg["Seasonality_Index"] == "Moderate"]
                [["Distributor_ID", "avg_liters_by_label"]]
                .rename(columns={"avg_liters_by_label": "moderate_avg_liters"})
            )

            dist_multipliers = dist_label_avg.merge(
                dist_moderate,
                on="Distributor_ID",
                how="left"
            )

            dist_multipliers["seasonality_multiplier_calibrated"] = (
                dist_multipliers["avg_liters_by_label"]
                / dist_multipliers["moderate_avg_liters"].replace(0, np.nan)
            )

            # Global fallback ratios
            global_label_avg = (
                sales_with_seasonality
                .dropna(subset=["Seasonality_Index"])
                .groupby("Seasonality_Index", as_index=False)
                .agg(global_avg_liters_by_label=("monthly_distributor_liters", "mean"))
            )

            global_moderate_value = global_label_avg.loc[
                global_label_avg["Seasonality_Index"] == "Moderate",
                "global_avg_liters_by_label"
            ]

            if len(global_moderate_value) > 0 and global_moderate_value.iloc[0] > 0:
                global_moderate_value = global_moderate_value.iloc[0]

                global_label_avg["global_multiplier"] = (
                    global_label_avg["global_avg_liters_by_label"]
                    / global_moderate_value
                )
            else:
                global_label_avg["global_multiplier"] = global_label_avg["Seasonality_Index"].map(
                    SEASONALITY_DEFAULT_MULTIPLIER
                )

            dist_multipliers = dist_multipliers.merge(
                global_label_avg[["Seasonality_Index", "global_multiplier"]],
                on="Seasonality_Index",
                how="left"
            )

            dist_multipliers["seasonality_multiplier_source"] = np.where(
                dist_multipliers["seasonality_multiplier_calibrated"].notna(),
                "distributor_calculated",
                "global_calculated"
            )

            dist_multipliers["seasonality_multiplier_calibrated"] = (
                dist_multipliers["seasonality_multiplier_calibrated"]
                .fillna(dist_multipliers["global_multiplier"])
            )

            dist_multipliers["seasonality_multiplier_calibrated"] = (
                dist_multipliers["seasonality_multiplier_calibrated"]
                .fillna(dist_multipliers["Seasonality_Index"].map(SEASONALITY_DEFAULT_MULTIPLIER))
            )

            dist_multipliers["seasonality_multiplier_calibrated"] = dist_multipliers.apply(
                cap_multiplier,
                axis=1
            )

            dist_multipliers_out = dist_multipliers[
                [
                    "Distributor_ID",
                    "Seasonality_Index",
                    "avg_liters_by_label",
                    "moderate_avg_liters",
                    "seasonality_multiplier_calibrated",
                    "seasonality_multiplier_source"
                ]
            ].copy()

            dist_multipliers_out.to_csv(
                SUMMARY_DIR / "calibrated_seasonality_multipliers.csv",
                index=False
            )

            context = context.drop(
                columns=["seasonality_multiplier_calibrated", "seasonality_multiplier_source"],
                errors="ignore"
            )

            context = context.merge(
                dist_multipliers_out[
                    [
                        "Distributor_ID",
                        "Seasonality_Index",
                        "seasonality_multiplier_calibrated",
                        "seasonality_multiplier_source"
                    ]
                ],
                on=["Distributor_ID", "Seasonality_Index"],
                how="left"
            )

            context["seasonality_multiplier_calibrated"] = (
                context["seasonality_multiplier_calibrated"]
                .fillna(context["seasonality_multiplier_default"])
            )

            context["seasonality_multiplier_source"] = (
                context["seasonality_multiplier_source"]
                .fillna("fallback_default")
            )

    context["distributor_month_demand_context_score"] = (
        context["holiday_intensity_score"]
        * context["seasonality_multiplier_calibrated"]
    )

    # Useful binary flags
    context["is_favorable_holiday_month"] = (
        (context["holiday_intensity_score"] > 0)
        & (context["Seasonality_Index"] == "Favorable")
    ).astype(int)

    context["is_unfavorable_holiday_month"] = (
        (context["holiday_intensity_score"] > 0)
        & (context["Seasonality_Index"] == "Un-Favorable")
    ).astype(int)

    context["is_festive_favorable_month"] = (
        (context["festive_holiday_date_count"] > 0)
        & (context["Seasonality_Index"] == "Favorable")
    ).astype(int)

    # Normalize context score 0-100 for interpretation only
    min_score = context["distributor_month_demand_context_score"].min()
    max_score = context["distributor_month_demand_context_score"].max()

    if pd.notna(min_score) and pd.notna(max_score) and max_score > min_score:
        context["distributor_month_context_score_0_100"] = (
            (context["distributor_month_demand_context_score"] - min_score)
            / (max_score - min_score)
            * 100
        ).round(2)
    else:
        context["distributor_month_context_score_0_100"] = 0.0

    context.to_csv(
        SILVER_DIR / "distributor_month_context_score_silver.csv",
        index=False
    )

    summary = (
        context
        .groupby(["Distributor_ID", "Seasonality_Index"], as_index=False)
        .agg(
            avg_context_score=("distributor_month_demand_context_score", "mean"),
            max_context_score=("distributor_month_demand_context_score", "max"),
            month_count=("Month", "count")
        )
    )

    summary.to_csv(
        SUMMARY_DIR / "distributor_month_context_score_summary.csv",
        index=False
    )

    print("\nDistributor-month demand context score created.")
    print("Saved:")
    print("-", SILVER_DIR / "distributor_month_context_score_silver.csv")
    print("-", SUMMARY_DIR / "distributor_month_context_score_summary.csv")
    print("-", SUMMARY_DIR / "calibrated_seasonality_multipliers.csv")

    return context


# ============================================================
# MAIN
# ============================================================

def main():
    fix_holiday_score()
    create_distributor_month_context_score()

    print("\nDone.")


if __name__ == "__main__":
    main()