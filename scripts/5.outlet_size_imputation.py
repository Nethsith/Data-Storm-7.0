"""
Outlet Size Imputation Script
Reads silver-prepared outlet features and performs ML-based imputation for missing Outlet_Size values.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
LOW_CONFIDENCE_THRESHOLD = 0.80
VALID_OUTLET_SIZES = {"Small", "Medium", "Large", "Extra Large"}


def get_project_dir() -> Path:
    """Find project root directory."""
    base = Path.cwd()
    if (base / "raw data").exists():
        return base
    if (base.parent / "raw data").exists():
        return base.parent
    return base


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read CSV with fallback to python engine."""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, engine="python", low_memory=False)


def safe_mode(series: pd.Series):
    """Get mode of series, handling empty case."""
    mode = series.mode(dropna=True)
    return mode.iloc[0] if len(mode) > 0 else pd.NA


# ============================================================
# SETUP PATHS
# ============================================================

PROJECT_DIR = get_project_dir()
SILVER_DIR = PROJECT_DIR / "processed/silver"
SUMMARY_DIR = PROJECT_DIR / "summaries"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Required silver files for imputation
REQUIRED_FILES = {
    "outlet_master": SILVER_DIR / "outlet_master_silver.csv",
    "outlet_coordinates": SILVER_DIR / "outlet_coordinates_silver.csv",
    "transactions_sales": SILVER_DIR / "transactions_sales_silver.csv",
    "transactions_returns": SILVER_DIR / "transactions_returns_silver.csv",
    "holidays_monthly": SILVER_DIR / "holiday_monthly_features_silver.csv",
    "seasonality": SILVER_DIR / "distributor_seasonality_silver.csv",
}

for name, path in REQUIRED_FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Required silver file not found: {path}\nRun script 3.silver_data.py first.")


# ============================================================
# LOAD SILVER DATA
# ============================================================

print("Loading silver data files...")
outlet_master = read_csv_safely(REQUIRED_FILES["outlet_master"])
outlet_coordinates = read_csv_safely(REQUIRED_FILES["outlet_coordinates"])
sales = read_csv_safely(REQUIRED_FILES["transactions_sales"])
returns = read_csv_safely(REQUIRED_FILES["transactions_returns"])
holidays = read_csv_safely(REQUIRED_FILES["holidays_monthly"])
seasonality = read_csv_safely(REQUIRED_FILES["seasonality"])

print(f"  Outlet Master: {len(outlet_master)} rows")
print(f"  Outlet Coordinates: {len(outlet_coordinates)} rows")
print(f"  Sales: {len(sales)} rows")
print(f"  Returns: {len(returns)} rows")


# ============================================================
# TYPE STANDARDIZATION
# ============================================================

# Numeric columns
for col in ["Volume_Liters", "Total_Bill_Value", "Price_Per_Liter"]:
    if col in sales.columns:
        sales[col] = pd.to_numeric(sales[col], errors="coerce")
    if col in returns.columns:
        returns[col] = pd.to_numeric(returns[col], errors="coerce")

for col in ["Cooler_Count"]:
    if col in outlet_master.columns:
        outlet_master[col] = pd.to_numeric(outlet_master[col], errors="coerce")

for col in ["Latitude", "Longitude"]:
    if col in outlet_coordinates.columns:
        outlet_coordinates[col] = pd.to_numeric(outlet_coordinates[col], errors="coerce")


# ============================================================
# BUILD OUTLET FEATURES FOR IMPUTATION
# ============================================================

# Monthly sales aggregation
monthly_sales = (
    sales.groupby(["Outlet_ID", "Year", "Month"])
    .agg(
        monthly_liters=("Volume_Liters", "sum"),
        monthly_bill_value=("Total_Bill_Value", "sum"),
        monthly_sku_count=("SKU_ID", "nunique"),
        monthly_avg_price_per_liter=("Price_Per_Liter", "mean"),
        monthly_transaction_count=("Outlet_ID", "count"),
    )
    .reset_index()
)

# Outlet-level sales statistics
outlet_sales_features = (
    monthly_sales.groupby("Outlet_ID")
    .agg(
        avg_monthly_liters=("monthly_liters", "mean"),
        median_monthly_liters=("monthly_liters", "median"),
        max_monthly_liters=("monthly_liters", "max"),
        sales_std=("monthly_liters", "std"),
        total_liters=("monthly_liters", "sum"),
        avg_bill_value=("monthly_bill_value", "mean"),
        max_bill_value=("monthly_bill_value", "max"),
        total_bill_value=("monthly_bill_value", "sum"),
        avg_sku_count=("monthly_sku_count", "mean"),
        max_sku_count=("monthly_sku_count", "max"),
        avg_transaction_count=("monthly_transaction_count", "mean"),
        total_transaction_count=("monthly_transaction_count", "sum"),
        avg_price_per_liter=("monthly_avg_price_per_liter", "mean"),
        active_months=("monthly_liters", "count"),
    )
    .reset_index()
)

outlet_sales_features["sales_cv"] = (
    outlet_sales_features["sales_std"]
    / outlet_sales_features["avg_monthly_liters"].replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan)

# Recent period features
recent_periods = monthly_sales.copy()
recent_periods["month_rank"] = recent_periods.groupby("Outlet_ID").cumcount(ascending=False)

recent_3 = (
    recent_periods[recent_periods["month_rank"] < 3]
    .groupby("Outlet_ID")
    .agg(
        recent_3_month_avg_liters=("monthly_liters", "mean"),
        recent_3_month_total_liters=("monthly_liters", "sum"),
    )
    .reset_index()
)

recent_6 = (
    recent_periods[recent_periods["month_rank"] < 6]
    .groupby("Outlet_ID")
    .agg(
        recent_6_month_avg_liters=("monthly_liters", "mean"),
        recent_6_month_total_liters=("monthly_liters", "sum"),
    )
    .reset_index()
)

# January features
january_features = (
    monthly_sales[monthly_sales["Month"] == 1]
    .groupby("Outlet_ID")
    .agg(
        avg_january_liters=("monthly_liters", "mean"),
        max_january_liters=("monthly_liters", "max"),
    )
    .reset_index()
)

# SKU diversity
sku_features = (
    sales.groupby("Outlet_ID")
    .agg(total_unique_skus=("SKU_ID", "nunique"))
    .reset_index()
)

# Primary distributor
distributor_map = (
    sales.groupby("Outlet_ID")
    .agg(
        Distributor_ID=("Distributor_ID", safe_mode),
        distributor_count=("Distributor_ID", "nunique"),
    )
    .reset_index()
)

# Return features
if len(returns) > 0:
    returns["return_liters_abs"] = returns["Volume_Liters"].abs()
    returns["return_value_abs"] = returns["Total_Bill_Value"].abs()
    return_features = (
        returns.groupby("Outlet_ID")
        .agg(
            return_record_count=("Outlet_ID", "count"),
            return_liters_abs=("return_liters_abs", "sum"),
            return_value_abs=("return_value_abs", "sum"),
        )
        .reset_index()
    )
else:
    return_features = pd.DataFrame(
        columns=["Outlet_ID", "return_record_count", "return_liters_abs", "return_value_abs"]
    )

# Holiday features per outlet (average over all months)
if len(holidays) > 0 and "Year" in holidays.columns:
    holidays_numeric = holidays.select_dtypes(include="number").columns.tolist()
    outlet_holiday_features = (
        monthly_sales.merge(holidays, on=["Year", "Month"], how="left")
        .groupby("Outlet_ID")
        .agg(
            {col: "mean" for col in holidays_numeric if col not in ["Year", "Month"]}
        )
        .reset_index()
    )
else:
    outlet_holiday_features = pd.DataFrame(columns=["Outlet_ID"])

# Seasonality per outlet (average score)
if len(seasonality) > 0 and "Distributor_ID" in seasonality.columns:
    monthly_sales_with_dist = monthly_sales.merge(distributor_map, on="Outlet_ID", how="left")
    monthly_sales_with_seasonality = monthly_sales_with_dist.merge(
        seasonality[["Distributor_ID", "Year", "Month", "seasonality_score"]],
        on=["Distributor_ID", "Year", "Month"],
        how="left",
    )
    outlet_seasonality_features = (
        monthly_sales_with_seasonality.groupby("Outlet_ID")
        .agg(avg_seasonality_score=("seasonality_score", "mean"))
        .reset_index()
    )
else:
    outlet_seasonality_features = pd.DataFrame(columns=["Outlet_ID"])


# ============================================================
# MERGE ALL FEATURES
# ============================================================

base = outlet_master.merge(outlet_coordinates, on="Outlet_ID", how="left", suffixes=("", "_coord"))
base = base.merge(outlet_sales_features, on="Outlet_ID", how="left")
base = base.merge(recent_3, on="Outlet_ID", how="left")
base = base.merge(recent_6, on="Outlet_ID", how="left")
base = base.merge(january_features, on="Outlet_ID", how="left")
base = base.merge(sku_features, on="Outlet_ID", how="left")
base = base.merge(distributor_map, on="Outlet_ID", how="left")
base = base.merge(return_features, on="Outlet_ID", how="left")
base = base.merge(outlet_holiday_features, on="Outlet_ID", how="left")

# Optional seasonality features
if len(outlet_seasonality_features) > 0:
    base = base.merge(outlet_seasonality_features, on="Outlet_ID", how="left")

# Fill numeric columns
numeric_cols = base.select_dtypes(include=["number"]).columns.tolist()
for col in numeric_cols:
    if col not in ["Latitude", "Longitude"]:
        base[col] = base[col].replace([np.inf, -np.inf], np.nan).fillna(0)

# Compute return ratios
base["return_ratio_liters"] = (
    base.get("return_liters_abs", 0)
    / base.get("total_liters", pd.Series(index=base.index, dtype=float)).replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).fillna(0)

base["return_ratio_value"] = (
    base.get("return_value_abs", 0)
    / base.get("total_bill_value", pd.Series(index=base.index, dtype=float)).replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).fillna(0)


# ============================================================
# PREPARE FOR IMPUTATION
# ============================================================

target_col = "Outlet_Size"

if target_col not in base.columns:
    raise ValueError("Outlet_Size column missing from outlet master.")

# Standardize outlet size values
base[target_col] = (
    base[target_col]
    .astype("string")
    .str.strip()
    .str.title()
)

base[target_col] = base[target_col].replace({
    "Nan": pd.NA,
    "None": pd.NA,
    "Null": pd.NA,
    "N/A": pd.NA,
    "Na": pd.NA,
})

# Remove invalid sizes
base.loc[
    base[target_col].notna() & ~base[target_col].isin(VALID_OUTLET_SIZES),
    target_col,
] = pd.NA

# Tracking columns
base["Outlet_Size_Imputed"] = 0
base["Outlet_Size_Confidence"] = np.nan
base["Outlet_Size_Review_Flag"] = "Original"

# Feature selection
numeric_features = [
    c
    for c in [
        "Cooler_Count",
        "Latitude",
        "Longitude",
        "avg_monthly_liters",
        "median_monthly_liters",
        "max_monthly_liters",
        "sales_std",
        "sales_cv",
        "total_liters",
        "avg_bill_value",
        "max_bill_value",
        "total_bill_value",
        "avg_sku_count",
        "max_sku_count",
        "total_unique_skus",
        "active_months",
        "recent_3_month_avg_liters",
        "recent_6_month_avg_liters",
        "avg_january_liters",
        "max_january_liters",
        "distributor_count",
        "return_record_count",
        "return_liters_abs",
        "return_ratio_liters",
        "return_ratio_value",
        "avg_seasonality_score",
    ]
    if c in base.columns
]

categorical_features = [c for c in ["Outlet_Type", "Distributor_ID"] if c in base.columns]

train = base[base[target_col].notna()].copy()
missing = base[base[target_col].isna()].copy()

print(f"\nOutlet Size Imputation")
print(f"Training rows: {len(train)}")
print(f"Missing rows: {len(missing)}")
print("\nClass distribution (train):")
print(train[target_col].value_counts())


# ============================================================
# IMPUTATION MODEL
# ============================================================

if len(missing) > 0 and len(train[target_col].unique()) >= 2 and numeric_features:
    X = train[numeric_features + categorical_features]
    y = train[target_col].astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Cross-validation with adaptive folds
    min_class_count = train[target_col].value_counts().min()
    n_splits = min(5, min_class_count)
    report_text = "Cross-validation not performed."

    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        oof_pred = np.zeros(len(X), dtype=int)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_enc), start=1):
            fold_model = clone(model)
            X_tr = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_tr = y_enc[train_idx]
            y_val = y_enc[val_idx]

            sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
            fold_model.fit(X_tr, y_tr, classifier__sample_weight=sample_weights)

            pred_val = fold_model.predict(X_val)
            oof_pred[val_idx] = pred_val

            acc = accuracy_score(y_val, pred_val)
            macro_f1 = f1_score(y_val, pred_val, average="macro", zero_division=0)

            fold_results.append({"fold": fold, "accuracy": acc, "macro_f1": macro_f1})
            print(
                f"  Fold {fold}: Accuracy={acc*100:.2f}% | Macro F1={macro_f1*100:.2f}%"
            )

        fold_results_df = pd.DataFrame(fold_results)
        fold_results_df.to_csv(
            SUMMARY_DIR / "outlet_size_imputation_cv_results.csv", index=False
        )

        report_text = classification_report(
            y_enc, oof_pred, target_names=le.classes_, zero_division=0
        )

    # Fit final model
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_enc)
    model.fit(X, y_enc, classifier__sample_weight=sample_weights)

    # Predict missing values
    X_missing = missing[numeric_features + categorical_features]
    pred_labels = le.inverse_transform(model.predict(X_missing))
    confidence = model.predict_proba(X_missing).max(axis=1)

    base.loc[missing.index, target_col] = pred_labels
    base.loc[missing.index, "Outlet_Size_Imputed"] = 1
    base.loc[missing.index, "Outlet_Size_Confidence"] = confidence
    base.loc[missing.index, "Outlet_Size_Review_Flag"] = np.where(
        confidence < LOW_CONFIDENCE_THRESHOLD,
        "Low Confidence - Review",
        "Imputed OK",
    )

    # Low confidence report
    low_confidence_rows = base[
        (base["Outlet_Size_Imputed"] == 1)
        & (base["Outlet_Size_Confidence"] < LOW_CONFIDENCE_THRESHOLD)
    ].copy()

    low_confidence_rows.to_csv(
        SUMMARY_DIR / "low_confidence_outlet_size_imputations.csv", index=False
    )

    # Diagnostics
    with open(
        SUMMARY_DIR / "outlet_size_imputation_diagnostics.txt", "w", encoding="utf-8"
    ) as f:
        f.write("Outlet Size Imputation Diagnostics\n")
        f.write("===================================\n\n")
        f.write(f"Training rows: {len(train)}\n")
        f.write(f"Missing rows imputed: {len(missing)}\n")
        f.write(f"Low confidence threshold: {LOW_CONFIDENCE_THRESHOLD}\n")
        f.write(f"Low confidence rows: {len(low_confidence_rows)}\n\n")
        f.write("Numeric features used:\n")
        for col in numeric_features:
            f.write(f"  - {col}\n")
        f.write("\nCategorical features used:\n")
        for col in categorical_features:
            f.write(f"  - {col}\n")
        f.write("\nClass distribution:\n")
        f.write(str(train[target_col].value_counts()))
        f.write("\n\nClassification report:\n")
        f.write(report_text)

else:
    print("No missing Outlet_Size to impute or insufficient training samples.")


# ============================================================
# SAVE OUTPUT
# ============================================================

base["Outlet_Size_Review_Flag"] = base["Outlet_Size_Review_Flag"].fillna("Original")

output_path = SILVER_DIR / "outlet_master_imputed.csv"
base.to_csv(output_path, index=False)

summary = {
    "total_outlets": len(base),
    "missing_outlet_size_after_imputation": int(base[target_col].isna().sum()),
    "outlet_size_imputed_rows": int(base["Outlet_Size_Imputed"].sum()),
    "low_confidence_outlet_size_rows": int(
        (base["Outlet_Size_Review_Flag"] == "Low Confidence - Review").sum()
    ),
}

pd.DataFrame([summary]).to_csv(
    SUMMARY_DIR / "outlet_size_imputation_summary.csv", index=False
)

print(f"\nImputation completed successfully.")
print(f"Output saved to: {output_path}")
print(f"Summary: {summary}")
