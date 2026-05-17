from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

GOLD_DIR = Path("processed/gold")
SUMMARY_DIR = Path("summaries")
SUBMISSION_DIR = Path("submissions")

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

TEAM_NAME = "teamname"  # change this to your real team name

RANDOM_STATE = 42

K_MIN = 3
K_MAX = 10
SILHOUETTE_SAMPLE_SIZE = 5000

MIN_CLUSTER_JAN_OBS = 30

SEASONALITY_ADJ_MIN = 0.90
SEASONALITY_ADJ_MAX = 1.10

HOLIDAY_ADJ_MIN = 0.97
HOLIDAY_ADJ_MAX = 1.03

DENSITY_ADJ_MIN = 0.95
DENSITY_ADJ_MAX = 1.05

RECENT_TREND_ADJ_MIN = 0.90
RECENT_TREND_ADJ_MAX = 1.10

CAPACITY_ADJ_MIN = 0.90
CAPACITY_ADJ_MAX = 1.10


# ============================================================
# HELPERS
# ============================================================

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def available_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_percentile(x, q):
    values = pd.to_numeric(x, errors="coerce").dropna()
    if len(values) == 0:
        return np.nan
    return np.percentile(values, q)


def safe_divide_series(num, den, default=1.0):
    result = num / den.replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(default)


def clip(s, lower, upper):
    return s.clip(lower=lower, upper=upper)


# ============================================================
# LOAD GOLD DATA
# ============================================================

def load_gold_data():
    outlet_path = GOLD_DIR / "outlet_modeling_table_gold.csv"
    monthly_path = GOLD_DIR / "monthly_sales_gold.csv"

    outlets = read_csv_required(outlet_path)
    monthly_sales = read_csv_required(monthly_path)

    print("Loaded Gold data:")
    print("-", outlet_path, outlets.shape)
    print("-", monthly_path, monthly_sales.shape)

    if "Outlet_ID" not in outlets.columns:
        raise ValueError("Outlet_ID missing in outlet_modeling_table_gold.csv")

    if "Outlet_ID" not in monthly_sales.columns:
        raise ValueError("Outlet_ID missing in monthly_sales_gold.csv")

    if outlets["Outlet_ID"].duplicated().any():
        dup_count = int(outlets["Outlet_ID"].duplicated().sum())
        raise ValueError(f"Duplicate Outlet_ID rows found in outlet_modeling_table_gold.csv: {dup_count}")

    for col in ["Year", "Month", "monthly_liters"]:
        if col not in monthly_sales.columns:
            raise ValueError(f"{col} missing in monthly_sales_gold.csv")

    monthly_sales["Year"] = pd.to_numeric(monthly_sales["Year"], errors="coerce").astype("Int64")
    monthly_sales["Month"] = pd.to_numeric(monthly_sales["Month"], errors="coerce").astype("Int64")
    monthly_sales["monthly_liters"] = pd.to_numeric(monthly_sales["monthly_liters"], errors="coerce")

    return outlets, monthly_sales


# ============================================================
# FEATURE SELECTION FOR CLUSTERING
# ============================================================

def get_clustering_features(outlets: pd.DataFrame):
    numeric_candidates = [
        "Cooler_Count",
        "Latitude",
        "Longitude",

        "avg_monthly_liters",
        "median_monthly_liters",
        "avg_monthly_liters_excluding_apr_dec",
        "max_monthly_liters",
        "p75_monthly_liters",
        "p90_monthly_liters",
        "sales_std",
        "sales_cv",
        "total_liters",

        "avg_bill_value",
        "total_bill_value",

        "avg_sku_count",
        "max_sku_count",
        "total_unique_skus",
        "active_months",

        "recent_3_month_avg_liters",
        "recent_6_month_avg_liters",

        "avg_january_liters",
        "max_january_liters",

        "april_spike_ratio",
        "december_spike_ratio",
        "seasonal_peak_liters",
        "peak_to_median_ratio",
        "seasonal_spike_flag",

        "return_ratio_liters",
        "return_ratio_value",
        "billing_anomaly_count",

        "avg_holiday_intensity_score",
        "avg_seasonality_adjusted_holiday_score",
        "avg_distributor_month_context_score",
        "avg_seasonality_multiplier_calibrated",

        "favorable_holiday_month_count",
        "unfavorable_holiday_month_count",
        "festive_favorable_month_count",

        "avg_monthly_density",
        "max_monthly_density",
        "avg_january_density",
        "max_january_density",
        "avg_base_population",
        "avg_mobility_factor",
        "avg_tourism_factor",

        "jan_2026_adjusted_holiday_score_proxy",
        "jan_2026_seasonality_multiplier_used",
        "jan_2026_monthly_density",
        "jan_2026_base_population",
        "jan_2026_mobility_factor",
        "jan_2026_tourism_factor",
    ]

    categorical_candidates = [
        "Outlet_Size",
        "Outlet_Type",
        "Distributor_ID",
        "province",
        "jan_2026_seasonality_index",
    ]

    numeric_features = available_columns(outlets, numeric_candidates)
    categorical_features = available_columns(outlets, categorical_candidates)

    if len(numeric_features) == 0:
        raise ValueError("No numeric clustering features found.")

    print("\nNumeric clustering features:")
    print(numeric_features)

    print("\nCategorical clustering features:")
    print(categorical_features)

    return numeric_features, categorical_features


# ============================================================
# CLUSTERING MODEL
# ============================================================

def run_clustering(outlets: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]):
    outlets = outlets.copy()

    outlets = ensure_numeric(outlets, numeric_features)

    for col in categorical_features:
        outlets[col] = outlets[col].fillna("Unknown").astype(str)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    X = outlets[numeric_features + categorical_features]
    X_processed = preprocessor.fit_transform(X)

    print("\nProcessed clustering matrix shape:", X_processed.shape)

    n_rows = X_processed.shape[0]

    if n_rows > SILHOUETTE_SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(n_rows, size=SILHOUETTE_SAMPLE_SIZE, replace=False)
        X_sample = X_processed[sample_idx]
    else:
        X_sample = X_processed

    k_results = []

    for k in range(K_MIN, K_MAX + 1):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20
        )

        labels_sample = model.fit_predict(X_sample)

        sil = silhouette_score(X_sample, labels_sample)

        k_results.append({
            "k": k,
            "silhouette_score": sil,
            "inertia": model.inertia_
        })

        print(f"k={k} | silhouette={sil:.4f} | inertia={model.inertia_:.2f}")

    k_results_df = pd.DataFrame(k_results)

    best_k = int(
        k_results_df
        .sort_values(["silhouette_score", "k"], ascending=[False, True])
        .iloc[0]["k"]
    )

    print("\nSelected best k:", best_k)

    k_results_df.to_csv(
        SUMMARY_DIR / "clustering_k_selection.csv",
        index=False
    )

    final_model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=30
    )

    outlets["cluster_id"] = final_model.fit_predict(X_processed)

    outlets[["Outlet_ID", "cluster_id"]].to_csv(
        GOLD_DIR / "outlet_clusters_gold.csv",
        index=False
    )

    print("\nCluster distribution:")
    print(outlets["cluster_id"].value_counts().sort_index())

    return outlets, best_k


# ============================================================
# CLUSTER PROFILES
# ============================================================

def save_cluster_profiles(outlets: pd.DataFrame):
    profile_cols = [
        "avg_monthly_liters",
        "median_monthly_liters",
        "p90_monthly_liters",
        "max_monthly_liters",
        "avg_january_liters",
        "max_january_liters",
        "Cooler_Count",
        "avg_sku_count",
        "total_unique_skus",
        "return_ratio_liters",
        "billing_anomaly_count",
        "avg_holiday_intensity_score",
        "avg_seasonality_adjusted_holiday_score",
        "avg_distributor_month_context_score",
        "avg_monthly_density",
        "avg_january_density",
    ]

    profile_cols = available_columns(outlets, profile_cols)

    cluster_profiles = (
        outlets
        .groupby("cluster_id")
        .agg(
            outlet_count=("Outlet_ID", "count"),
            **{
                f"{col}_mean": (col, "mean")
                for col in profile_cols
            }
        )
        .reset_index()
    )

    cluster_profiles.to_csv(
        SUMMARY_DIR / "cluster_profiles.csv",
        index=False
    )


# ============================================================
# CLUSTER JANUARY BENCHMARKS
# ============================================================

def create_cluster_benchmarks(outlets: pd.DataFrame, monthly_sales: pd.DataFrame):
    monthly_clustered = monthly_sales.merge(
        outlets[["Outlet_ID", "cluster_id"]],
        on="Outlet_ID",
        how="left"
    )

    monthly_clustered = monthly_clustered[monthly_clustered["cluster_id"].notna()].copy()
    monthly_clustered["cluster_id"] = monthly_clustered["cluster_id"].astype(int)

    monthly_clustered.to_csv(
        GOLD_DIR / "monthly_sales_with_clusters_gold.csv",
        index=False
    )

    jan_sales = monthly_clustered[monthly_clustered["Month"] == 1].copy()

    cluster_overall = (
        monthly_clustered
        .groupby("cluster_id")
        .agg(
            cluster_overall_obs=("monthly_liters", "count"),
            cluster_overall_mean=("monthly_liters", "mean"),
            cluster_overall_p75=("monthly_liters", lambda x: safe_percentile(x, 75)),
            cluster_overall_p90=("monthly_liters", lambda x: safe_percentile(x, 90)),
            cluster_overall_p95=("monthly_liters", lambda x: safe_percentile(x, 95)),
            cluster_overall_p98=("monthly_liters", lambda x: safe_percentile(x, 98)),
            cluster_overall_max=("monthly_liters", "max"),
        )
        .reset_index()
    )

    cluster_jan = (
        jan_sales
        .groupby("cluster_id")
        .agg(
            cluster_jan_obs=("monthly_liters", "count"),
            cluster_jan_mean=("monthly_liters", "mean"),
            cluster_jan_p75=("monthly_liters", lambda x: safe_percentile(x, 75)),
            cluster_jan_p90=("monthly_liters", lambda x: safe_percentile(x, 90)),
            cluster_jan_p95=("monthly_liters", lambda x: safe_percentile(x, 95)),
            cluster_jan_p98=("monthly_liters", lambda x: safe_percentile(x, 98)),
            cluster_jan_max=("monthly_liters", "max"),
        )
        .reset_index()
    )

    cluster_benchmarks = cluster_overall.merge(
        cluster_jan,
        on="cluster_id",
        how="left"
    )

    cluster_benchmarks["cluster_jan_ratio"] = (
        cluster_benchmarks["cluster_jan_mean"]
        / cluster_benchmarks["cluster_overall_mean"].replace(0, np.nan)
    )

    cluster_benchmarks["cluster_jan_ratio"] = (
        cluster_benchmarks["cluster_jan_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(0.75, 1.25)
    )

    # If enough January observations exist, use January P90 directly.
    # Otherwise use overall P90 adjusted by historical January ratio.
    cluster_benchmarks["cluster_january_ceiling"] = np.where(
        cluster_benchmarks["cluster_jan_obs"].fillna(0) >= MIN_CLUSTER_JAN_OBS,
        cluster_benchmarks["cluster_jan_p90"],
        cluster_benchmarks["cluster_overall_p90"] * cluster_benchmarks["cluster_jan_ratio"]
    )

    cluster_benchmarks["cluster_january_cap"] = np.where(
        cluster_benchmarks["cluster_jan_obs"].fillna(0) >= MIN_CLUSTER_JAN_OBS,
        cluster_benchmarks["cluster_jan_p98"],
        cluster_benchmarks["cluster_overall_p98"] * cluster_benchmarks["cluster_jan_ratio"]
    )

    cluster_benchmarks["january_benchmark_source"] = np.where(
        cluster_benchmarks["cluster_jan_obs"].fillna(0) >= MIN_CLUSTER_JAN_OBS,
        "cluster_january_p90",
        "cluster_overall_p90_adjusted_by_january_ratio"
    )

    # Historical January context baseline by cluster.
    monthly_clustered["hist_seasonality_multiplier"] = 1.0

    for col in [
        "seasonality_multiplier_calibrated",
        "holiday_seasonality_multiplier_used",
        "seasonality_multiplier"
    ]:
        if col in monthly_clustered.columns:
            monthly_clustered["hist_seasonality_multiplier"] = pd.to_numeric(
                monthly_clustered[col],
                errors="coerce"
            ).fillna(1.0)
            break

    if "seasonality_adjusted_holiday_score" in monthly_clustered.columns:
        monthly_clustered["hist_adjusted_holiday_score"] = pd.to_numeric(
            monthly_clustered["seasonality_adjusted_holiday_score"],
            errors="coerce"
        ).fillna(0)
    elif "distributor_month_demand_context_score" in monthly_clustered.columns:
        monthly_clustered["hist_adjusted_holiday_score"] = pd.to_numeric(
            monthly_clustered["distributor_month_demand_context_score"],
            errors="coerce"
        ).fillna(0)
    elif "holiday_intensity_score" in monthly_clustered.columns:
        monthly_clustered["hist_adjusted_holiday_score"] = pd.to_numeric(
            monthly_clustered["holiday_intensity_score"],
            errors="coerce"
        ).fillna(0)
    else:
        monthly_clustered["hist_adjusted_holiday_score"] = 0

    if "monthly_density" in monthly_clustered.columns:
        monthly_clustered["hist_monthly_density"] = pd.to_numeric(
            monthly_clustered["monthly_density"],
            errors="coerce"
        )
    else:
        monthly_clustered["hist_monthly_density"] = np.nan

    jan_context = monthly_clustered[monthly_clustered["Month"] == 1].copy()

    cluster_jan_context = (
        jan_context
        .groupby("cluster_id")
        .agg(
            cluster_hist_jan_seasonality_multiplier=("hist_seasonality_multiplier", "mean"),
            cluster_hist_jan_adjusted_holiday_score=("hist_adjusted_holiday_score", "mean"),
            cluster_hist_jan_density=("hist_monthly_density", "mean"),
        )
        .reset_index()
    )

    cluster_benchmarks = cluster_benchmarks.merge(
        cluster_jan_context,
        on="cluster_id",
        how="left"
    )

    cluster_benchmarks["cluster_hist_jan_seasonality_multiplier"] = (
        cluster_benchmarks["cluster_hist_jan_seasonality_multiplier"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    cluster_benchmarks["cluster_hist_jan_adjusted_holiday_score"] = (
        cluster_benchmarks["cluster_hist_jan_adjusted_holiday_score"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    cluster_benchmarks["cluster_hist_jan_density"] = (
        cluster_benchmarks["cluster_hist_jan_density"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(cluster_benchmarks["cluster_hist_jan_density"].median())
    )

    # Cluster median outlet feature benchmark for capacity adjustment.
    if "p90_monthly_liters" in outlets.columns:
        cluster_capacity = (
            outlets
            .groupby("cluster_id")
            .agg(cluster_median_outlet_p90=("p90_monthly_liters", "median"))
            .reset_index()
        )

        cluster_benchmarks = cluster_benchmarks.merge(
            cluster_capacity,
            on="cluster_id",
            how="left"
        )
    else:
        cluster_benchmarks["cluster_median_outlet_p90"] = np.nan

    cluster_benchmarks.to_csv(
        SUMMARY_DIR / "cluster_potential_benchmarks.csv",
        index=False
    )

    return cluster_benchmarks


# ============================================================
# JANUARY 2026 POTENTIAL CALCULATION
# ============================================================

def calculate_january_2026_potential(outlets: pd.DataFrame, cluster_benchmarks: pd.DataFrame):
    potential = outlets.merge(
        cluster_benchmarks[
            [
                "cluster_id",
                "cluster_jan_obs",
                "cluster_january_ceiling",
                "cluster_january_cap",
                "january_benchmark_source",
                "cluster_hist_jan_seasonality_multiplier",
                "cluster_hist_jan_adjusted_holiday_score",
                "cluster_hist_jan_density",
                "cluster_median_outlet_p90",
            ]
        ],
        on="cluster_id",
        how="left"
    )

    # ------------------------
    # Seasonality adjustment
    # ------------------------
    if "jan_2026_seasonality_multiplier_used" in potential.columns:
        potential["jan_2026_multiplier_used"] = pd.to_numeric(
            potential["jan_2026_seasonality_multiplier_used"],
            errors="coerce"
        ).fillna(1.0)
    elif "jan_2026_seasonality_multiplier" in potential.columns:
        potential["jan_2026_multiplier_used"] = pd.to_numeric(
            potential["jan_2026_seasonality_multiplier"],
            errors="coerce"
        ).fillna(1.0)
    else:
        potential["jan_2026_multiplier_used"] = 1.0

    potential["seasonality_relative_adjustment"] = (
        potential["jan_2026_multiplier_used"]
        / potential["cluster_hist_jan_seasonality_multiplier"].replace(0, np.nan)
    )

    potential["seasonality_relative_adjustment"] = (
        potential["seasonality_relative_adjustment"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(SEASONALITY_ADJ_MIN, SEASONALITY_ADJ_MAX)
    )

    # ------------------------
    # Holiday/context adjustment
    # ------------------------
    if "jan_2026_adjusted_holiday_score_proxy" in potential.columns:
        potential["jan_2026_adjusted_holiday_score_used"] = pd.to_numeric(
            potential["jan_2026_adjusted_holiday_score_proxy"],
            errors="coerce"
        ).fillna(0)
    elif "jan_2026_distributor_month_context_score" in potential.columns:
        potential["jan_2026_adjusted_holiday_score_used"] = pd.to_numeric(
            potential["jan_2026_distributor_month_context_score"],
            errors="coerce"
        ).fillna(0)
    elif "jan_2026_holiday_intensity_proxy" in potential.columns:
        potential["jan_2026_adjusted_holiday_score_used"] = pd.to_numeric(
            potential["jan_2026_holiday_intensity_proxy"],
            errors="coerce"
        ).fillna(0)
    else:
        potential["jan_2026_adjusted_holiday_score_used"] = 0

    holiday_diff = (
        potential["jan_2026_adjusted_holiday_score_used"]
        - potential["cluster_hist_jan_adjusted_holiday_score"]
    )

    # Holiday should support the estimate, not dominate it.
    potential["holiday_relative_adjustment"] = 1 + (holiday_diff / 10.0) * 0.02

    potential["holiday_relative_adjustment"] = (
        potential["holiday_relative_adjustment"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(HOLIDAY_ADJ_MIN, HOLIDAY_ADJ_MAX)
    )

    # ------------------------
    # Population density adjustment
    # ------------------------
    if "jan_2026_monthly_density" in potential.columns:
        potential["jan_2026_density_used"] = pd.to_numeric(
            potential["jan_2026_monthly_density"],
            errors="coerce"
        )
    elif "avg_january_density" in potential.columns:
        potential["jan_2026_density_used"] = pd.to_numeric(
            potential["avg_january_density"],
            errors="coerce"
        )
    elif "avg_monthly_density" in potential.columns:
        potential["jan_2026_density_used"] = pd.to_numeric(
            potential["avg_monthly_density"],
            errors="coerce"
        )
    else:
        potential["jan_2026_density_used"] = np.nan

    density_ratio = (
        potential["jan_2026_density_used"]
        / potential["cluster_hist_jan_density"].replace(0, np.nan)
    )

    density_ratio = (
        density_ratio
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    # Convert density ratio into a small capped adjustment.
    potential["density_relative_adjustment"] = 1 + ((density_ratio - 1) * 0.20)

    potential["density_relative_adjustment"] = (
        potential["density_relative_adjustment"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(DENSITY_ADJ_MIN, DENSITY_ADJ_MAX)
    )

    # ------------------------
    # Recent trend adjustment
    # ------------------------
    if {"recent_3_month_avg_liters", "recent_6_month_avg_liters"}.issubset(potential.columns):
        potential["recent_trend_adjustment"] = (
            pd.to_numeric(potential["recent_3_month_avg_liters"], errors="coerce")
            / pd.to_numeric(potential["recent_6_month_avg_liters"], errors="coerce").replace(0, np.nan)
        )
    else:
        potential["recent_trend_adjustment"] = 1.0

    potential["recent_trend_adjustment"] = (
        potential["recent_trend_adjustment"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(RECENT_TREND_ADJ_MIN, RECENT_TREND_ADJ_MAX)
    )

    # ------------------------
    # Capacity adjustment
    # ------------------------
    if "p90_monthly_liters" in potential.columns:
        outlet_p90 = pd.to_numeric(potential["p90_monthly_liters"], errors="coerce")
        cluster_median_p90 = pd.to_numeric(
            potential["cluster_median_outlet_p90"],
            errors="coerce"
        ).replace(0, np.nan)

        capacity_ratio = outlet_p90 / cluster_median_p90
    else:
        capacity_ratio = pd.Series(1.0, index=potential.index)

    capacity_ratio = (
        capacity_ratio
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    potential["capacity_relative_adjustment"] = 1 + ((capacity_ratio - 1) * 0.15)

    potential["capacity_relative_adjustment"] = (
        potential["capacity_relative_adjustment"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(CAPACITY_ADJ_MIN, CAPACITY_ADJ_MAX)
    )

    # ------------------------
    # Raw potential
    # ------------------------
    potential["raw_jan_2026_potential"] = (
        potential["cluster_january_ceiling"]
        * potential["seasonality_relative_adjustment"]
        * potential["holiday_relative_adjustment"]
        * potential["density_relative_adjustment"]
        * potential["recent_trend_adjustment"]
        * potential["capacity_relative_adjustment"]
    )

    # ------------------------
    # Lower bound: own proven January peak
    # ------------------------
    if "max_january_liters" in potential.columns:
        potential["own_january_peak_liters"] = pd.to_numeric(
            potential["max_january_liters"],
            errors="coerce"
        ).fillna(0)
    else:
        potential["own_january_peak_liters"] = 0

    if "avg_january_liters" in potential.columns:
        potential["own_january_peak_liters"] = np.maximum(
            potential["own_january_peak_liters"],
            pd.to_numeric(potential["avg_january_liters"], errors="coerce").fillna(0)
        )

    # ------------------------
    # Upper cap: cluster high performer cap
    # ------------------------
    potential["cluster_january_cap"] = pd.to_numeric(
        potential["cluster_january_cap"],
        errors="coerce"
    )

    potential["cluster_january_ceiling"] = pd.to_numeric(
        potential["cluster_january_ceiling"],
        errors="coerce"
    )

    potential["potential_upper_cap"] = np.maximum(
        potential["cluster_january_cap"].fillna(potential["cluster_january_ceiling"] * 1.20),
        potential["own_january_peak_liters"]
    )

    potential["Maximum_Monthly_Liters"] = potential["raw_jan_2026_potential"]

    potential["Maximum_Monthly_Liters"] = np.maximum(
        potential["Maximum_Monthly_Liters"],
        potential["own_january_peak_liters"]
    )

    potential["Maximum_Monthly_Liters"] = np.minimum(
        potential["Maximum_Monthly_Liters"],
        potential["potential_upper_cap"]
    )

    potential["Maximum_Monthly_Liters"] = (
        potential["Maximum_Monthly_Liters"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(potential["own_january_peak_liters"])
        .clip(lower=0)
        .round(2)
    )

    # ------------------------
    # Potential gap
    # ------------------------
    if "avg_january_liters" in potential.columns:
        potential["current_january_baseline_liters"] = pd.to_numeric(
            potential["avg_january_liters"],
            errors="coerce"
        ).fillna(0)
    elif "avg_monthly_liters" in potential.columns:
        potential["current_january_baseline_liters"] = pd.to_numeric(
            potential["avg_monthly_liters"],
            errors="coerce"
        ).fillna(0)
    else:
        potential["current_january_baseline_liters"] = 0

    potential["potential_gap_liters"] = (
        potential["Maximum_Monthly_Liters"]
        - potential["current_january_baseline_liters"]
    ).clip(lower=0).round(2)

    potential["potential_gap_ratio"] = (
        potential["potential_gap_liters"]
        / potential["current_january_baseline_liters"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0).round(4)

    return potential


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_prediction_outputs(potential: pd.DataFrame, best_k: int):
    potential_cols = [
        "Outlet_ID",
        "cluster_id",
        "Maximum_Monthly_Liters",
        "current_january_baseline_liters",
        "potential_gap_liters",
        "potential_gap_ratio",

        "cluster_jan_obs",
        "cluster_january_ceiling",
        "cluster_january_cap",
        "january_benchmark_source",
        "own_january_peak_liters",

        "jan_2026_multiplier_used",
        "seasonality_relative_adjustment",
        "jan_2026_adjusted_holiday_score_used",
        "holiday_relative_adjustment",
        "jan_2026_density_used",
        "density_relative_adjustment",
        "recent_trend_adjustment",
        "capacity_relative_adjustment",
        "raw_jan_2026_potential",
    ]

    potential_cols = available_columns(potential, potential_cols)

    potential[potential_cols].to_csv(
        GOLD_DIR / "outlet_january_2026_potential_gold.csv",
        index=False
    )

    submission = potential[["Outlet_ID", "Maximum_Monthly_Liters"]].copy()

    submission_path = SUBMISSION_DIR / f"{TEAM_NAME}_predictions.csv"
    submission.to_csv(submission_path, index=False)

    # Generic copy
    submission.to_csv(GOLD_DIR / "teamname_predictions.csv", index=False)

    potential_summary = {
        "total_outlets": len(potential),
        "unique_outlets": potential["Outlet_ID"].nunique(),
        "duplicate_outlet_rows": int(potential["Outlet_ID"].duplicated().sum()),
        "selected_k": best_k,
        "cluster_count": int(potential["cluster_id"].nunique()),
        "min_prediction": float(potential["Maximum_Monthly_Liters"].min()),
        "mean_prediction": float(potential["Maximum_Monthly_Liters"].mean()),
        "median_prediction": float(potential["Maximum_Monthly_Liters"].median()),
        "max_prediction": float(potential["Maximum_Monthly_Liters"].max()),
        "missing_predictions": int(potential["Maximum_Monthly_Liters"].isna().sum()),
        "mean_potential_gap": float(potential["potential_gap_liters"].mean()),
        "outlets_with_positive_gap": int((potential["potential_gap_liters"] > 0).sum()),
    }

    pd.DataFrame([potential_summary]).to_csv(
        SUMMARY_DIR / "potential_summary.csv",
        index=False
    )

    cluster_prediction_summary = (
        potential
        .groupby("cluster_id")
        .agg(
            outlet_count=("Outlet_ID", "count"),
            avg_prediction=("Maximum_Monthly_Liters", "mean"),
            median_prediction=("Maximum_Monthly_Liters", "median"),
            max_prediction=("Maximum_Monthly_Liters", "max"),
            avg_gap=("potential_gap_liters", "mean"),
        )
        .reset_index()
    )

    cluster_prediction_summary.to_csv(
        SUMMARY_DIR / "cluster_prediction_summary.csv",
        index=False
    )

    print("\nJanuary 2026 potential model completed.")
    print("Saved:")
    print("-", GOLD_DIR / "outlet_clusters_gold.csv")
    print("-", GOLD_DIR / "monthly_sales_with_clusters_gold.csv")
    print("-", GOLD_DIR / "outlet_january_2026_potential_gold.csv")
    print("-", submission_path)
    print("-", SUMMARY_DIR / "potential_summary.csv")
    print("-", SUMMARY_DIR / "cluster_prediction_summary.csv")

    print("\nSubmission preview:")
    print(submission.head())


# ============================================================
# MAIN
# ============================================================

def run_prediction_pipeline():
    outlets, monthly_sales = load_gold_data()

    numeric_features, categorical_features = get_clustering_features(outlets)

    clustered_outlets, best_k = run_clustering(
        outlets=outlets,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    save_cluster_profiles(clustered_outlets)

    cluster_benchmarks = create_cluster_benchmarks(
        outlets=clustered_outlets,
        monthly_sales=monthly_sales
    )

    potential = calculate_january_2026_potential(
        outlets=clustered_outlets,
        cluster_benchmarks=cluster_benchmarks
    )

    save_prediction_outputs(
        potential=potential,
        best_k=best_k
    )


if __name__ == "__main__":
    run_prediction_pipeline()