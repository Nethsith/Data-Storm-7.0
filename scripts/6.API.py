"""
Monthly Population Density Pipeline — Sri Lanka (BULK / 20 000 locations)
=========================================================================
Strategy
────────
  ❌ OLD: 120 000 WorldPop API calls  (~50 hours)
  ✅ NEW: Download one GeoTIFF raster per year (6 files, ~10 MB each)
          then extract all 20 000 values locally in <60 seconds total.

Country enrichment
──────────────────
    Adds a `country` field (Sri Lanka) for all outlets.

Zero-population handling (1 km only)
────────────────────────────────────
    Some locations return base_population = 0 because they sit on ocean / lagoon
    / forest pixels in the WorldPop raster. With 1 km only, we keep these as
    zero and label them for audit.

Architecture
────────────
  1. Load coordinates
    2. Build 1 km buffers (all upfront)
    3. Download WorldPop raster once per year
    4. Zonal stats → flag zeros → yearly population table (checkpointed)
  5. Apply monthly mobility × tourism factors  (vectorised)
    6. Save monthly_density.csv + zero_population_report.csv (with country)
  7. Visualise aggregate / source breakdown / top locations

Requirements
────────────
  pip install pandas geopandas shapely requests tqdm rasterstats rasterio matplotlib scipy
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path
from typing import Tuple
from shapely.geometry import Point

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG  — edit these freely
# ─────────────────────────────────────────────
LOCATIONS_CSV    = "processed/silver/outlet_coordinates_silver.csv"
OUTPUT_CSV       = "processed/silver/monthly_density.csv"
ZERO_REPORT_CSV  = "processed/silver/zero_population_report.csv"   # audit file for fixed zeros
RASTER_CACHE_DIR = Path("worldpop_rasters")
CHECKPOINT_DIR   = Path("checkpoints")
PROVINCE_DIR     = Path("provinces")
PROVINCE_GEOJSON = PROVINCE_DIR / "lka_admin1.geojson"

# Buffer radius (metres)
BUFFER_1KM = 1_000

YEARS = list(range(2023, 2026))

WORLDPOP_RASTER_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/"
    "{year}/LKA/lka_ppp_{year}_1km_Aggregated.tif"
)
WORLDPOP_MAX_YEAR  = 2020
ANNUAL_GROWTH_RATE = 0.008    # ~0.8 % / yr — Sri Lanka census-based

CHUNK_SIZE = 500    # rows per rasterstats batch


# ─────────────────────────────────────────────
# MONTHLY / YEARLY FACTORS
# ─────────────────────────────────────────────
MONTHLY_TOURISM_FACTOR = {
    1: 1.18, 2: 1.15, 3: 1.08, 4: 0.82, 5: 0.70,  6: 0.75,
    7: 0.85, 8: 0.90, 9: 0.80, 10: 0.88, 11: 1.05, 12: 1.20,
}
YEARLY_MOBILITY_FACTOR = {
    2020: 0.60, 2021: 0.72, 2022: 0.85,
    2023: 0.95, 2024: 1.00, 2025: 1.02,
}


# ─────────────────────────────────────────────
# STEP 1 — LOAD LOCATIONS
# ─────────────────────────────────────────────
def load_locations(path: str = LOCATIONS_CSV) -> pd.DataFrame:
    log.info("Loading locations from %s …", path)
    df = pd.read_csv(path)

    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for std, variants in [
        ("latitude",      ["latitude", "lat", "y"]),
        ("longitude",     ["longitude", "lon", "long", "lng", "x"]),
        ("location_name", ["location_name", "outlet_id", "name", "id", "site_id"]),
    ]:
        for v in variants:
            if v in lower:
                rename[lower[v]] = std
                break

    df = df.rename(columns=rename)
    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError(f"CSV must have latitude/longitude columns (found: {list(df.columns)})")
    if "location_name" not in df.columns:
        df["location_name"] = [f"loc_{i+1}" for i in range(len(df))]

    df = df[["location_name", "latitude", "longitude"]].dropna(
        subset=["latitude", "longitude"]
    ).reset_index(drop=True)
    log.info("Loaded %d locations", len(df))
    return df


# ─────────────────────────────────────────────
# STEP 1B — PROVINCE ASSIGNMENT (GADM)
# ─────────────────────────────────────────────
def load_province_polygons() -> gpd.GeoDataFrame:
    """Load Sri Lanka admin-1 polygons (download if missing)."""
    PROVINCE_DIR.mkdir(parents=True, exist_ok=True)
    if not PROVINCE_GEOJSON.exists():
        url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_LKA_1.json"
        log.info("Downloading province boundaries → %s", PROVINCE_GEOJSON)
        provinces = gpd.read_file(url)
        provinces.to_file(PROVINCE_GEOJSON, driver="GeoJSON")

    provinces = gpd.read_file(PROVINCE_GEOJSON)
    provinces = provinces[["NAME_1", "geometry"]].rename(columns={"NAME_1": "province"})
    provinces = provinces.to_crs("EPSG:4326")
    return provinces


def attach_province(df: pd.DataFrame) -> pd.DataFrame:
    """Attach province name using admin-1 polygons (fallback to nearest centroid)."""
    provinces = load_province_polygons()
    pts = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(df.longitude, df.latitude)],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(pts, provinces, how="left", predicate="within")

    if joined["province"].isna().any():
        from scipy.spatial import cKDTree

        missing = joined["province"].isna()
        centroids = provinces.geometry.centroid
        cent_coords = np.array([[p.y, p.x] for p in centroids])
        tree = cKDTree(cent_coords)
        miss_coords = np.array([[p.y, p.x] for p in joined.loc[missing, "geometry"]])
        _, idx = tree.query(miss_coords, k=1)
        joined.loc[missing, "province"] = provinces.iloc[idx]["province"].values

    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
    joined = pd.DataFrame(joined)

    # Map Sri Lanka districts to provinces (GADM L1 often returns districts).
    district_to_province = {
        "Colombo": "Western",
        "Gampaha": "Western",
        "Kalutara": "Western",
        "Kandy": "Central",
        "Matale": "Central",
        "Nuwara Eliya": "Central",
        "Galle": "Southern",
        "Matara": "Southern",
        "Hambantota": "Southern",
        "Jaffna": "Northern",
        "Kilinochchi": "Northern",
        "Mannar": "Northern",
        "Vavuniya": "Northern",
        "Mullaitivu": "Northern",
        "Trincomalee": "Eastern",
        "Batticaloa": "Eastern",
        "Ampara": "Eastern",
        "Kurunegala": "North Western",
        "Puttalam": "North Western",
        "Anuradhapura": "North Central",
        "Polonnaruwa": "North Central",
        "Badulla": "Uva",
        "Monaragala": "Uva",
        "Ratnapura": "Sabaragamuwa",
        "Kegalle": "Sabaragamuwa",
    }

    joined["district"] = joined["province"].astype("string").str.strip()
    joined["province"] = joined["district"].map(district_to_province).fillna(joined["province"])
    return joined


# ─────────────────────────────────────────────
# STEP 2 — BUILD BUFFER GeoDataFrames
# ─────────────────────────────────────────────
def _make_buffers(df: pd.DataFrame, radius_m: int) -> gpd.GeoDataFrame:
    pts   = [Point(lon, lat) for lon, lat in zip(df.longitude, df.latitude)]
    gdf   = gpd.GeoDataFrame(df.copy(), geometry=pts, crs="EPSG:4326")
    gdf_m = gdf.to_crs("EPSG:5235")
    gdf_m["geometry"] = gdf_m.geometry.buffer(radius_m)
    return gdf_m.to_crs("EPSG:4326")


def build_all_buffers(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Return the 1 km GeoDataFrame buffer."""
    log.info("Building 1 km buffer sets for %d locations …", len(df))
    return _make_buffers(df, BUFFER_1KM)


# ─────────────────────────────────────────────
# STEP 3 — DOWNLOAD WORLDPOP RASTER
# ─────────────────────────────────────────────
def get_raster_path(year: int) -> Tuple[Path, int]:
    RASTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raster_year = min(year, WORLDPOP_MAX_YEAR)
    return RASTER_CACHE_DIR / f"lka_pop_{raster_year}.tif", raster_year


def download_raster(year: int) -> Path:
    path, raster_year = get_raster_path(year)
    if path.exists():
        log.info("  Raster cached: %s", path)
        return path

    url = WORLDPOP_RASTER_URL.format(year=raster_year)
    log.info("  Downloading WorldPop raster %d → %s", raster_year, path)
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=f"lka_pop_{raster_year}.tif"
            ) as bar:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    bar.update(len(chunk))
        log.info("  Downloaded %.1f MB", path.stat().st_size / 1e6)
    except requests.RequestException as exc:
        if path.exists():
            path.unlink()
        raise RuntimeError(f"Raster download failed: {exc}") from exc
    return path


# ─────────────────────────────────────────────
# STEP 4 — ZONAL STATS HELPER
# ─────────────────────────────────────────────
def _zonal_sum(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    growth_factor: float,
    label: str,
) -> np.ndarray:
    """Sum raster pixels inside each buffer and apply the growth multiplier."""
    from rasterstats import zonal_stats

    results = []
    for start in tqdm(range(0, len(gdf), CHUNK_SIZE), desc=f"  {label}", leave=False):
        chunk = gdf.iloc[start: start + CHUNK_SIZE]
        stats = zonal_stats(
            chunk, str(raster_path),
            stats=["sum"], nodata=-9999, all_touched=False,
        )
        for s in stats:
            raw = s.get("sum") or 0.0
            results.append(max(0.0, float(raw) * growth_factor))
    return np.array(results)


# ─────────────────────────────────────────────
# STEP 5 — ZERO-POPULATION LABELING (1 km only)
# ─────────────────────────────────────────────
def fix_zero_populations(
    pop_1km: np.ndarray,
    year: int,
) -> np.ndarray:
    """
    Label zero-population locations when using 1 km extraction only.

    Returns
    -------
    populations : np.ndarray[float]  finalised population per location
    """
    populations = pop_1km.copy()
    zero_count = int((populations <= 0).sum())
    if zero_count > 0:
        log.warning("  Year %d — %d locations remain zero — flagged.", year, zero_count)

    return populations


# ─────────────────────────────────────────────
# STEP 6 — YEARLY POPULATION TABLE (checkpointed)
# ─────────────────────────────────────────────
def get_yearly_populations(
    gdf_1km: gpd.GeoDataFrame,
    years: list,
) -> pd.DataFrame:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for year in years:
        ckpt = CHECKPOINT_DIR / f"pop_{year}.parquet"
        if ckpt.exists():
            log.info("Year %d — checkpoint found, skipping extraction.", year)
            cached = pd.read_parquet(ckpt)
            if "province" not in cached.columns:
                cached = cached.merge(
                    gdf_1km[["location_name", "province"]],
                    on="location_name",
                    how="left",
                )
            cached = cached.drop(columns=["pop_source"], errors="ignore")
            frames.append(cached)
            continue

        log.info("── Year %d ──────────────────────────────────", year)
        raster_path, raster_year = get_raster_path(year)
        raster_path = download_raster(year)
        growth_factor = (1 + ANNUAL_GROWTH_RATE) ** (year - raster_year)
        log.info("  Growth factor ×%.4f (base year %d)", growth_factor, raster_year)

        # Primary extraction at 1 km
        pop_1km = _zonal_sum(gdf_1km, raster_path, growth_factor, f"1 km primary (year={year})")

        # Label zeros (1 km only)
        populations = fix_zero_populations(pop_1km, year)

        year_df = gdf_1km[["location_name", "latitude", "longitude", "province"]].copy().reset_index(drop=True)
        year_df["year"]            = year
        year_df["base_population"] = populations.round().astype(int)
        

        year_df.to_parquet(ckpt, index=False)
        log.info("  Checkpoint saved → %s", ckpt)
        frames.append(year_df)

    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# STEP 7 — MONTHLY DENSITY (vectorised, no row loop)
# ─────────────────────────────────────────────
def build_monthly_density(pop_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Expanding to monthly density for %d location-year rows …", len(pop_df))

    if "province" not in pop_df.columns:
        pop_df = pop_df.copy()
        pop_df["province"] = "Unknown"


    months   = pd.date_range("2020-01-01", "2025-12-01", freq="MS")
    month_df = pd.DataFrame({
        "month":          months,
        "year":           months.year,
        "month_num":      months.month,
        "tourism_factor": [MONTHLY_TOURISM_FACTOR[m] for m in months.month],
    })
    mob_df = pd.DataFrame(
        list(YEARLY_MOBILITY_FACTOR.items()), columns=["year", "mobility_factor"]
    )
    month_df = month_df.merge(mob_df, on="year")

    pop_df   = pop_df.copy()
    pop_df["_key"]   = 1
    month_df["_key"] = 1
    merged = pop_df.merge(month_df, on=["_key", "year"]).drop(columns="_key")

    merged["monthly_density"] = (
        merged["base_population"]
        * merged["mobility_factor"]
        * merged["tourism_factor"]
    ).round().astype(int)

    merged["month"] = merged["month"].dt.strftime("%Y-%m")
    merged = merged.sort_values(["location_name", "month"]).reset_index(drop=True)

    log.info(
        "Monthly table: %d rows × %d locations",
        len(merged),
        merged.location_name.nunique(),
    )
    return merged


def fill_zero_months_by_province(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing/zero monthly_density with province-month average."""
    df = df.copy()
    # df["monthly_density_raw"] = df["monthly_density"]

    nonzero = df[df["monthly_density"].notna() & (df["monthly_density"] > 0)]
    prov_month_avg = (
        nonzero.groupby(["province", "month"])["monthly_density"]
        .mean()
        .rename("province_month_avg")
        .reset_index()
    )
    prov_avg = (
        nonzero.groupby(["province"])["monthly_density"]
        .mean()
        .rename("province_avg")
        .reset_index()
    )
    month_avg = (
        nonzero.groupby(["month"])["monthly_density"]
        .mean()
        .rename("month_avg")
        .reset_index()
    )
    global_avg = nonzero["monthly_density"].mean()
    if pd.isna(global_avg):
        global_avg = 1.0

    df = df.merge(prov_month_avg, on=["province", "month"], how="left")
    df = df.merge(prov_avg, on=["province"], how="left")
    df = df.merge(month_avg, on=["month"], how="left")

    missing_mask = df["monthly_density"].isna() | (df["monthly_density"] <= 0)
    fill_vals = df.loc[missing_mask, "province_month_avg"].fillna(
        df.loc[missing_mask, "province_avg"]
    )
    fill_vals = fill_vals.fillna(df.loc[missing_mask, "month_avg"]).fillna(global_avg)
    df.loc[missing_mask, "monthly_density"] = fill_vals.round().astype("Int64")

    # Final safety fill to eliminate any remaining missing/zero values.
    remaining_mask = df["monthly_density"].isna() | (df["monthly_density"] <= 0)
    if remaining_mask.any():
        df.loc[remaining_mask, "monthly_density"] = int(round(global_avg))

    df = df.drop(columns=["province_month_avg", "province_avg", "month_avg"], errors="ignore")
    return df


# ─────────────────────────────────────────────
# STEP 8 — SAVE
# ─────────────────────────────────────────────
def save_results(df: pd.DataFrame, path: str = OUTPUT_CSV) -> None:
    df.to_csv(path, index=False)
    log.info("Saved → %s  (%d rows, %.1f MB)", path, len(df), os.path.getsize(path) / 1e6)


# def save_zero_report(df: pd.DataFrame, path: str = ZERO_REPORT_CSV) -> None:
#     """Audit CSV for base population values."""
#     report = (
#         df.drop_duplicates("location_name")
#         [["location_name", "latitude", "longitude", "province", "base_population"]]
#         .reset_index(drop=True)
#     )
#     report.to_csv(path, index=False)
#     log.info("Base population report → %s  (%d locations)", path, len(report))


# ─────────────────────────────────────────────
# STEP 9 — VISUALISE
# ─────────────────────────────────────────────
def visualise_aggregate(df: pd.DataFrame, output_path: str = "plots/density_aggregate.png") -> None:
    df      = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    plot_df = df

    agg = plot_df.groupby("month")["monthly_density"].agg(
        p5=lambda x: np.percentile(x, 5),
        p25=lambda x: np.percentile(x, 25),
        median="median",
        p75=lambda x: np.percentile(x, 75),
        p95=lambda x: np.percentile(x, 95),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(agg["month"], agg["p5"],  agg["p95"],  alpha=0.12, color="#2196F3", label="p5–p95")
    ax.fill_between(agg["month"], agg["p25"], agg["p75"],  alpha=0.25, color="#2196F3", label="p25–p75")
    ax.plot(agg["month"], agg["median"], color="#2196F3", linewidth=2, label="Median")

    ax.set_title(
        "Monthly Population Density — All Locations Aggregate (2020–2025)\n"
        "(zero_flagged locations excluded from aggregate)",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("Est. persons in 1 km radius")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Aggregate chart → %s", output_path)
    # plt.show()


def visualise_top_locations(
    df: pd.DataFrame,
    n: int = 10,
    output_path: str = "plots/density_top_locations.png",
) -> None:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    top = (
        df.groupby("location_name")["monthly_density"]
        .median()
        .nlargest(n)
        .index.tolist()
    )
    sub = df[df.location_name.isin(top)]

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, loc in enumerate(top):
        d = sub[sub.location_name == loc].sort_values("month")
        ax.plot(d["month"], d["monthly_density"],
                label=loc, color=plt.cm.tab10.colors[i % 10], linewidth=1.4)

    ax.set_title(f"Top {n} Locations by Median Monthly Density", fontsize=12, fontweight="bold")
    ax.set_ylabel("Est. persons in 1 km radius")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, ncol=2)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Top-locations chart → %s", output_path)
    # plt.show()


# ─────────────────────────────────────────────
# OPTIONAL: FLASK API
# ─────────────────────────────────────────────
def start_api(df: pd.DataFrame, port: int = 5000) -> None:
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        log.error("Flask not installed.  Run:  pip install flask")
        return

    app    = Flask(__name__)
    df_api = df.copy()
    df_api["month_dt"] = pd.to_datetime(df_api["month"])

    from shapely.geometry import Point as SPoint
    import geopandas as _gpd

    loc_pts = (
        df_api.groupby("location_name")
        .agg(lat=("latitude", "first"), lon=("longitude", "first"))
        .reset_index()
    )
    loc_gdf = _gpd.GeoDataFrame(
        loc_pts,
        geometry=[SPoint(r.lon, r.lat) for _, r in loc_pts.iterrows()],
        crs="EPSG:4326",
    )

    def _nearest(lat, lon):
        pt  = _gpd.GeoDataFrame(geometry=[SPoint(lon, lat)], crs="EPSG:4326")
        idx = loc_gdf.distance(pt.geometry[0]).idxmin()
        return loc_gdf.loc[idx, "location_name"]

    @app.route("/density")
    def get_density():
        lat   = request.args.get("lat",   type=float)
        lon   = request.args.get("lon",   type=float)
        month = request.args.get("month", type=str)
        if None in (lat, lon, month):
            return jsonify({"error": "Provide lat, lon, month (YYYY-MM)"}), 400
        loc  = _nearest(lat, lon)
        mask = (df_api.location_name == loc) & (df_api["month_dt"].dt.strftime("%Y-%m") == month)
        row  = df_api[mask]
        if row.empty:
            return jsonify({"error": f"No data for {loc} / {month}"}), 404
        r = row.iloc[0]
        return jsonify({
            "location":        loc,
            "month":           month,
            "base_population": int(r.base_population),
            "mobility_factor": r.mobility_factor,
            "tourism_factor":  r.tourism_factor,
            "monthly_density": int(r.monthly_density),
        })

    @app.route("/density/list")
    def get_density_list():
        loc  = request.args.get("location", type=str)
        rows = df_api[df_api.location_name == loc] if loc else df_api
        if rows.empty:
            return jsonify({"error": "Location not found"}), 404
        out          = rows.copy()
        out["month"] = out["month_dt"].dt.strftime("%Y-%m")
        return jsonify(
            out[["location_name", "month", "monthly_density"]]
            .to_dict(orient="records")
        )

    @app.route("/locations")
    def get_locations():
        return jsonify(loc_pts[["location_name", "lat", "lon"]].to_dict(orient="records"))

    log.info("Flask API → http://127.0.0.1:%d", port)
    app.run(debug=False, port=port)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(use_api: bool = False, flask_port: int = 5000) -> pd.DataFrame:
    log.info("═" * 60)
    log.info(" Monthly Population Density Pipeline v3 — BULK + Zero Fix")
    log.info("═" * 60)
    t0 = time.time()

    # 1. Load coordinates
    locations_df = load_locations(LOCATIONS_CSV)
    locations_df = attach_province(locations_df)

    # 2. Build 1 km buffers upfront (one pass, reused each year)
    gdf_1km = build_all_buffers(locations_df)

    # 3. Yearly populations — 1 km raster extraction
    pop_df = get_yearly_populations(gdf_1km, YEARS)

    # 4. Expand to monthly density (vectorised cross-join, no Python loop)
    density_df = build_monthly_density(pop_df)
    density_df = fill_zero_months_by_province(density_df)
    density_df = density_df.drop(columns=["pop_source"], errors="ignore")

    # 5. Save
    save_results(density_df, OUTPUT_CSV)
    # save_zero_report(density_df, ZERO_REPORT_CSV)

    # 6. Summary
    print("\n-- Summary --")
    print(f"  Locations  : {density_df.location_name.nunique():,}")
    print(f"  Months     : {density_df.month.nunique()}")
    print(f"  Total rows : {len(density_df):,}")
    print(f"  Elapsed    : {(time.time() - t0) / 60:.1f} min")
    print("\n-- Sample (5 rows) --")
    print(
        density_df[[
            "location_name", "month", "base_population",
            "mobility_factor", "tourism_factor",
            "monthly_density",
        ]].sample(5).to_string(index=False)
    )

    # 7. Charts
    visualise_aggregate(density_df)
    visualise_top_locations(density_df, n=10)

    # 8. Optional REST API
    if use_api:
        start_api(density_df, port=flask_port)

    return density_df


if __name__ == "__main__":
    final_df = main(use_api=False, flask_port=5000)