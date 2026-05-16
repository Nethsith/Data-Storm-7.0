"""
Monthly Population Density Pipeline — Sri Lanka (BULK / 20 000 locations)
=========================================================================
Strategy change from single-location version
─────────────────────────────────────────────
  ❌ OLD: 120 000 WorldPop API calls  (~50 hours with polite pausing)
  ✅ NEW: Download one GeoTIFF raster per year (6 files, ~10 MB each)
          then extract all 20 000 values locally in <60 seconds total.

Architecture
────────────
  1. Load 20 000 coordinates from CSV
  2. For each year  → download WorldPop raster (Sri Lanka GeoTIFF, once)
  3. Zonal statistics on the raster for every 1 km buffer  (rasterstats)
  4. Apply monthly mobility × tourism factors
  5. Checkpoint after each year  → safe to resume if interrupted
  6. Merge all years → monthly_density.csv
  7. Visualise aggregate / per-location

Requirements
────────────
  pip install pandas geopandas shapely requests tqdm rasterstats rasterio matplotlib
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path
from typing import Optional
from shapely.geometry import Point
from tqdm import tqdm

# Optional but recommended for progress bars in notebooks
try:
    from tqdm.auto import tqdm
except ImportError:
    pass

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
# CONFIG
# ─────────────────────────────────────────────
LOCATIONS_CSV    = "processed/silver/outlet_coordinates_silver.csv"
OUTPUT_CSV       = "monthly_density.csv"
RASTER_CACHE_DIR = Path("worldpop_rasters")   # GeoTIFFs stored here
CHECKPOINT_DIR   = Path("checkpoints")        # one .parquet per year
BUFFER_METERS    = 1_000
YEARS            = list(range(2023, 2026))

# WorldPop GeoTIFF download template for Sri Lanka (ISO3 = LKA)
# Docs: https://hub.worldpop.org/geodata/listing?id=75
WORLDPOP_RASTER_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/"
    "{year}/LKA/lka_ppp_{year}_1km_Aggregated.tif"
)

# For 2021+ WorldPop uses Constrained data (only available to 2020 in free tier).
# Fallback for 2021-2025: project forward from 2020 raster with growth rate.
WORLDPOP_MAX_YEAR  = 2020
ANNUAL_GROWTH_RATE = 0.008   # 0.8 % / yr — Sri Lanka census-based

CHUNK_SIZE = 500   # locations processed per rasterstats batch (memory control)


# ─────────────────────────────────────────────
# MONTHLY / YEARLY FACTORS  (same as before)
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

    # Normalise column names (case-insensitive)
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
        raise ValueError(
            "CSV must contain latitude and longitude columns "
            f"(found: {list(df.columns)})"
        )
    if "location_name" not in df.columns:
        df["location_name"] = [f"loc_{i+1}" for i in range(len(df))]

    df = df[["location_name", "latitude", "longitude"]].copy()
    df = df.dropna(subset=["latitude", "longitude"])
    log.info("Loaded %d locations", len(df))
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 2 — BUILD 1 km BUFFER GeoDataFrame
# ─────────────────────────────────────────────
def build_buffers(df: pd.DataFrame) -> gpd.GeoDataFrame:
    log.info("Building 1 km buffers for %d locations …", len(df))
    pts = [Point(lon, lat) for lon, lat in zip(df.longitude, df.latitude)]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=pts, crs="EPSG:4326")
    gdf_m = gdf.to_crs("EPSG:5235")
    gdf_m["geometry"] = gdf_m.geometry.buffer(BUFFER_METERS)
    return gdf_m.to_crs("EPSG:4326")


# ─────────────────────────────────────────────
# STEP 3 — DOWNLOAD WORLDPOP RASTER (once/year)
# ─────────────────────────────────────────────
def get_raster_path(year: int) -> Path:
    RASTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # We always use the 2020 raster as the base; project forward if needed
    raster_year = min(year, WORLDPOP_MAX_YEAR)
    path = RASTER_CACHE_DIR / f"lka_pop_{raster_year}.tif"
    return path, raster_year


def download_raster(year: int) -> Path:
    path, raster_year = get_raster_path(year)
    if path.exists():
        log.info("  Raster already cached: %s", path)
        return path

    url = WORLDPOP_RASTER_URL.format(year=raster_year)
    log.info("  Downloading WorldPop raster for %d → %s", raster_year, path)
    log.info("  URL: %s", url)

    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=f"  lka_pop_{raster_year}.tif"
            ) as bar:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    bar.update(len(chunk))
        log.info("  Download complete: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    except requests.RequestException as exc:
        log.error("  Download failed: %s", exc)
        if path.exists():
            path.unlink()
        raise

    return path


# ─────────────────────────────────────────────
# STEP 4 — ZONAL STATISTICS (local, very fast)
# ─────────────────────────────────────────────
def extract_population_from_raster(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    year: int,
) -> pd.Series:
    """
    Use rasterstats to sum pixel values inside each 1 km buffer.
    Returns a Series of population counts aligned to gdf index.

    For years beyond 2020, applies compound growth to the 2020 raster.
    """
    try:
        from rasterstats import zonal_stats
    except ImportError:
        raise ImportError(
            "rasterstats is required for bulk extraction. "
            "Install it with:  pip install rasterstats"
        )

    _, raster_year = get_raster_path(year)
    growth_factor = (1 + ANNUAL_GROWTH_RATE) ** (year - raster_year)

    log.info(
        "  Extracting population (year=%d, raster_year=%d, growth=×%.4f) …",
        year, raster_year, growth_factor,
    )

    all_pops = []
    # Process in chunks to keep memory under control for 20 k rows
    indices = list(range(len(gdf)))
    for start in tqdm(range(0, len(gdf), CHUNK_SIZE), desc=f"  Zonal stats {year}"):
        chunk = gdf.iloc[start : start + CHUNK_SIZE]
        stats = zonal_stats(
            chunk,
            str(raster_path),
            stats=["sum"],
            nodata=-9999,
            all_touched=False,
        )
        for s in stats:
            raw = s.get("sum") or 0.0
            all_pops.append(max(0.0, float(raw) * growth_factor))

    return pd.Series(all_pops, index=gdf.index, name="base_population")


# ─────────────────────────────────────────────
# STEP 5 — YEARLY POPULATION TABLE (with checkpointing)
# ─────────────────────────────────────────────
def get_yearly_populations(gdf: gpd.GeoDataFrame, years: list) -> pd.DataFrame:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for year in years:
        ckpt = CHECKPOINT_DIR / f"pop_{year}.parquet"

        # Resume from checkpoint if already done
        if ckpt.exists():
            log.info("Year %d — loading from checkpoint %s", year, ckpt)
            frames.append(pd.read_parquet(ckpt))
            continue

        log.info("── Year %d ──", year)
        raster_path = download_raster(year)
        pop_series  = extract_population_from_raster(gdf, raster_path, year)

        year_df = gdf[["location_name", "latitude", "longitude"]].copy()
        year_df["year"]            = year
        year_df["base_population"] = pop_series.values.round().astype(int)
        year_df = year_df.reset_index(drop=True)

        year_df.to_parquet(ckpt, index=False)
        log.info("  Checkpoint saved → %s", ckpt)
        frames.append(year_df)

    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# STEP 6 — EXPAND TO MONTHLY DENSITY
# ─────────────────────────────────────────────
def build_monthly_density(pop_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Building monthly density for %d location-year rows …", len(pop_df))

    # Vectorised approach — no Python loop over rows
    months = pd.date_range("2020-01-01", "2025-12-01", freq="MS")
    month_df = pd.DataFrame({
        "month":          months,
        "year":           months.year,
        "month_num":      months.month,
        "tourism_factor": [MONTHLY_TOURISM_FACTOR[m] for m in months.month],
    })

    mob_df = pd.DataFrame(
        list(YEARLY_MOBILITY_FACTOR.items()),
        columns=["year", "mobility_factor"]
    )
    month_df = month_df.merge(mob_df, on="year")

    # Cross-join: every location × every month
    pop_df["_key"]    = 1
    month_df["_key"]  = 1
    merged = pop_df.merge(month_df, on=["_key", "year"]).drop(columns="_key")

    merged["monthly_density"] = (
        merged["base_population"]
        * merged["mobility_factor"]
        * merged["tourism_factor"]
    ).round().astype(int)

    merged["month"] = merged["month"].dt.strftime("%Y-%m")
    merged = merged.sort_values(["location_name", "month"]).reset_index(drop=True)

    log.info("Monthly density table: %d rows × %d locations",
             len(merged), merged.location_name.nunique())
    return merged


# ─────────────────────────────────────────────
# STEP 7 — SAVE
# ─────────────────────────────────────────────
def save_results(df: pd.DataFrame, path: str = OUTPUT_CSV) -> None:
    df.to_csv(path, index=False)
    log.info("Saved → %s  (%d rows, %.1f MB)",
             path, len(df), os.path.getsize(path) / 1e6)


# ─────────────────────────────────────────────
# STEP 8 — VISUALISE (aggregate view for 20 k locations)
# ─────────────────────────────────────────────
def visualise_aggregate(df: pd.DataFrame, output_path: str = "density_aggregate.png") -> None:
    """
    For 20 000 locations, plotting every line would be unreadable.
    Instead show: median, p25-p75 band, and p5-p95 band across all locations.
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])

    agg = df.groupby("month")["monthly_density"].agg(
        p5=lambda x: np.percentile(x, 5),
        p25=lambda x: np.percentile(x, 25),
        median="median",
        p75=lambda x: np.percentile(x, 75),
        p95=lambda x: np.percentile(x, 95),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(agg["month"], agg["p5"],  agg["p95"], alpha=0.12, color="#2196F3", label="p5–p95")
    ax.fill_between(agg["month"], agg["p25"], agg["p75"], alpha=0.25, color="#2196F3", label="p25–p75")
    ax.plot(agg["month"], agg["median"], color="#2196F3", linewidth=2, label="Median")

    ax.set_title("Monthly Population Density — All Locations Aggregate (2020–2025)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Est. persons in 1 km radius")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Aggregate chart saved → %s", output_path)
    plt.show()


def visualise_top_locations(
    df: pd.DataFrame,
    n: int = 10,
    output_path: str = "density_top_locations.png",
) -> None:
    """Plot the top N highest-density locations."""
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
    colors = plt.cm.tab10.colors
    for i, loc in enumerate(top):
        d = sub[sub.location_name == loc].sort_values("month")
        ax.plot(d["month"], d["monthly_density"],
                label=loc, color=colors[i % 10], linewidth=1.4)

    ax.set_title(f"Top {n} Locations by Median Monthly Density",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Est. persons in 1 km radius")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, ncol=2)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Top-locations chart saved → %s", output_path)
    plt.show()


# ─────────────────────────────────────────────
# OPTIONAL: FLASK API  (unchanged from v1)
# ─────────────────────────────────────────────
def start_api(df: pd.DataFrame, port: int = 5000) -> None:
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        log.error("Flask not installed.  Run:  pip install flask")
        return

    app = Flask(__name__)
    df_api = df.copy()
    df_api["month_dt"] = pd.to_datetime(df_api["month"])

    # Build a spatial index for fast nearest-location lookup
    from shapely.geometry import Point as SPoint
    import geopandas as _gpd
    loc_pts = (
        df_api.groupby("location_name")
        .agg(lat=("latitude","first"), lon=("longitude","first"))
        .reset_index()
    )
    loc_gdf = _gpd.GeoDataFrame(
        loc_pts,
        geometry=[SPoint(r.lon, r.lat) for _, r in loc_pts.iterrows()],
        crs="EPSG:4326",
    )

    def _nearest(lat, lon):
        pt = _gpd.GeoDataFrame(geometry=[SPoint(lon, lat)], crs="EPSG:4326")
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
            "location": loc, "month": month,
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
        out = rows.copy()
        out["month"] = out["month_dt"].dt.strftime("%Y-%m")
        return jsonify(out[["location_name","month","monthly_density"]].to_dict(orient="records"))

    @app.route("/locations")
    def get_locations():
        return jsonify(
            loc_pts[["location_name","lat","lon"]].to_dict(orient="records")
        )

    log.info("Flask API → http://127.0.0.1:%d", port)
    app.run(debug=False, port=port)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(use_api: bool = False, flask_port: int = 5000) -> pd.DataFrame:
    log.info("═" * 60)
    log.info(" Monthly Population Density Pipeline — BULK (20 000 locations)")
    log.info("═" * 60)

    t0 = time.time()

    # 1. Load
    locations_df = load_locations(LOCATIONS_CSV)

    # 2. Buffers
    gdf = build_buffers(locations_df)

    # 3. Yearly populations via raster (6 downloads + local extraction)
    pop_df = get_yearly_populations(gdf, YEARS)

    # 4. Monthly table (vectorised, no row loop)
    density_df = build_monthly_density(pop_df)

    # 5. Save
    save_results(density_df, OUTPUT_CSV)

    # 6. Summary stats
    print("\n── Summary ──")
    print(f"  Locations : {density_df.location_name.nunique():,}")
    print(f"  Months    : {density_df.month.nunique()}")
    print(f"  Total rows: {len(density_df):,}")
    print(f"  Elapsed   : {(time.time()-t0)/60:.1f} min")
    print("\n── Sample (5 rows) ──")
    print(density_df[["location_name","month","base_population",
                       "mobility_factor","tourism_factor","monthly_density"]]
          .sample(5).to_string(index=False))

    # 7. Visualise
    visualise_aggregate(density_df)
    visualise_top_locations(density_df, n=10)

    # 8. Optional API
    if use_api:
        start_api(density_df, port=flask_port)

    return density_df


if __name__ == "__main__":
    final_df = main(use_api=False, flask_port=5000)