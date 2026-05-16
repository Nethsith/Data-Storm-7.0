"""
Monthly Population Density Pipeline — Sri Lanka
================================================
Architecture:
  Coordinates → 1km buffer → WorldPop base population →
  Monthly mobility/tourism factors → Monthly density → Store → Visualise

Requirements:
  pip install pandas geopandas shapely requests matplotlib flask
"""

import os
import json
import time
import logging
import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from shapely.geometry import Point, mapping
from typing import Optional

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
# CONFIG  (edit these freely)
# ─────────────────────────────────────────────
LOCATIONS_CSV   = "locations.csv"
OUTPUT_CSV      = "monthly_density.csv"
BUFFER_METERS   = 1_000          # 1 km radius
YEARS           = list(range(2020, 2026))   # 2020 – 2025 inclusive
WORLDPOP_URL    = "https://api.worldpop.org/v1/services/stats"
WORLDPOP_DATASET = "wpgppop"
API_PAUSE_SEC   = 1.5            # be polite to the API between calls


# ─────────────────────────────────────────────
# STEP 1 — LOAD / CREATE LOCATIONS
# ─────────────────────────────────────────────
DEFAULT_LOCATIONS = [
    {"location_name": "Colombo Fort",  "latitude": 6.9344, "longitude": 79.8428},
    {"location_name": "Negombo Beach", "latitude": 7.2083, "longitude": 79.8358},
    {"location_name": "Kandy Town",    "latitude": 7.2906, "longitude": 80.6337},
    {"location_name": "Galle Fort",    "latitude": 6.0329, "longitude": 80.2168},
]

def load_locations(path: str = LOCATIONS_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        log.info("Loaded %d locations from %s", len(df), path)
    else:
        log.warning("%s not found — using built-in defaults", path)
        df = pd.DataFrame(DEFAULT_LOCATIONS)
        df.to_csv(path, index=False)
        log.info("Saved default locations to %s", path)
    return df


# ─────────────────────────────────────────────
# STEP 2 — BUILD GeoDataFrame + 1 km BUFFERS
# ─────────────────────────────────────────────
def build_buffers(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame where each row's geometry is a 1 km circle."""
    geometry = [Point(lon, lat) for lon, lat in zip(df.longitude, df.latitude)]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")

    # Sri Lanka — use a metre-based CRS for accurate buffering
    gdf_m = gdf.to_crs("EPSG:5235")
    gdf_m["geometry"] = gdf_m.geometry.buffer(BUFFER_METERS)

    # Back to WGS-84 for the API
    gdf_wgs = gdf_m.to_crs("EPSG:4326")
    log.info("Created %d m buffers for %d locations", BUFFER_METERS, len(gdf_wgs))
    return gdf_wgs


def get_density_1km(lat: float, lon: float, year: int, month: Optional[int] = None) -> Optional[float]:
    """
    Return estimated population density (persons per km^2)
    for a 1 km radius buffer around a point.
    """
    point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    point_m = point.to_crs("EPSG:5235")
    buffer_m = point_m.buffer(BUFFER_METERS).iloc[0]

    area_km2 = buffer_m.area / 1_000_000
    if area_km2 <= 0:
        return None

    geojson_str = geom_to_geojson(gpd.GeoSeries([buffer_m], crs="EPSG:5235").to_crs("EPSG:4326").iloc[0])
    pop = fetch_worldpop_population(geojson_str, year)
    if pop is None:
        pop = _synthetic_population("unknown", year)

    if month is not None:
        mob = YEARLY_MOBILITY_FACTOR.get(year, 1.0)
        tour = MONTHLY_TOURISM_FACTOR.get(month, 1.0)
        pop = pop * mob * tour

    return float(pop) / area_km2


# ─────────────────────────────────────────────
# STEP 3 — WORLDPOP API
# ─────────────────────────────────────────────
def geom_to_geojson(geom) -> str:
    """Convert a Shapely geometry to a JSON string suitable for the API."""
    return json.dumps(mapping(geom))


def fetch_worldpop_population(geojson_str: str, year: int) -> Optional[float]:
    """
    Call the WorldPop REST API and return the estimated population.
    Returns None on error so the pipeline can continue gracefully.

    WorldPop docs: https://www.worldpop.org/sdi/introapi
    """
    params = {
        "dataset": WORLDPOP_DATASET,
        "year":    year,
        "geojson": geojson_str,
    }
    try:
        resp = requests.get(WORLDPOP_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # The API nests the value differently depending on version:
        # data["data"]["total_population"] or data["population"]
        pop = (
            data.get("data", {}).get("total_population")
            or data.get("population")
            or data.get("data", {}).get("pop")
        )
        return float(pop) if pop is not None else None

    except requests.RequestException as exc:
        log.error("WorldPop API error (year=%d): %s", year, exc)
        return None
    except (KeyError, ValueError, TypeError) as exc:
        log.error("WorldPop parse error (year=%d): %s | response: %s", year, exc, resp.text[:200])
        return None


def get_yearly_populations(gdf: gpd.GeoDataFrame, years: list) -> pd.DataFrame:
    """
    Fetch base population for every (location, year) combination.
    Falls back to synthetic estimates if the API is unreachable.
    """
    records = []
    for _, row in gdf.iterrows():
        geojson_str = geom_to_geojson(row.geometry)
        for year in years:
            log.info("  Fetching: %s — %d …", row.location_name, year)
            pop = fetch_worldpop_population(geojson_str, year)

            if pop is None:
                # Graceful fallback — synthetic estimate so the pipeline still runs
                pop = _synthetic_population(row.location_name, year)
                log.warning("  ↳ Using synthetic fallback: %.0f", pop)

            records.append({
                "location_name": row.location_name,
                "latitude":      row.latitude,
                "longitude":     row.longitude,
                "year":          year,
                "base_population": round(pop),
            })
            time.sleep(API_PAUSE_SEC)

    return pd.DataFrame(records)


def _synthetic_population(name: str, year: int) -> float:
    """
    Rough synthetic baseline so the pipeline can demonstrate results
    even when the WorldPop API is unavailable or returns nothing.
    Replace / remove once you have real API access.
    """
    base_map = {
        "Colombo Fort":  55_000,
        "Negombo Beach": 18_000,
        "Kandy Town":    22_000,
        "Galle Fort":    14_000,
    }
    base = base_map.get(name, 20_000)
    growth = 0.008           # ~0.8 % / year
    return base * ((1 + growth) ** (year - 2020))


# ─────────────────────────────────────────────
# STEP 4 — MONTHLY MOBILITY & TOURISM FACTORS
# ─────────────────────────────────────────────
# Source basis: SLTDA monthly arrivals pattern + Google Mobility trends.
# All values are multipliers around 1.0.  Update from real SLTDA data when
# available: https://www.sltda.gov.lk/en/statistical-data

MONTHLY_TOURISM_FACTOR = {
    1:  1.18,   # Jan  — high season (NE monsoon over, dry west coast)
    2:  1.15,   # Feb
    3:  1.08,   # Mar
    4:  0.82,   # Apr  — Avurudu holiday, slight dip in international arrivals
    5:  0.70,   # May  — SW monsoon begins
    6:  0.75,   # Jun
    7:  0.85,   # Jul  — Kandy Esala Perahera crowd build-up
    8:  0.90,   # Aug
    9:  0.80,   # Sep
    10: 0.88,   # Oct  — inter-monsoon
    11: 1.05,   # Nov  — NE monsoon, east coast rough but west coast picks up
    12: 1.20,   # Dec  — Christmas / New Year peak
}

# Simple mobility factor: COVID dip in 2020-21, recovery thereafter
YEARLY_MOBILITY_FACTOR = {
    2020: 0.60,
    2021: 0.72,
    2022: 0.85,
    2023: 0.95,
    2024: 1.00,
    2025: 1.02,
}


# ─────────────────────────────────────────────
# STEP 5 — BUILD MONTHLY DENSITY TABLE
# ─────────────────────────────────────────────
def build_monthly_density(pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand yearly populations to monthly records and apply factors.

    Final formula:
        monthly_density = base_population × mobility_factor × tourism_factor
    """
    rows = []
    months = pd.date_range("2020-01-01", "2025-12-01", freq="MS")

    for _, pop_row in pop_df.iterrows():
        year = pop_row.year
        mob  = YEARLY_MOBILITY_FACTOR.get(year, 1.0)

        year_months = [m for m in months if m.year == year]
        for month_ts in year_months:
            tour = MONTHLY_TOURISM_FACTOR[month_ts.month]
            density = pop_row.base_population * mob * tour

            rows.append({
                "location_name": pop_row.location_name,
                "latitude":      pop_row.latitude,
                "longitude":     pop_row.longitude,
                "month":         month_ts.strftime("%Y-%m"),
                "year":          year,
                "month_num":     month_ts.month,
                "base_population":   round(pop_row.base_population),
                "mobility_factor":   mob,
                "tourism_factor":    tour,
                "monthly_density":   round(density),
            })

    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["month"])
    return df.sort_values(["location_name", "month"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 6 — STORE RESULTS
# ─────────────────────────────────────────────
def save_results(df: pd.DataFrame, path: str = OUTPUT_CSV) -> None:
    export = df.copy()
    export["month"] = export["month"].dt.strftime("%Y-%m")
    export.to_csv(path, index=False)
    log.info("Results saved → %s  (%d rows)", path, len(df))


# ─────────────────────────────────────────────
# STEP 7 — VISUALISE
# ─────────────────────────────────────────────
def visualise(df: pd.DataFrame, output_path: str = "density_chart.png") -> None:
    locations = df["location_name"].unique()
    fig, axes = plt.subplots(
        len(locations), 1,
        figsize=(13, 3.5 * len(locations)),
        sharex=True,
    )
    if len(locations) == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

    for ax, loc, color in zip(axes, locations, colors):
        sub = df[df.location_name == loc].sort_values("month")
        ax.fill_between(sub["month"], sub["monthly_density"],
                        alpha=0.15, color=color)
        ax.plot(sub["month"], sub["monthly_density"],
                color=color, linewidth=1.8, label=loc)

        ax.set_title(loc, fontsize=11, fontweight="bold", pad=6)
        ax.set_ylabel("Est. density\n(persons in 1 km²)", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.suptitle(
        "Monthly Population Density — Sri Lanka Locations (2020–2025)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Chart saved → %s", output_path)
    plt.show()


# ─────────────────────────────────────────────
# OPTIONAL: STEP 8 — FLASK REST API
# ─────────────────────────────────────────────
def start_api(df: pd.DataFrame, port: int = 5000) -> None:
    """
    Spin up a lightweight Flask API so other services can query results.

    Usage:
        GET /density?lat=7.2&lon=79.8&month=2024-06
        GET /density/list?location=Negombo+Beach
        GET /locations
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        log.error("Flask not installed.  Run:  pip install flask")
        return

    app = Flask(__name__)
    # Store a datetime version for easy filtering
    df_api = df.copy()
    df_api["month_dt"] = pd.to_datetime(df_api["month"])

    def _nearest_location(lat: float, lon: float) -> str:
        """Return the location_name closest to the supplied coordinates."""
        from math import hypot
        best, best_dist = None, float("inf")
        for name, grp in df_api.groupby("location_name"):
            r = grp.iloc[0]
            d = hypot(r.latitude - lat, r.longitude - lon)
            if d < best_dist:
                best_dist, best = d, name
        return best

    @app.route("/density")
    def get_density():
        lat   = request.args.get("lat",   type=float)
        lon   = request.args.get("lon",   type=float)
        month = request.args.get("month", type=str)   # e.g. "2024-06"

        if lat is None or lon is None or month is None:
            return jsonify({"error": "Provide lat, lon, and month (YYYY-MM)"}), 400

        loc  = _nearest_location(lat, lon)
        mask = (df_api.location_name == loc) & (df_api["month_dt"].dt.strftime("%Y-%m") == month)
        row  = df_api[mask]

        if row.empty:
            return jsonify({"error": f"No data for {loc} / {month}"}), 404

        r = row.iloc[0]
        return jsonify({
            "location":         loc,
            "month":            month,
            "base_population":  int(r.base_population),
            "mobility_factor":  r.mobility_factor,
            "tourism_factor":   r.tourism_factor,
            "monthly_density":  int(r.monthly_density),
        })

    @app.route("/density/list")
    def get_density_list():
        loc  = request.args.get("location", type=str)
        rows = df_api[df_api.location_name == loc] if loc else df_api
        if rows.empty:
            return jsonify({"error": "Location not found"}), 404

        out = rows[["location_name", "month_dt", "monthly_density"]].copy()
        out["month"] = out["month_dt"].dt.strftime("%Y-%m")
        return jsonify(out[["location_name", "month", "monthly_density"]].to_dict(orient="records"))

    @app.route("/locations")
    def get_locations():
        locs = (
            df_api.groupby("location_name")
            .agg(latitude=("latitude", "first"), longitude=("longitude", "first"))
            .reset_index()
            .to_dict(orient="records")
        )
        return jsonify(locs)

    log.info("Starting Flask API on http://127.0.0.1:%d", port)
    log.info("  Example: http://127.0.0.1:%d/density?lat=7.21&lon=79.84&month=2024-06", port)
    app.run(debug=False, port=port)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(
    use_api: bool = False,    # set True to launch Flask after the pipeline
    flask_port: int = 5000,
) -> pd.DataFrame:

    log.info("═" * 55)
    log.info(" Monthly Population Density Pipeline — Sri Lanka")
    log.info("═" * 55)

    # 1. Load locations
    locations_df = load_locations(LOCATIONS_CSV)

    # 2. Build 1 km buffers
    gdf = build_buffers(locations_df)

    # 3. Fetch yearly population from WorldPop (with fallback)
    log.info("Calling WorldPop API for %d location × %d year combinations …",
             len(gdf), len(YEARS))
    pop_df = get_yearly_populations(gdf, YEARS)

    # 4. Build monthly density table
    log.info("Expanding to monthly records and applying factors …")
    density_df = build_monthly_density(pop_df)

    # 5. Save CSV
    save_results(density_df, OUTPUT_CSV)

    # 6. Print a sample
    print("\n── Sample output (first 12 rows) ──")
    print(
        density_df[["location_name", "month", "base_population",
                     "mobility_factor", "tourism_factor", "monthly_density"]]
        .head(12)
        .to_string(index=False)
    )

    # 7. Visualise
    visualise(density_df)

    # 8. Optional API server
    if use_api:
        start_api(density_df, port=flask_port)

    return density_df


if __name__ == "__main__":
    # ── Change use_api=True to also launch the Flask server ──
    final_df = main(use_api=False, flask_port=5000)