# Data Storm v7.0 - Latent Outlet Potential Estimation

## 1. Project Overview

This repository contains our solution for **Data Storm v7.0 - Preliminary Round**.

The challenge objective is to estimate the **Maximum Monthly Purchase Potential** of each traditional trade outlet for **January 2026**.

In this problem, outlet potential is not directly available as a target variable. Historical sales only show what an outlet actually sold, not what it could have sold under ideal conditions. Therefore, our solution treats outlet potential as a **latent ceiling** and estimates it through a combination of:

- Data engineering and data-forensics checks
- Bronze → Silver → Gold pipeline architecture
- Transaction anomaly classification
- Missing value handling
- Holiday and seasonality feature engineering
- External population-density enrichment
- Outlet clustering
- Cluster-level peer benchmarking
- January 2026 adjustment factors

The final official output file is:

```text
submissions/teamname_predictions.csv
```

with the required columns:

```text
Outlet_ID
Maximum_Monthly_Liters
```

---

## 2. Quick Run Guide

Run the full pipeline from the project root:

```bash
python scripts/main.py
```

This will execute all scripts in the required order and generate the final prediction file.

Final output:

```text
submissions/teamname_predictions.csv
```

---

## 3. Environment Setup

### 3.1 Clone Repository

```bash
git clone https://github.com/Nethsith/Data-Storm-7.0.git
cd Data-Storm-7.0
```

### 3.2 Create Virtual Environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.3 Install Required Libraries

```bash
python -m pip install -r requirements.txt
```

The `requirements.txt` file contains all Python packages required for:

- Data loading and cleaning
- Feature engineering
- Missing value handling
- Geospatial / population-density enrichment
- Clustering
- Final prediction generation

Recommended `requirements.txt` content:

```text
numpy
pandas
requests
scikit-learn
xgboost
geopandas
shapely
matplotlib
tqdm
rasterstats
rasterio
scipy
pyarrow
```

Optional, only if the API server component is used:

```text
flask
```

---

## 4. Required Input Files

Place the competition datasets inside the `raw data/` folder.

Expected files:

```text
raw data/transactions_history_final.csv
raw data/outlet_master.csv
raw data/outlet_coordinates.csv
raw data/distributor_seasonality_details.csv
raw data/holiday_list.csv
```

The pipeline expects this folder structure before running.

---

## 5. Repository Structure

```text
Data-Storm-7.0/
│
├── README.md
├── requirements.txt
│
├── raw data/
│   ├── transactions_history_final.csv
│   ├── outlet_master.csv
│   ├── outlet_coordinates.csv
│   ├── distributor_seasonality_details.csv
│   └── holiday_list.csv
│
├── scripts/
│   ├── main.py
│   ├── 1.read_data.py
│   ├── 2.miss_val.py
│   ├── 3.silver_data.py
│   ├── 4.plot_seasonality.py
│   ├── 5.outlet_size_imputation.py
│   ├── 6.API.py
│   ├── 6.holiday.py
│   ├── 7.gold_data.py
│   └── 8.model.py
│
├── notebook/
│   └── data_process.ipynb
│
├── processed/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   ├── rejected/
│   └── anomalies/
│
├── summaries/
├── submissions/
├── worldpop_rasters/
├── provinces/
├── geonames/
└── checkpoints/
```

Note: If the final prediction script is named `8.prediction.py` instead of `8.model.py`, update the script name inside `scripts/main.py` accordingly.

---

## 6. Full Pipeline Running Order

The full pipeline is controlled by:

```text
scripts/main.py
```

It runs the following scripts in order:

```text
1. scripts/1.read_data.py
2. scripts/2.miss_val.py
3. scripts/3.silver_data.py
4. scripts/5.outlet_size_imputation.py
5. scripts/6.holiday.py
6. scripts/6.API.py
7. scripts/7.gold_data.py
8. scripts/8.model.py
```

Manual run order:

```bash
python scripts/1.read_data.py
python scripts/2.miss_val.py
python scripts/3.silver_data.py
python scripts/5.outlet_size_imputation.py
python scripts/6.holiday.py
python scripts/6.API.py
python scripts/7.gold_data.py
python scripts/8.model.py
```

Optional EDA visualization:

```bash
python scripts/4.plot_seasonality.py
```

---

## 7. End-to-End Workflow

The project follows a **Bronze → Silver → Gold → Prediction** pipeline.

```text
Raw Data
   ↓
Bronze Layer
   ↓
Silver Layer
   ↓
Missing Value Handling
   ↓
Holiday + Seasonality Feature Engineering
   ↓
Population Density Enrichment
   ↓
Gold Layer
   ↓
Clustering + Potential Estimation
   ↓
Final Prediction CSV
```

---

## 8. Layer Explanation

## 8.1 Bronze Layer

The Bronze layer preserves raw files as received.

Purpose:

- Keep original data unchanged
- Provide traceability
- Make it possible to compare raw data with cleaned data

Typical output folder:

```text
processed/bronze/
```

---

## 8.2 Silver Layer

The Silver layer applies data cleaning, anomaly detection, and data-forensics checks.

Main script:

```text
scripts/3.silver_data.py
```

Main actions:

- Standardizes column names
- Handles missing values and common null tokens
- Detects duplicate records
- Checks required fields
- Corrects recoverable latitude/longitude swaps
- Rejects invalid coordinate records
- Classifies transactions into business-meaningful groups
- Creates rejected records with documented reasons
- Creates anomaly files for non-demand artifacts
- Applies SKU-level price validation

Important Silver outputs:

```text
processed/silver/outlet_master_silver.csv
processed/silver/outlet_coordinates_silver.csv
processed/silver/transactions_sales_silver.csv
processed/silver/transactions_returns_silver.csv
processed/rejected/
processed/anomalies/
```

---

## 8.3 Gold Layer

The Gold layer integrates all cleaned and engineered datasets into model-ready files.

Main script:

```text
scripts/7.gold_data.py
```

Gold does not clean raw data again. It reads already prepared Silver outputs and combines them into final modelling tables.

Important Gold outputs:

```text
processed/gold/monthly_sales_gold.csv
processed/gold/outlet_transaction_features_gold.csv
processed/gold/outlet_features_gold.csv
processed/gold/outlet_modeling_table_gold.csv
```

The most important file for the model is:

```text
processed/gold/outlet_modeling_table_gold.csv
```

---

## 9. Script-by-Script Explanation

## 9.1 `scripts/1.read_data.py`

Purpose:

- Reads raw CSV files
- Checks available datasets
- Creates initial dataset summaries
- Reports row counts, column counts, data types, missing values, and sample values

Main outputs:

```text
summaries/dataset_summary.csv
summaries/column_summary.csv
```

---

## 9.2 `scripts/2.miss_val.py`

Purpose:

- Performs missing-value checks
- Helps understand which fields need cleaning, imputation, or further investigation
- Supports initial EDA and data-quality understanding

Main outputs:

```text
summaries/
```

---

## 9.3 `scripts/3.silver_data.py`

Purpose:

Creates the cleaned Silver layer.

Main cleaning and data-forensics work:

### Outlet master

- Cleans outlet categories
- Standardizes `Outlet_Size`
- Standardizes `Outlet_Type`
- Handles missing or invalid values
- Checks duplicate `Outlet_ID`

### Outlet coordinates

- Validates latitude and longitude ranges
- Corrects swapped latitude/longitude values when recoverable
- Rejects invalid coordinates that cannot be corrected

### Transactions

Transactions are classified into:

```text
Positive sales:
Volume_Liters > 0 and Total_Bill_Value > 0

Returns / reversals:
Volume_Liters < 0 and Total_Bill_Value < 0

Billing anomalies:
zero-volume or zero-bill records

Rejected records:
invalid sign mismatch or invalid required fields
```

### SKU price validation

The pipeline uses SKU-level price benchmarks instead of a single global price threshold because different SKUs have different normal price-per-liter ranges.

Main outputs:

```text
processed/silver/transactions_sales_silver.csv
processed/silver/transactions_returns_silver.csv
processed/anomalies/transactions_billing_anomalies.csv
processed/anomalies/transactions_price_outliers.csv
processed/rejected/
summaries/silver_quality_summary.csv
```

---

## 9.4 `scripts/5.outlet_size_imputation.py`

Purpose:

Handles missing `Outlet_Size` values before Gold integration.

Why this is needed:

`Outlet_Size` is an important outlet-level feature for clustering and potential estimation. Missing outlet size values are predicted using available outlet, coordinate, and transaction-derived features.

Method:

- Uses known `Outlet_Size` rows as training data
- Uses an XGBoost classifier for imputation
- Evaluates imputation with cross-validation
- Saves imputed outlet master data
- Adds confidence and review flags

Main outputs:

```text
processed/silver/outlet_master_imputed.csv
processed/silver/outlet_master_imputed_with_flags.csv
summaries/outlet_size_imputation_diagnostics.txt
```

Important note:

The Gold pipeline does not train this model again. It only reads the already-imputed outlet master file.

---

## 9.5 `scripts/6.holiday.py`

Purpose:

Fixes holiday overcounting and creates holiday/seasonality demand context features.

Problem:

The raw holiday file may contain multiple rows for the same holiday date because one holiday can be classified as:

```text
Public
Bank
Mercantile
Poya Day
```

If these rows are counted directly, one actual holiday date can be overcounted multiple times.

Solution:

- Parse holiday dates using pandas datetime
- Identify day of week
- Collapse holidays to unique date level
- Count holiday dates, not raw classification rows
- Create corrected monthly holiday features
- Combine holiday intensity with distributor seasonality
- Create distributor-month demand context score

Important outputs:

```text
processed/silver/holiday_list_silver.csv
processed/silver/holiday_date_level_silver.csv
processed/silver/holiday_monthly_features_silver.csv
processed/silver/distributor_month_context_score_silver.csv
summaries/holiday_monthly_feature_summary.csv
```

Important holiday features:

```text
holiday_date_count
holiday_classification_count
public_holiday_date_count
poya_day_date_count
festive_holiday_date_count
long_weekend_holiday_date_count
holiday_intensity_score
distributor_month_demand_context_score
```

---

## 9.6 `scripts/6.API.py`

Purpose:

Generates external population-density features using outlet coordinates.

This is used as an external catchment-demand signal.

Main idea:

- Uses outlet coordinates
- Generates monthly population-density related features
- Adds location context to outlet potential estimation

Main output:

```text
monthly_density.csv
```

Important features:

```text
base_population
monthly_density
mobility_factor
tourism_factor
province
```

These features are integrated into the Gold layer.

---

## 9.7 `scripts/7.gold_data.py`

Purpose:

Creates the final Gold model-ready data.

Gold integrates:

- imputed outlet master data
- outlet coordinates
- clean positive sales
- return/reversal features
- billing anomaly features
- holiday features
- distributor seasonality
- distributor-month demand context score
- population density features

Main sales features:

```text
avg_monthly_liters
median_monthly_liters
p75_monthly_liters
p90_monthly_liters
max_monthly_liters
sales_std
sales_cv
total_liters
```

Main January features:

```text
avg_january_liters
max_january_liters
avg_january_density
jan_2026_holiday_intensity_proxy
jan_2026_adjusted_holiday_score_proxy
jan_2026_seasonality_multiplier_used
```

Main seasonal spike features:

```text
april_spike_ratio
december_spike_ratio
seasonal_peak_liters
seasonal_spike_flag
```

Main return/anomaly features:

```text
return_record_count
return_liters_abs
return_ratio_liters
billing_anomaly_count
has_return_flag
has_billing_anomaly_flag
```

Main population-density features:

```text
avg_monthly_density
max_monthly_density
jan_2026_monthly_density
province
```

Main outputs:

```text
processed/gold/monthly_sales_gold.csv
processed/gold/outlet_transaction_features_gold.csv
processed/gold/outlet_features_gold.csv
processed/gold/outlet_modeling_table_gold.csv
summaries/gold_pipeline_summary.csv
summaries/gold_missing_value_report.csv
```

---

## 9.8 `scripts/8.model.py`

Purpose:

Estimates January 2026 maximum monthly potential.

Since there is no true target variable for potential, the model does not use a normal supervised regression approach. Instead, it uses:

```text
KMeans clustering
+ cluster-level top performer benchmarking
+ January 2026 adjustment factors
```

Main steps:

1. Read Gold model-ready data
2. Select clustering features
3. Scale numeric variables
4. One-hot encode categorical variables
5. Select best cluster count using silhouette score
6. Cluster similar outlets
7. Calculate January benchmark inside each cluster
8. Estimate January 2026 potential
9. Save final predictions

Clustering features include:

```text
Outlet_Size
Outlet_Type
Cooler_Count
Latitude
Longitude
Distributor_ID
sales behavior features
SKU variety features
return/anomaly features
holiday/context features
population-density features
January 2026 context features
```

Potential formula concept:

```text
January 2026 Potential
=
Cluster January Ceiling
× Seasonality Adjustment
× Holiday Context Adjustment
× Density Adjustment
× Recent Trend Adjustment
× Capacity Adjustment
```

Safety rules:

```text
Final Potential >= outlet's own historical January peak
Final Potential <= cluster January upper cap
```

Main outputs:

```text
processed/gold/outlet_clusters_gold.csv
processed/gold/monthly_sales_with_clusters_gold.csv
processed/gold/outlet_january_2026_potential_gold.csv
summaries/clustering_k_selection.csv
summaries/cluster_profiles.csv
summaries/cluster_potential_benchmarks.csv
summaries/potential_summary.csv
submissions/teamname_predictions.csv
```

---

## 10. Potential Calculation Logic

The final model estimates potential using peer comparison.

### Step 1: Cluster similar outlets

Outlets are grouped using KMeans clustering. This ensures that a small rural shop is not directly compared with a large urban grocery.

### Step 2: Calculate cluster-level January benchmark

For each cluster, the model calculates historical January sales percentiles:

```text
cluster_jan_p90
cluster_jan_p98
```

The P90 value is used as a realistic potential ceiling.

The P98 value is used as a safety cap to prevent unrealistic overestimation.

### Step 3: Apply January 2026 adjustments

The cluster ceiling is adjusted using:

- January 2026 distributor seasonality
- January holiday/context score
- population density
- recent trend
- outlet capacity signal

### Step 4: Apply business constraints

The final prediction is constrained:

```text
Final Potential >= outlet's own historical January peak
Final Potential <= cluster January upper cap
```

This makes sure that the model does not predict below an outlet’s proven clean January performance and does not overestimate beyond realistic cluster top-performer levels.

---

## 11. Output Files

## 11.1 Final Submission

```text
submissions/teamname_predictions.csv
```

Columns:

```text
Outlet_ID
Maximum_Monthly_Liters
```

## 11.2 Gold Outputs

```text
processed/gold/monthly_sales_gold.csv
processed/gold/outlet_modeling_table_gold.csv
processed/gold/outlet_january_2026_potential_gold.csv
```

## 11.3 Summary / Diagnostic Outputs

```text
summaries/main_pipeline_run_log.csv
summaries/final_pipeline_validation_report.csv
summaries/silver_quality_summary.csv
summaries/gold_pipeline_summary.csv
summaries/gold_missing_value_report.csv
summaries/clustering_k_selection.csv
summaries/cluster_profiles.csv
summaries/cluster_potential_benchmarks.csv
summaries/potential_summary.csv
```

---

## 12. Validation Checklist

After running:

```bash
python scripts/main.py
```

check:

```text
submissions/teamname_predictions.csv
summaries/final_pipeline_validation_report.csv
summaries/main_pipeline_run_log.csv
```

The final validation should confirm:

```text
submission file exists
columns are Outlet_ID and Maximum_Monthly_Liters
all outlets are predicted
no duplicate Outlet_ID
no missing Maximum_Monthly_Liters
Gold modeling table exists
potential output exists
```

---

## 13. Troubleshooting

### 13.1 `scripts/scripts/1.read_data.py` not found

This happens when `main.py` calculates the project root incorrectly.

If `main.py` is inside the `scripts/` folder, the root should be one level above `scripts`.

Correct root logic:

```python
CURRENT_FILE = Path(__file__).resolve()

if CURRENT_FILE.parent.name == "scripts":
    ROOT = CURRENT_FILE.parents[1]
else:
    ROOT = CURRENT_FILE.parent

SCRIPTS_DIR = ROOT / "scripts"
```

Run from project root:

```bash
python scripts/main.py
```

---

### 13.2 Missing `monthly_density.csv`

Run:

```bash
python scripts/6.API.py
```

Then run:

```bash
python scripts/7.gold_data.py
```

---

### 13.3 Missing Gold modeling table

Run:

```bash
python scripts/7.gold_data.py
```

Expected output:

```text
processed/gold/outlet_modeling_table_gold.csv
```

---

### 13.4 Missing final prediction file

Run:

```bash
python scripts/8.model.py
```

Expected output:

```text
submissions/teamname_predictions.csv
```

---

### 13.5 Kaggle row_id error

Kaggle may expect a different format such as:

```text
row_id
```

or fewer rows.

However, the official competition output for this project is:

```text
Outlet_ID
Maximum_Monthly_Liters
```

Therefore, the official final file should not be converted into Kaggle `row_id` format unless specifically required by the organizers.

---

## 14. GenAI Usage

Generative AI tools were used as engineering accelerators during the project.

AI support was used for:

- brainstorming the latent potential framework
- designing the Bronze → Silver → Gold pipeline
- identifying possible data-forensics checks
- improving holiday overcounting logic
- structuring clustering and peer benchmarking logic
- debugging pipeline execution errors
- improving documentation and README quality

Human validation was applied by:

- checking intermediate CSV outputs
- validating anomaly samples
- confirming corrected holiday counts
- checking transaction classification logic
- reviewing final prediction format
- reviewing cluster and potential summary reports

The final pipeline was reviewed and validated by the team. AI-generated suggestions were not used blindly.

---

## 15. Notes for Evaluators

The complete end-to-end run command is:

```bash
python scripts/main.py
```

Final prediction file:

```text
submissions/teamname_predictions.csv
```

Main model-ready input used by the final prediction step:

```text
processed/gold/outlet_modeling_table_gold.csv
```

Main final prediction detail file:

```text
processed/gold/outlet_january_2026_potential_gold.csv
```

The project is structured to show the full process from raw data inspection to data cleaning, Gold feature integration, clustering, and final January 2026 outlet potential estimation.