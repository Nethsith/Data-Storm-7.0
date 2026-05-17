from pathlib import Path
import subprocess
import sys
import time
import pandas as pd


ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"

SUMMARY_DIR = ROOT / "summaries"
SUBMISSION_DIR = ROOT / "submissions"
GOLD_DIR = ROOT / "processed" / "gold"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PIPELINE ORDER
# ============================================================
# Important:
# - 7.gold_data.py and 8.prediction.py must exist before running main.py.
# - 4.plot_seasonality.py is EDA/visualization, so it is not mandatory for final prediction.
#   Keep it in repo as evidence, but do not block final model run with plotting.

PIPELINE_STEPS = [
    {
        "name": "Step 1 - Raw data inspection and initial summaries",
        "script": "1.read_data.py",
        "required": True,
    },
    {
        "name": "Step 2 - Missing value and data quality overview",
        "script": "2.miss_val.py",
        "required": True,
    },
    {
        "name": "Step 3 - Silver data cleaning and anomaly classification",
        "script": "3.silver_data.py",
        "required": True,
    },
    {
        "name": "Step 4 - Outlet size missing value imputation",
        "script": "5.outlet_size_imputation.py",
        "required": True,
    },
    {
        "name": "Step 5 - Holiday correction and demand context score",
        "script": "6.holiday.py",
        "required": True,
    },
    {
        "name": "Step 6 - Population density feature generation",
        "script": "6.API.py",
        "required": True,
    },
    {
        "name": "Step 7 - Gold feature integration",
        "script": "7.gold_data.py",
        "required": True,
    },
    {
        "name": "Step 8 - Clustering and January 2026 potential prediction",
        "script": "8.prediction.py",
        "required": True,
    },
]


EXPECTED_OUTPUTS = [
    GOLD_DIR / "monthly_sales_gold.csv",
    GOLD_DIR / "outlet_modeling_table_gold.csv",
    GOLD_DIR / "outlet_january_2026_potential_gold.csv",
    SUBMISSION_DIR / "teamname_predictions.csv",
]


def run_script(step: dict) -> dict:
    script_path = SCRIPTS_DIR / step["script"]

    result = {
        "step": step["name"],
        "script": str(script_path),
        "status": "not_started",
        "runtime_seconds": None,
        "error": "",
    }

    if not script_path.exists():
        message = f"Missing script: {script_path}"

        if step["required"]:
            result["status"] = "failed"
            result["error"] = message
            raise FileNotFoundError(message)

        result["status"] = "skipped"
        result["error"] = message
        print(f"[SKIPPED] {message}")
        return result

    print("\n" + "=" * 80)
    print(step["name"])
    print(f"Running: {script_path}")
    print("=" * 80)

    start = time.time()

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    runtime = round(time.time() - start, 2)

    result["runtime_seconds"] = runtime

    if completed.stdout:
        print(completed.stdout)

    if completed.stderr:
        print(completed.stderr)

    if completed.returncode != 0:
        result["status"] = "failed"
        result["error"] = completed.stderr[-2000:] if completed.stderr else "Unknown error"
        raise RuntimeError(f"{step['script']} failed. Check terminal output.")

    result["status"] = "success"
    print(f"[OK] {step['script']} completed in {runtime} seconds.")

    return result


def validate_final_outputs() -> pd.DataFrame:
    validation = {
        "submission_exists": False,
        "submission_rows": 0,
        "submission_columns": "",
        "duplicate_outlet_ids": -1,
        "missing_predictions": -1,
        "official_format_valid": False,
        "gold_modeling_table_exists": False,
        "potential_gold_exists": False,
    }

    submission_path = SUBMISSION_DIR / "teamname_predictions.csv"

    validation["gold_modeling_table_exists"] = (GOLD_DIR / "outlet_modeling_table_gold.csv").exists()
    validation["potential_gold_exists"] = (GOLD_DIR / "outlet_january_2026_potential_gold.csv").exists()
    validation["submission_exists"] = submission_path.exists()

    if submission_path.exists():
        submission = pd.read_csv(submission_path)

        validation["submission_rows"] = len(submission)
        validation["submission_columns"] = ", ".join(submission.columns.tolist())

        if "Outlet_ID" in submission.columns:
            validation["duplicate_outlet_ids"] = int(submission["Outlet_ID"].duplicated().sum())

        if "Maximum_Monthly_Liters" in submission.columns:
            validation["missing_predictions"] = int(submission["Maximum_Monthly_Liters"].isna().sum())

        validation["official_format_valid"] = (
            list(submission.columns) == ["Outlet_ID", "Maximum_Monthly_Liters"]
            and len(submission) == 20000
            and validation["duplicate_outlet_ids"] == 0
            and validation["missing_predictions"] == 0
        )

    validation_df = pd.DataFrame([validation])
    validation_df.to_csv(SUMMARY_DIR / "final_pipeline_validation_report.csv", index=False)

    return validation_df


def main():
    print("Data Storm v7.0 - End-to-End Pipeline")
    print(f"Project root: {ROOT}")

    run_logs = []

    try:
        for step in PIPELINE_STEPS:
            run_logs.append(run_script(step))

        print("\nValidating final outputs...")
        validation_df = validate_final_outputs()

        run_log_df = pd.DataFrame(run_logs)
        run_log_df.to_csv(SUMMARY_DIR / "main_pipeline_run_log.csv", index=False)

        print("\nFinal validation report:")
        print(validation_df.to_string(index=False))

        print("\nPipeline completed successfully.")
        print("Final official output:")
        print(SUBMISSION_DIR / "teamname_predictions.csv")

    except Exception as e:
        print("\nPipeline failed.")
        print(str(e))

        if run_logs:
            pd.DataFrame(run_logs).to_csv(SUMMARY_DIR / "main_pipeline_run_log.csv", index=False)

        raise


if __name__ == "__main__":
    main()