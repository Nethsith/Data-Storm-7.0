from pathlib import Path
import pandas as pd


def get_project_dir() -> Path:
    base = Path.cwd()
    if (base / "raw data").exists():
        return base
    if (base.parent / "raw data").exists():
        return base.parent
    return base


PROJECT_DIR = get_project_dir()
RAW_DIR = PROJECT_DIR / "raw data"

holiday_data = pd.read_csv(RAW_DIR / "holiday_list.csv")
outlet_coordinates = pd.read_csv(RAW_DIR / "outlet_coordinates.csv")
outlet_master = pd.read_csv(RAW_DIR / "outlet_master.csv")
transaction_data = pd.read_csv(RAW_DIR / "transactions_history_final.csv")
distributor = pd.read_csv(RAW_DIR / "distributor_seasonality_details.csv")