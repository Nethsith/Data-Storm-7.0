from pathlib import Path
import shutil

import kagglehub


RAW_DIR = Path("raw data")
REQUIRED_FILES = [
	"distributor_seasonality_details.csv",
	"holiday_list.csv",
	"outlet_coordinates.csv",
	"outlet_master.csv",
	"transactions_history_final.csv",
]


def raw_data_present() -> bool:
	if not RAW_DIR.exists():
		return False
	return all((RAW_DIR / name).exists() for name in REQUIRED_FILES)


def main() -> None:
	if raw_data_present():
		print("Raw data already present. Skipping download.")
		return

	path = Path(kagglehub.competition_download("datastorm-7-0-rotaract"))
	print("Path to competition files:", path)

	RAW_DIR.mkdir(parents=True, exist_ok=True)

	for name in REQUIRED_FILES:
		src = path / name
		dest = RAW_DIR / name
		if not src.exists():
			continue
		if dest.exists():
			continue
		shutil.copy2(src, dest)

	if raw_data_present():
		print("Raw data copied to:", RAW_DIR)
	else:
		print("Raw data download completed, but some files are missing.")


if __name__ == "__main__":
	main()