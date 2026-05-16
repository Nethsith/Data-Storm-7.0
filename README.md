# Data-Storm-7.0

Data processing and population density tooling for the Data Storm 7.0 project.

## What is here

- Raw datasets in [raw data](raw%20data)
- Data processing notebooks in [notebook](notebook)
- A population density pipeline and API script in [scripts/1.API.py](scripts/1.API.py)
- Generated outputs in `processed/` and `summaries/` (gitignored)

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
./.venv/Scripts/activate
python -m pip install -r requirements.txt
```

## Run the population density pipeline

```bash
python scripts/1.API.py
```

Outputs:

- `monthly_density.csv`
- `density_chart.png`
- `locations.csv` (created with defaults if missing)

## Run notebooks

Open [notebook/data_process.ipynb](notebook/data_process.ipynb) in VS Code and run cells.
The notebook resolves file locations relative to the project root or the `notebook/` folder.

## Notes

- Raw data CSVs are tracked. Generated outputs in `processed/` and `summaries/` are ignored.
- The WorldPop API may return 400 errors for some years. The script falls back to
	synthetic values to keep the pipeline running.