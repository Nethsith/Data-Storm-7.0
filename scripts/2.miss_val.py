from pathlib import Path
import json
import pandas as pd


def get_project_dir() -> Path:
    base = Path.cwd()
    if (base / "raw data").exists():
        return base
    if (base.parent / "raw data").exists():
        return base.parent
    return base


PROJECT_DIR = get_project_dir()

# Summaries output folder
outdir = PROJECT_DIR / "summaries"
outdir.mkdir(parents=True, exist_ok=True)

p = PROJECT_DIR / "raw data"
csvs = sorted(p.glob("*.csv"))

print("Found CSV files:", [str(x) for x in csvs])

dataset_rows = []
column_rows = []

for f in csvs:
    size_bytes = f.stat().st_size
    size_mb = size_bytes / 1024 / 1024
    fname = str(f)
    try:
        df = pd.read_csv(f, low_memory=False)
    except Exception:
        df = pd.read_csv(f, engine="python", low_memory=False)
    rows, cols = df.shape
    dataset_rows.append({"file": fname, "rows": int(rows), "cols": int(cols), "size_mb": round(size_mb, 2)})
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        unique_ratio = round(unique / rows, 4) if rows > 0 else None
        missing_ratio = round(missing / rows, 4) if rows > 0 else None
        dtype = str(s.dtype)
        try:
            sample_vals = s.dropna().unique()[:5].tolist()
        except Exception:
            sample_vals = []
        column_rows.append({
            "file": fname,
            "column": col,
            "dtype": dtype,
            "unique_count": unique,
            "unique_ratio": unique_ratio,
            "missing_count": missing,
            "missing_ratio": missing_ratio,
            "sample_values": json.dumps(sample_vals)
        })

# Write summaries into summaries/ folder
pd.DataFrame(dataset_rows).to_csv(outdir / "dataset_summary.csv", index=False)
pd.DataFrame(column_rows).to_csv(outdir / "column_summary.csv", index=False)

print("Wrote", outdir / "dataset_summary.csv", "and", outdir / "column_summary.csv")
