import argparse
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

MAPPING = {
    "Favorable": 1,
    "Moderate": 0,
    "Un-Favorable": -1,
}

def plot_all_seasonality(csv_path, out_dir="plots"):
    df = pd.read_csv(csv_path)
    
    if df.empty:
        raise SystemExit(f"No data found in {csv_path}")

    # Prepare data types and map seasonality values
    df["Month"] = df["Month"].astype(int)
    df["Year"] = df["Year"].astype(int)
    df["Seasonality_Num"] = df["Seasonality_Index"].map(MAPPING)

    # Get all unique distributors
    distributors = df["Distributor_ID"].unique()
    num_distributors = len(distributors)
    
    if num_distributors == 0:
        raise SystemExit("No distributors found in the dataset.")

    # Define a grid layout: 2 columns, dynamic number of rows
    cols = 2
    rows = math.ceil(num_distributors / cols)

    os.makedirs(out_dir, exist_ok=True)
    
    # Create the figure with a dynamically calculated size (width=15, height scales with rows)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    # Loop through each distributor and plot on a corresponding subplot
    for i, distributor_id in enumerate(distributors):
        ax = axes_flat[i]
        dist_df = df[df["Distributor_ID"] == distributor_id]

        # Group by Year and plot the lines
        for year, grp in sorted(dist_df.groupby("Year")):
            grp = grp.sort_values("Month")
            ax.plot(grp["Month"], grp["Seasonality_Num"], marker="o", label=str(year))

        # Format the specific subplot
        ax.set_xticks(list(range(1, 13)))
        ax.set_xticklabels([
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ])
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["Un-Favorable", "Moderate", "Favorable"])
        ax.set_xlabel("Month")
        ax.set_ylabel("Seasonality Index")
        ax.set_title(f"Seasonality Index by Month — {distributor_id}")
        ax.legend(title="Year")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    # If there is an odd number of distributors, hide the extra empty subplots
    for j in range(num_distributors, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # Save the combined figure
    out_path = os.path.join(out_dir, "seasonality_all_distributors.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot containing {num_distributors} graphs to: {out_path}")
    
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Plot Seasonality_Index vs Month for all distributors in one PNG")
    parser.add_argument("--csv", default=os.path.join("raw data", "distributor_seasonality_details.csv"))
    parser.add_argument("--out", default="plots")
    args = parser.parse_args()

    plot_all_seasonality(args.csv, args.out)

if __name__ == "__main__":
    main()