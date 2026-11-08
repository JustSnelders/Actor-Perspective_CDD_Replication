import ast
import re
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# Determine filepaths
ROOT = Path(__file__).resolve().parent.parent
SUBGRAPHS = ROOT / "work" / "output_intermediate" / "BPIC17" / "subgraphs"
CPS = ROOT / "work" / "output_final" / "BPIC17" / "task_concept_drift_detection" / "process_level_drift" / \
    "cps_all_features.csv"

# Set parameters
START = pd.Timestamp("2016-01-01")  # first day of the log; window i starts at START + i days
COLORS = {"T02": "#8db84a", "T03": "#f4a259", "T10": "#a593cc"}
EXCLUDED_TASK = "T05"  # leftover cluster, excluded from the features

# Main function
def main(tasks):
    # Collect cached files
    pd.set_option("future.infer_string", False)
    files = sorted(SUBGRAPHS.glob("task_node_ws1_w*.pkl"), key=lambda f: int(re.search(r"_w(\d+)\.pkl", f.name).group(1)))
    
    # Read task instances and compute task shares
    rows = []
    for f in files:
        df = pd.read_pickle(f)
        if df.empty:
            rows.append({})
            continue
        df = df[df["task"] != EXCLUDED_TASK]
        rows.append((df["task"].value_counts() / len(df) * 100).to_dict())

    # Combine days into one table
    series = pd.DataFrame(rows).fillna(0)
    series.index = START + pd.to_timedelta(series.index, unit="D")

    # Read relevant change points from output
    cps = ast.literal_eval(pd.read_csv(CPS, index_col=0).loc["task_relative", "1_3"])
    cps = [cp for cp in cps if 150 < cp < 250]  # cp2 only

    # Plot the figure
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.5, 2.1), sharey=True)
    for ax, t in zip(axes, tasks):
        ax.plot(series.index, series.get(t, 0), lw=0.9, color=COLORS.get(t))
        for cp in cps:
            ax.axvline(START + pd.Timedelta(days=cp), color="black", ls="--", lw=0.9)
        ax.set_title(t, fontweight="bold", fontsize=10)
        ax.set_ylim(0, 21)
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_xlim(series.index[0], series.index[-1])
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(labelsize=8, labelleft=True)
    fig.tight_layout(w_pad=1.5)

    # Store the output figure
    (ROOT / "results").mkdir(exist_ok=True)
    out = ROOT / "results" / "fig_task_signals.pdf"
    fig.savefig(out)
    print(f"saved {out}; cp2 = {cps}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["T02", "T03", "T10"])
