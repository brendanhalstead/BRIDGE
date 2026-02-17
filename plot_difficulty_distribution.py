"""Plot the distribution of IRT difficulty (b) for SWE-bench problems."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# --- Load data ---
# IRT parameters (task_id, a, b, human_minutes)
df_irt = pd.read_csv(BASE_DIR / "params" / "all_a_pyirt.csv")
df_irt.rename(columns={df_irt.columns[0]: "task_id"}, inplace=True)

# SWE-bench task IDs
swebench_task_ids = set()
with open(BASE_DIR / "data" / "swebench_normalized_results.jsonl") as f:
    for line in f:
        record = json.loads(line)
        swebench_task_ids.add(record["task_id"])

# Filter to SWE-bench tasks with valid difficulty
swe_df = df_irt[df_irt["task_id"].isin(swebench_task_ids) & df_irt["b"].notna()].copy()
difficulty = swe_df["b"].to_numpy()

print(f"SWE-bench tasks with IRT difficulty: {len(difficulty)}")
print(f"  min b = {difficulty.min():.2f}")
print(f"  max b = {difficulty.max():.2f}")
print(f"  mean b = {difficulty.mean():.2f}")
print(f"  median b = {np.median(difficulty):.2f}")

# --- Plot ---
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PRIMARY_COLOR = "#0077b6"
EDGE_COLOR = "#023e8a"

fig, ax = plt.subplots(figsize=(8, 5))

bins = np.linspace(difficulty.min() - 0.5, difficulty.max() + 0.5, 35)
ax.hist(difficulty, bins=bins, color=PRIMARY_COLOR, alpha=0.5, edgecolor=EDGE_COLOR, linewidth=1.2)

# Add vertical lines for mean and median
mean_val = difficulty.mean()
median_val = np.median(difficulty)
ax.axvline(mean_val, color="#e63946", linestyle="--", linewidth=1.8, label=f"Mean = {mean_val:.2f}")
ax.axvline(median_val, color="#f77f00", linestyle="-.", linewidth=1.8, label=f"Median = {median_val:.2f}")

ax.set_xlabel("Task Difficulty (b)", fontsize=14, labelpad=8)
ax.set_ylabel("Number of Tasks", fontsize=14, labelpad=8)
ax.set_title("SWE-bench: Distribution of IRT Difficulty", fontsize=16, fontweight="bold", pad=12)
ax.legend(loc="upper right", frameon=True, fancybox=True, facecolor="white")
ax.grid(True, which="major", linestyle="--", alpha=0.4)

fig.tight_layout()
output_path = BASE_DIR / "plots" / "swebench_difficulty_distribution.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved to {output_path}")
plt.show()
