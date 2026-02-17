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

# Load baseline success rates
df_bl = pd.read_csv(BASE_DIR / "params" / "all_a_pyirt_baseline.csv")
df_bl.rename(columns={df_bl.columns[0]: "task_id"}, inplace=True)

# Filter to SWE-bench tasks with valid difficulty
swe_df = df_irt[df_irt["task_id"].isin(swebench_task_ids) & df_irt["b"].notna()].copy()
swe_df = swe_df.merge(df_bl[["task_id", "success_rate"]], on="task_id", how="left")
difficulty = swe_df["b"].to_numpy()
diff_solved = swe_df.loc[swe_df["success_rate"] > 0, "b"].to_numpy()
diff_unsolved = swe_df.loc[swe_df["success_rate"] == 0, "b"].to_numpy()

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
ax.hist(diff_solved, bins=bins, color=PRIMARY_COLOR, alpha=0.5, edgecolor=EDGE_COLOR,
        linewidth=1.2, label=f"Solved by at least 1 model (n={len(diff_solved)})")
ax.hist(diff_unsolved, bins=bins, color="#d62828", alpha=0.6, edgecolor="#6a040f",
        linewidth=1.2, hatch="///", label=f"Never solved (n={len(diff_unsolved)})")

# Add vertical lines for percentiles
percentiles = [10, 30, 50, 70, 90]
colors = ["#2a9d8f", "#f77f00", "#e63946", "#7209b7", "#023e8a"]
linestyles = ["--", "-.", "--", "-.", "--"]
for pct, color, ls in zip(percentiles, colors, linestyles):
    val = np.percentile(difficulty, pct)
    ax.axvline(val, color=color, linestyle=ls, linewidth=1.8, label=f"P{pct} = {val:.2f}")

ax.set_xlabel("Task Difficulty (b)", fontsize=14, labelpad=8)
ax.set_ylabel("Number of Tasks", fontsize=14, labelpad=8)
ax.set_title("SWE-bench: Distribution of IRT Difficulty", fontsize=16, fontweight="bold", pad=12)
ax.legend(loc="upper right", frameon=True, fancybox=True, facecolor="white", fontsize=10)
ax.grid(True, which="major", linestyle="--", alpha=0.4)

fig.tight_layout()
output_path = BASE_DIR / "plots" / "swebench_difficulty_distribution.pdf"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved to {output_path}")

# --- CDF Plot ---
fig_cdf, ax_cdf = plt.subplots(figsize=(8, 5))

sorted_diff = np.sort(difficulty)
cdf = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)
ax_cdf.plot(sorted_diff, cdf, color=PRIMARY_COLOR, linewidth=2.5)

# Add horizontal + vertical lines for the same percentiles
for pct, color, ls in zip(percentiles, colors, linestyles):
    val = np.percentile(difficulty, pct)
    frac = pct / 100
    ax_cdf.hlines(frac, ax_cdf.get_xlim()[0] if ax_cdf.get_xlim()[0] < val else sorted_diff[0] - 1,
                   val, color=color, linestyle=ls, linewidth=1.2, alpha=0.6)
    ax_cdf.vlines(val, 0, frac, color=color, linestyle=ls, linewidth=1.2, alpha=0.6)
    ax_cdf.plot(val, frac, 'o', color=color, markersize=6, zorder=5, label=f"P{pct} = {val:.2f}")

ax_cdf.set_xlabel("Task Difficulty (b)", fontsize=14, labelpad=8)
ax_cdf.set_ylabel("Cumulative Proportion", fontsize=14, labelpad=8)
ax_cdf.set_title("SWE-bench: CDF of IRT Difficulty", fontsize=16, fontweight="bold", pad=12)
ax_cdf.legend(loc="lower right", frameon=True, fancybox=True, facecolor="white")
ax_cdf.grid(True, which="major", linestyle="--", alpha=0.4)
ax_cdf.set_ylim(0, 1.02)

fig_cdf.tight_layout()
output_path_cdf = BASE_DIR / "plots" / "swebench_difficulty_cdf.pdf"
fig_cdf.savefig(output_path_cdf, dpi=300, bbox_inches="tight")
print(f"CDF plot saved to {output_path_cdf}")

# --- Frontier model ability over time ---
from scipy import stats

df_abilities = pd.read_csv(BASE_DIR / "params" / "all_a_pyirt_abilities.csv")
# Keep only models with release dates
df_abilities = df_abilities.dropna(subset=["release_time"])
df_abilities["release_date"] = pd.to_datetime(df_abilities["release_time"])
df_abilities["release_months"] = (
    (df_abilities["release_date"] - pd.Timestamp("2019-01-01")).dt.days / 30.44
)

# Compute the frontier: for each date, the max ability seen so far
df_abilities = df_abilities.sort_values("release_date")
df_abilities["frontier_ability"] = df_abilities["ability"].cummax()

# Keep only frontier models (those that set a new max at their release)
frontier = df_abilities[df_abilities["ability"] == df_abilities["frontier_ability"]].copy()

# Linear regression on frontier
slope, intercept, r_value, p_value, std_err = stats.linregress(
    frontier["release_months"], frontier["ability"]
)
r2 = r_value ** 2

fig_ab, ax_ab = plt.subplots(figsize=(10, 6))

# Plot all models with release dates
ax_ab.scatter(df_abilities["release_date"], df_abilities["ability"],
              color="#adb5bd", s=30, alpha=0.5, zorder=2, label="All models")

# Highlight frontier models
ax_ab.scatter(frontier["release_date"], frontier["ability"],
              color="#e63946", s=70, edgecolor="#6a040f", linewidth=1.2, zorder=4,
              label=f"Frontier (n={len(frontier)})")

# Regression line
x_fit = np.linspace(df_abilities["release_months"].min(), df_abilities["release_months"].max(), 100)
y_fit = slope * x_fit + intercept
dates_fit = pd.Timestamp("2019-01-01") + pd.to_timedelta(x_fit * 30.44, unit="D")
ax_ab.plot(dates_fit, y_fit, color="#023e8a", linewidth=2.5, linestyle="--", zorder=3,
           label=f"Linear fit (R\u00b2={r2:.2f}, slope={slope:.3f}/mo)")

# Label frontier models
for _, row in frontier.iterrows():
    label = row["subject_id"]
    # Shorten labels for readability
    for prefix in ["claude-", "gpt-", "gpt", "o", "gemini-"]:
        if label.startswith(prefix):
            break
    ax_ab.annotate(label, (row["release_date"], row["ability"]),
                   textcoords="offset points", xytext=(8, 4), fontsize=7,
                   color="#1a1a1a", alpha=0.8)

ax_ab.set_xlabel("Release Date", fontsize=14, labelpad=8)
ax_ab.set_ylabel("Model Ability (\u03b8)", fontsize=14, labelpad=8)
ax_ab.set_title("Frontier Model Ability Over Time", fontsize=16, fontweight="bold", pad=12)
ax_ab.legend(loc="upper left", frameon=True, fancybox=True, facecolor="white", fontsize=11)
ax_ab.grid(True, which="major", linestyle="--", alpha=0.4)

fig_ab.tight_layout()
output_path_ab = BASE_DIR / "plots" / "frontier_ability_over_time.pdf"
fig_ab.savefig(output_path_ab, dpi=300, bbox_inches="tight")
print(f"Frontier ability plot saved to {output_path_ab}")
print(f"  Frontier regression: slope={slope:.4f}/month, intercept={intercept:.2f}, R\u00b2={r2:.3f}")
plt.show()
