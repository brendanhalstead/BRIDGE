"""Plot the distribution of IRT difficulty (b) for benchmark problems."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# --- Load data ---
df_irt = pd.read_csv(BASE_DIR / "params" / "all_a_pyirt.csv")
df_irt.rename(columns={df_irt.columns[0]: "task_id"}, inplace=True)

df_bl = pd.read_csv(BASE_DIR / "params" / "all_a_pyirt_baseline.csv")
df_bl.rename(columns={df_bl.columns[0]: "task_id"}, inplace=True)

# Collect task IDs per benchmark from normalized results
BENCHMARKS = {
    "SWE-bench": "swebench",
    "GDPval": "gdpval",
    "Cybench": "cybench",
}

benchmark_task_ids = {}
for display_name, file_key in BENCHMARKS.items():
    ids = set()
    with open(BASE_DIR / "data" / f"{file_key}_normalized_results.jsonl") as f:
        for line in f:
            ids.add(json.loads(line)["task_id"])
    benchmark_task_ids[display_name] = ids

# MLE-bench tasks use task_id::metric format in IRT params
mlebench_ids = set()
with open(BASE_DIR / "data" / "mlebench_normalized_results.jsonl") as f:
    for line in f:
        mlebench_ids.add(json.loads(line)["task_id"])
# Match IRT items whose prefix (before ::) is an MLE-bench task
mlebench_irt_ids = set()
for task_id in df_irt["task_id"]:
    base = task_id.split("::")[0]
    if base in mlebench_ids:
        mlebench_irt_ids.add(task_id)
benchmark_task_ids["MLE-bench"] = mlebench_irt_ids

# METR benchmarks (RE-Bench, HCAST, SWAA) — identified via task_source in all_runs.jsonl
metr_source_ids = {}
with open(BASE_DIR / "data" / "all_runs.jsonl") as f:
    for line in f:
        r = json.loads(line)
        src = r["task_source"]
        if src not in metr_source_ids:
            metr_source_ids[src] = set()
        metr_source_ids[src].add(r["task_id"])
for src, ids in metr_source_ids.items():
    benchmark_task_ids[src] = ids

# Build per-benchmark dataframes
benchmark_data = {}
for name, ids in benchmark_task_ids.items():
    bdf = df_irt[df_irt["task_id"].isin(ids) & df_irt["b"].notna()].copy()
    bdf = bdf.merge(df_bl[["task_id", "success_rate"]], on="task_id", how="left")
    n_unsolved = (bdf["success_rate"] == 0).sum()
    n_solved = len(bdf) - n_unsolved
    print(f"{name}: {len(bdf)} items in IRT ({n_solved} solved, {n_unsolved} never solved)")
    benchmark_data[name] = bdf

# Use SWE-bench as the primary for the histogram
swe_df = benchmark_data["SWE-bench"]
difficulty = swe_df["b"].to_numpy()
diff_solved = swe_df.loc[swe_df["success_rate"] > 0, "b"].to_numpy()
diff_unsolved = swe_df.loc[swe_df["success_rate"] == 0, "b"].to_numpy()

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

# --- CDF Plot function ---
def plot_cdf(difficulty_vals, title, file_key, n_unsolved=0):
    """Plot a CDF of IRT difficulty with percentile markers and methodology note."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mono = {"fontfamily": "monospace"}

    sorted_d = np.sort(difficulty_vals)
    cdf_vals = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.plot(sorted_d, cdf_vals, color=PRIMARY_COLOR, linewidth=2.5)

    for pct, color, ls in zip(percentiles, colors, linestyles):
        val = np.percentile(difficulty_vals, pct)
        frac = pct / 100
        ax.hlines(frac, ax.get_xlim()[0] if ax.get_xlim()[0] < val else sorted_d[0] - 1,
                  val, color=color, linestyle=ls, linewidth=1.2, alpha=0.6)
        ax.vlines(val, 0, frac, color=color, linestyle=ls, linewidth=1.2, alpha=0.6)
        ax.plot(val, frac, 'o', color=color, markersize=6, zorder=5, label=f"P{pct} = {val:.2f}")

    ax.set_xlabel("Task Difficulty (b)", fontsize=14, labelpad=8, **mono)
    ax.set_ylabel("Cumulative Proportion", fontsize=14, labelpad=8, **mono)
    ax.set_title(f"{title}: CDF of IRT Difficulty", fontsize=16, fontweight="bold", pad=12, **mono)
    ax.legend(loc="lower right", frameon=True, fancybox=True, facecolor="white",
              prop={"family": "monospace", "size": 12})
    ax.grid(True, which="major", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.02)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("monospace")

    n_total = len(difficulty_vals)
    unsolved_line = ""
    if n_unsolved > 0:
        unsolved_line = f"\n{n_unsolved} items never solved by any model; their b values are prior-regularized extrapolations."
    methodology = (
        f"Difficulty (b) estimated via 2-parameter logistic IRT (py-irt, hierarchical priors, 1000 epochs SVI).\n"
        f"{n_total} {title} items scored binary pass/fail across 176 model+scaffold submissions."
        f"{unsolved_line}\n"
        f"Data: Liu et al., \"BRIDGE: Predicting Human Task Completion Time From Model Performance\" (arXiv:2602.07267, 2026)."
    )
    fig.text(0.5, -0.02, methodology, ha="center", va="top", fontsize=7,
             fontfamily="monospace", color="#555555", style="italic")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    out = BASE_DIR / "plots" / f"{file_key}_difficulty_cdf.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"CDF plot saved to {out}")
    plt.close(fig)

# Generate CDF for each benchmark
FILE_KEYS = {
    "SWE-bench": "swebench", "GDPval": "gdpval", "MLE-bench": "mlebench",
    "Cybench": "cybench", "RE-Bench": "rebench", "HCAST": "hcast", "SWAA": "swaa",
}
for name, bdf in benchmark_data.items():
    vals = bdf["b"].to_numpy()
    n_unsolved = (bdf["success_rate"] == 0).sum()
    plot_cdf(vals, name, FILE_KEYS[name], n_unsolved=n_unsolved)

# --- Violin comparison across all benchmarks ---
bench_order = ["SWE-bench", "GDPval", "MLE-bench", "HCAST", "SWAA", "Cybench", "RE-Bench"]
violin_data = [benchmark_data[name]["b"].to_numpy() for name in bench_order]
violin_counts = [len(d) for d in violin_data]
violin_labels = [f"{name}\n(n={n})" for name, n in zip(bench_order, violin_counts)]

fig_v, ax_v = plt.subplots(figsize=(12, 6))
mono_v = {"fontfamily": "monospace"}

parts_v = ax_v.violinplot(violin_data, positions=range(len(bench_order)),
                          showmedians=True, showextrema=False, vert=True)
for body in parts_v["bodies"]:
    body.set_facecolor(PRIMARY_COLOR)
    body.set_alpha(0.5)
    body.set_edgecolor(EDGE_COLOR)
    body.set_linewidth(1.2)
parts_v["cmedians"].set_color("#e63946")
parts_v["cmedians"].set_linewidth(2)

ax_v.set_xticks(range(len(bench_order)))
ax_v.set_xticklabels(violin_labels, fontsize=10, fontfamily="monospace")
ax_v.set_ylabel("Task Difficulty (b)", fontsize=14, labelpad=8, **mono_v)
ax_v.set_title("Task Difficulty Distribution by Benchmark", fontsize=16, fontweight="bold", pad=12, **mono_v)
ax_v.grid(True, which="major", linestyle="--", alpha=0.4, axis="y")
for label in ax_v.get_yticklabels():
    label.set_fontfamily("monospace")

# Annotate unsolved counts
for i, name in enumerate(bench_order):
    bdf = benchmark_data[name]
    n_unsolved = (bdf["success_rate"] == 0).sum()
    if n_unsolved > 0:
        ax_v.text(i, bdf["b"].max() + 0.3, f"{n_unsolved} unsolved",
                  ha="center", va="bottom", fontsize=7, fontfamily="monospace",
                  color="#d62828", style="italic")

fig_v.text(0.5, -0.02,
    "Difficulty (b) estimated via 2-parameter logistic IRT (py-irt, hierarchical priors, 1000 epochs SVI).\n"
    "Red line = median. Tasks never solved by any model have prior-regularized b estimates.\n"
    'Data: Liu et al., "BRIDGE: Predicting Human Task Completion Time From Model Performance" (arXiv:2602.07267, 2026).',
    ha="center", va="top", fontsize=7, fontfamily="monospace", color="#555555", style="italic")

fig_v.tight_layout()
fig_v.subplots_adjust(bottom=0.16)
output_path_v = BASE_DIR / "plots" / "benchmark_difficulty_violins.pdf"
fig_v.savefig(output_path_v, dpi=300, bbox_inches="tight")
print(f"Violin comparison plot saved to {output_path_v}")
plt.close(fig_v)

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

def plot_frontier(df_abilities, frontier, slope, intercept, r2,
                  violin_difficulty, bench_names, output_file, title_suffix=""):
    """Plot frontier ability over time with a difficulty violin on the right."""
    fig, (ax_main, ax_vln) = plt.subplots(
        1, 2, figsize=(12, 6), width_ratios=[5, 1], sharey=True,
        gridspec_kw={"wspace": 0.02},
    )

    ax_main.scatter(df_abilities["release_date"], df_abilities["ability"],
                    color="#adb5bd", s=30, alpha=0.5, zorder=2, label="All models")
    ax_main.scatter(frontier["release_date"], frontier["ability"],
                    color="#e63946", s=70, edgecolor="#6a040f", linewidth=1.2, zorder=4,
                    label=f"Frontier (n={len(frontier)})")

    x_fit = np.linspace(df_abilities["release_months"].min(), df_abilities["release_months"].max(), 100)
    y_fit = slope * x_fit + intercept
    dates_fit = pd.Timestamp("2019-01-01") + pd.to_timedelta(x_fit * 30.44, unit="D")
    ax_main.plot(dates_fit, y_fit, color="#023e8a", linewidth=2.5, linestyle="--", zorder=3,
                 label=f"Linear fit (R\u00b2={r2:.2f}, slope={slope:.3f}/mo)")

    for _, row in frontier.iterrows():
        lbl = row["subject_id"]
        for prefix in ["claude-", "gpt-", "gpt", "o", "gemini-"]:
            if lbl.startswith(prefix):
                break
        ax_main.annotate(lbl, (row["release_date"], row["ability"]),
                         textcoords="offset points", xytext=(8, 4), fontsize=7,
                         fontfamily="monospace", color="#1a1a1a", alpha=0.8)

    mono = {"fontfamily": "monospace"}
    title = "Frontier Model Ability Over Time"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax_main.set_xlabel("Release Date", fontsize=14, labelpad=8, **mono)
    ax_main.set_ylabel("Model Ability (\u03b8) / Task Difficulty (b)", fontsize=14, labelpad=8, **mono)
    ax_main.set_title(title, fontsize=16, fontweight="bold", pad=12, **mono)
    ax_main.legend(loc="upper left", frameon=True, fancybox=True, facecolor="white",
                   prop={"family": "monospace", "size": 11})
    ax_main.grid(True, which="major", linestyle="--", alpha=0.4)
    for lbl in ax_main.get_xticklabels() + ax_main.get_yticklabels():
        lbl.set_fontfamily("monospace")

    # Violin
    parts = ax_vln.violinplot(violin_difficulty, positions=[0], showmedians=True, showextrema=False, vert=True)
    for body in parts["bodies"]:
        body.set_facecolor(PRIMARY_COLOR)
        body.set_alpha(0.5)
        body.set_edgecolor(EDGE_COLOR)
        body.set_linewidth(1.2)
    parts["cmedians"].set_color("#e63946")
    parts["cmedians"].set_linewidth(2)
    ax_vln.set_xticks([0])
    ax_vln.set_xticklabels(["Task\nDifficulty\n(b)"], fontsize=9, fontfamily="monospace")
    ax_vln.tick_params(axis="y", labelleft=False)
    ax_vln.grid(True, which="major", linestyle="--", alpha=0.4)
    ax_vln.spines["top"].set_visible(False)
    ax_vln.spines["right"].set_visible(False)

    bench_list = ", ".join(bench_names)
    fig.text(0.5, -0.02,
        f"Ability (\u03b8) jointly estimated via 2PL IRT. Violin shows difficulty from: {bench_list}.\n"
        'Adopted from the data in Liu et al., "BRIDGE: Predicting Human Task Completion Time '
        'From Model Performance" (arXiv:2602.07267, 2026).',
        ha="center", va="top", fontsize=7, fontfamily="monospace", color="#555555", style="italic")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    out = BASE_DIR / "plots" / output_file
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Frontier plot saved to {out}")
    print(f"  Regression: slope={slope:.4f}/mo, intercept={intercept:.2f}, R\u00b2={r2:.3f}")
    plt.close(fig)


# All benchmarks
all_difficulty = df_irt["b"].dropna().to_numpy()
all_bench_names = list(FILE_KEYS.keys())
plot_frontier(df_abilities, frontier, slope, intercept, r2,
              all_difficulty, all_bench_names, "frontier_ability_over_time.pdf")

# Excluding HCAST, RE-Bench, SWAA — use REFIT IRT parameters
held_out = {"HCAST", "RE-Bench", "SWAA"}
kept_names = [n for n in FILE_KEYS if n not in held_out]

# Load refit item parameters and abilities (fitted on 4-benchmark subset)
df_irt_no_metr = pd.read_csv(BASE_DIR / "params" / "no_metr_pyirt.csv")
df_irt_no_metr.rename(columns={df_irt_no_metr.columns[0]: "task_id"}, inplace=True)
kept_difficulty = df_irt_no_metr["b"].dropna().to_numpy()

df_abilities_no_metr = pd.read_csv(BASE_DIR / "params" / "no_metr_pyirt_abilities.csv")
df_abilities_no_metr = df_abilities_no_metr.dropna(subset=["release_time"])
df_abilities_no_metr["release_date"] = pd.to_datetime(df_abilities_no_metr["release_time"])
df_abilities_no_metr["release_months"] = (
    (df_abilities_no_metr["release_date"] - pd.Timestamp("2019-01-01")).dt.days / 30.44
)
df_abilities_no_metr = df_abilities_no_metr.sort_values("release_date")
df_abilities_no_metr["frontier_ability"] = df_abilities_no_metr["ability"].cummax()
frontier_no_metr = df_abilities_no_metr[
    df_abilities_no_metr["ability"] == df_abilities_no_metr["frontier_ability"]
].copy()

slope_nm, intercept_nm, r_value_nm, p_value_nm, std_err_nm = stats.linregress(
    frontier_no_metr["release_months"], frontier_no_metr["ability"]
)
r2_nm = r_value_nm ** 2

print(f"\nHeld-out version (refit): {len(kept_difficulty)} tasks from {kept_names}")
print(f"  Refit abilities: {len(df_abilities_no_metr)} models with release dates")
plot_frontier(df_abilities_no_metr, frontier_no_metr, slope_nm, intercept_nm, r2_nm,
              kept_difficulty, kept_names, "frontier_ability_over_time_no_metr.pdf",
              title_suffix="(excluding HCAST, RE-Bench, SWAA)")
