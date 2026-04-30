import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

RESULTS_PATH = "models/training/isolation_forest_results.csv"
TEST_DATA_PATH = "feature_engineering/X_test_isolation.csv"
OUTPUT_DIR = "visuals"

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = pd.read_csv(RESULTS_PATH)
X_test = pd.read_csv(TEST_DATA_PATH).reset_index(drop=True)

test_results = results[results.split == "test"].reset_index(drop=True)
X_test["anomaly_score"] = test_results["anomaly_score"]
X_test["flag"] = test_results["flag"]

normal = X_test[X_test.flag == 0]
anomaly = X_test[X_test.flag == 1]

NORMAL_COLOR  = "#3266ad"
ANOMALY_COLOR = "#d34444"
ALPHA = 0.5

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#eeeeee",
    "grid.linewidth":    0.6,
    "font.size":         11,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.subplots_adjust(wspace=0.35)

#Anomaly score distribution (normal vs flagged)
ax = axes[0]
bins = np.linspace(X_test["anomaly_score"].min(), X_test["anomaly_score"].max(), 45)
ax.hist(normal["anomaly_score"],  bins=bins, color=NORMAL_COLOR,  alpha=ALPHA, density=True, label="normal")
ax.hist(anomaly["anomaly_score"], bins=bins, color=ANOMALY_COLOR, alpha=ALPHA, density=True, label="anomaly")
ax.set_xlabel("anomaly score")
ax.set_ylabel("density")
ax.set_title("anomaly score distribution", fontsize=11, fontweight="normal", pad=10)
ax.legend(frameon=False, fontsize=10)

#Transaction amount vs account balance scatter
ax = axes[1]
ax.scatter(normal["AccountBalance"],  normal["TransactionAmount"],
           c=NORMAL_COLOR,  alpha=0.25, s=8,  label="normal",  linewidths=0)
ax.scatter(anomaly["AccountBalance"], anomaly["TransactionAmount"],
           c=ANOMALY_COLOR, alpha=0.65, s=18, label="anomaly", linewidths=0)
ax.set_xlabel("account balance")
ax.set_ylabel("transaction amount")
ax.set_title("transaction amount vs account balance", fontsize=11, fontweight="normal", pad=10)
ax.legend(frameon=False, fontsize=10, markerscale=1.8)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))


#Sorted anomaly scores vs contamination threshold
ax = axes[2]
sorted_scores = np.sort(X_test["anomaly_score"].values)
n = len(sorted_scores)
threshold_idx = int(n * (1 - X_test["flag"].mean()))  # point where flags begin
threshold_val = sorted_scores[threshold_idx]

ax.plot(np.arange(n), sorted_scores, color=NORMAL_COLOR, lw=1.2, zorder=2)
ax.fill_between(np.arange(threshold_idx, n), sorted_scores[threshold_idx:],
                threshold_val, color=ANOMALY_COLOR, alpha=0.15, zorder=1)
ax.axhline(threshold_val, color=ANOMALY_COLOR, lw=1.2, linestyle="--",
           label=f"threshold = {threshold_val:.3f}")
ax.axvline(threshold_idx, color="#aaaaaa", lw=0.8, linestyle=":")
ax.text(threshold_idx + n * 0.01, sorted_scores.min() + (sorted_scores.max() - sorted_scores.min()) * 0.05,
        f"{X_test['flag'].mean():.1%} flagged", fontsize=9, color=ANOMALY_COLOR)
ax.set_xlabel("samples (sorted by score)")
ax.set_ylabel("anomaly score")
ax.set_title("score vs contamination threshold", fontsize=11, fontweight="normal", pad=10)
ax.legend(frameon=False, fontsize=10)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

out_path = os.path.join(OUTPUT_DIR, "isolation_forest_diagnostics.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved → {out_path}")