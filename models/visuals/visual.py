import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

RESULTS_PATH  = "models/training/autoencoder_results.csv"
HISTORY_PATH  = "models/training/autoencoder_history.csv"
OUTPUT_DIR    = "visuals"
THRESHOLD_PERCENTILE = 95

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = pd.read_csv(RESULTS_PATH)
history = pd.read_csv(HISTORY_PATH)

train_errors = results.loc[results.split == "train", "reconstruction_error"].values
valid_errors = results.loc[results.split == "valid", "reconstruction_error"].values
test_errors  = results.loc[results.split == "test",  "reconstruction_error"].values
threshold    = np.percentile(valid_errors, THRESHOLD_PERCENTILE)

COLORS = {"train": "#3266ad", "val": "#d34444", "test": "#d34444", "valid": "#34a853"}
ALPHA  = 0.65

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#eeeeee",
    "grid.linewidth":   0.6,
    "font.size":        11,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.subplots_adjust(wspace=0.35)

# --- Plot 1: Loss curves ---
ax = axes[0]
epochs = range(1, len(history) + 1)
ax.plot(epochs, history["train"], color=COLORS["train"], lw=1.5, label="train")
ax.plot(epochs, history["val"],   color=COLORS["val"],   lw=1.5, linestyle="--", label="val")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE loss")
ax.set_title("training & validation loss", fontsize=11, fontweight="normal", pad=10)
ax.legend(frameon=False, fontsize=10)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

# --- Plot 2: Reconstruction error distributions ---
ax = axes[1]
bins = np.linspace(0, max(train_errors.max(), test_errors.max(), valid_errors.max()) * 1.05, 40)
ax.hist(train_errors, bins=bins, color=COLORS["train"], alpha=ALPHA, label="train", density=True)
ax.hist(test_errors,  bins=bins, color=COLORS["test"],  alpha=ALPHA, label="test",  density=True)
ax.hist(valid_errors, bins=bins, color=COLORS["valid"], alpha=ALPHA, label="val",   density=True)
ax.axvline(threshold, color="#e67e22", lw=1.5, linestyle="--",
           label=f"P{THRESHOLD_PERCENTILE} = {threshold:.4f}")
ax.set_xlabel("reconstruction error")
ax.set_ylabel("density")
ax.set_title("error distributions", fontsize=11, fontweight="normal", pad=10)
ax.legend(frameon=False, fontsize=10)

# --- Plot 3: Anomaly flag rates ---
ax = axes[2]
splits = ["train", "valid", "test"]
rates  = [
    results.loc[results.split == s, "flag"].mean() * 100
    for s in splits
]
bar_colors = [COLORS["train"], COLORS["valid"], COLORS["test"]]
bars = ax.bar(["train", "val", "test"], rates, color=bar_colors, width=0.5, zorder=2)
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{rate:.1f}%",
            ha="center", va="bottom", fontsize=10, color="#555555")
ax.set_ylabel("% flagged")
ax.set_title("anomaly flag rate by split", fontsize=11, fontweight="normal", pad=10)
ax.set_ylim(0, max(rates) * 1.3)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

out_path = os.path.join(OUTPUT_DIR, "autoencoder_diagnostics.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved → {out_path}")