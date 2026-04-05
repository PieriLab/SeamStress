"""
Combined 2×3 scatter figure: densMAP component 1 vs 2 for all datasets.
Panel 6 holds the shared legend.

Usage:
    python plot_combined_scatter.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pickle
from pathlib import Path
from scipy.ndimage import maximum_filter

# ── Tuneable parameters ───────────────────────────────────────────────────────

TITLES = {
    "ethylene":     "Ethylene",
    "butadiene_s0": "Butadiene S$_0$",
    "butadiene_s1": "Butadiene S$_1$",
    "benzene_s0":   "Benzene S$_0$",
    "benzene_s1":   "Benzene S$_1$",
}

# Per-dataset peak-finding parameters
DATASET_PARAMS = {
    "ethylene":     dict(min_peak_dist=5, peak_threshold=0.005),
    "butadiene_s0": dict(min_peak_dist=5, peak_threshold=0.1),
    "butadiene_s1": dict(min_peak_dist=5, peak_threshold=0.01),
    "benzene_s0":   dict(min_peak_dist=5, peak_threshold=0.005),
    "benzene_s1":   dict(min_peak_dist=5, peak_threshold=0.005),
}

DATASET_ORDER = ["ethylene", "butadiene_s0", "butadiene_s1", "benzene_s0", "benzene_s1"]

# Figure layout
FIG_W       = 4.5      # width of each panel (inches)
FIG_H       = 4.5      # height of each panel (inches)
PAD_FRAC    = 0.20     # fractional padding around MECI range

# Scatter appearance
SCATTER_S     = 6
SCATTER_ALPHA = 0.6
DENSITY_CMAP  = "viridis"
MECI_S        = 120
BASIN_S       = 150
LABEL_SIZE    = 11
TICK_SIZE     = 9
MECI_ANNOTATE = True    # set False to hide MECI labels

OUTPUT_FILE   = "seam_basin/combined_scatter_dm1_dm2.png"
OUTPUT_DPI    = 150

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dataset(name):
    out = Path("seam_basin") / name
    emb_spawn = np.load(out / "spawn_embedding_3d.npy")
    emb_meci  = np.load(out / "meci_embedding_3d.npy")

    # KDE cache
    kde_caches = list(out.glob("kde_cache_bw*.pkl"))
    with open(kde_caches[0], "rb") as f:
        cache = pickle.load(f)
    kde           = cache["kde"]
    spawn_density = cache["spawn_density"]

    # Grid cache
    grid_caches = list(out.glob("kde_grid_cache_bw*_n200.npy"))
    axes_caches = list(out.glob("kde_grid_axes_bw*_n200.pkl"))
    zz3d = np.load(grid_caches[0])
    with open(axes_caches[0], "rb") as f:
        axes_1d = pickle.load(f)

    # MECI names from metrics CSV
    import pandas as pd
    metrics = pd.read_csv(out / "meci_metrics_grid_3d.csv")
    meci_names = metrics["meci"].tolist()

    return emb_spawn, emb_meci, spawn_density, zz3d, axes_1d, meci_names


def find_modes(zz3d, axes_1d, min_peak_dist, peak_threshold):
    fp       = np.ones([min_peak_dist * 2 + 1] * 3)
    local_max = (maximum_filter(zz3d, footprint=fp) == zz3d)
    local_max &= zz3d >= peak_threshold * zz3d.max()
    peak_idx  = np.argwhere(local_max)
    modes     = np.array([[axes_1d[d][idx[d]] for d in range(3)] for idx in peak_idx])
    return modes


def draw_panel(ax, emb_spawn, emb_meci, spawn_density, modes, meci_names,
               title, xi=0, yi=1):
    sc = ax.scatter(
        emb_spawn[:, xi], emb_spawn[:, yi],
        c=spawn_density, s=SCATTER_S, alpha=SCATTER_ALPHA,
        cmap=DENSITY_CMAP, rasterized=True,
    )

    ax.scatter(
        emb_meci[:, xi], emb_meci[:, yi],
        marker="*", s=MECI_S, c="red", edgecolor="k", linewidths=0.5,
        zorder=5, label="MECIs",
    )

    ax.scatter(
        modes[:, xi], modes[:, yi],
        marker="D", s=BASIN_S, c="yellow", edgecolor="k", linewidths=0.8,
        zorder=5, label="Density maxima",
    )

    if MECI_ANNOTATE:
        for i, name in enumerate(meci_names):
            ax.annotate(
                name,
                (emb_meci[i, xi], emb_meci[i, yi]),
                fontsize=6, ha="left", va="bottom",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            )

    # Zoom to MECI range
    xlo, xhi = emb_meci[:, xi].min(), emb_meci[:, xi].max()
    ylo, yhi = emb_meci[:, yi].min(), emb_meci[:, yi].max()
    xpad = (xhi - xlo) * PAD_FRAC or 1.0
    ypad = (yhi - ylo) * PAD_FRAC or 1.0
    ax.set_xlim(xlo - xpad, xhi + xpad)
    ax.set_ylim(ylo - ypad, yhi + ypad)
    ax.set_aspect("equal", adjustable="datalim")

    ax.set_xlabel("densMAP 1", fontsize=LABEL_SIZE)
    ax.set_ylabel("densMAP 2", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title(title, fontsize=LABEL_SIZE + 1, fontweight="bold")

    return sc


# ── Main ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(FIG_W * 3, FIG_H * 2))
axes_flat = axes.flatten()

last_sc = None

for i, name in enumerate(DATASET_ORDER):
    ax = axes_flat[i]
    emb_spawn, emb_meci, spawn_density, zz3d, axes_1d, meci_names = load_dataset(name)
    params = DATASET_PARAMS[name]
    modes  = find_modes(zz3d, axes_1d, **params)
    last_sc = draw_panel(ax, emb_spawn, emb_meci, spawn_density, modes,
                         meci_names, title=TITLES[name])

# ── Legend panel (bottom-right, index 5) ─────────────────────────────────────
ax_leg = axes_flat[5]
ax_leg.set_axis_off()

legend_handles = [
    plt.scatter([], [], marker="*", s=MECI_S, c="red",   edgecolor="k", linewidths=0.5, label="MECIs"),
    plt.scatter([], [], marker="D", s=BASIN_S, c="yellow", edgecolor="k", linewidths=0.8, label="Density maxima"),
]
cbar_ax = fig.add_axes([
    ax_leg.get_position().x0 + 0.02,
    ax_leg.get_position().y0 + 0.25,
    ax_leg.get_position().width * 0.5,
    ax_leg.get_position().height * 0.08,
])
cbar = fig.colorbar(last_sc, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Normalised KDE density", fontsize=LABEL_SIZE - 1)
cbar.ax.tick_params(labelsize=TICK_SIZE - 1)

ax_leg.legend(
    handles=legend_handles,
    fontsize=LABEL_SIZE,
    loc="center",
    frameon=True,
    framealpha=0.9,
    bbox_to_anchor=(0.5, 0.6),
)

fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=OUTPUT_DPI, bbox_inches="tight")
print(f"Saved → {OUTPUT_FILE}")
plt.show()
