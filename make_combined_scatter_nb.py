"""Writes plotting_notebooks/combined_scatter.ipynb with all fixes applied."""
import json

CELL_IMPORTS = """\
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pickle, pandas as pd, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
from scipy.ndimage import maximum_filter
"""

CELL_CONFIG = """\
from pathlib import Path
# Path to repo root — change if needed
BASE_DIR = Path("/Users/elipieri/repos/SeamStress")

# ── Panel titles ──────────────────────────────────────────────────────────────
TITLES = {
    "ethylene":     "Ethylene",
    "butadiene_s0": r"Butadiene S$_0$",
    "butadiene_s1": r"Butadiene S$_1$",
    "benzene_s0":   r"Benzene S$_0$",
    "benzene_s1":   r"Benzene S$_1$",
}

# ── Order of panels (top-left to bottom-right, row by row) ───────────────────
DATASET_ORDER = ["ethylene", "butadiene_s0", "butadiene_s1", "benzene_s0", "benzene_s1"]

# ── Per-dataset peak-finding parameters ──────────────────────────────────────
DATASET_PARAMS = {
    "ethylene":     dict(min_peak_dist=10, peak_threshold=0.005, n_grid=200),
    "butadiene_s0": dict(min_peak_dist=5, peak_threshold=0.1,   n_grid=200),
    "butadiene_s1": dict(min_peak_dist=5, peak_threshold=0.01,  n_grid=200),
    "benzene_s0":   dict(min_peak_dist=5, peak_threshold=0.005, n_grid=200),
    "benzene_s1":   dict(min_peak_dist=5, peak_threshold=0.005, n_grid=200),
}

# ── Custom MECI display labels ({} = use raw name) ───────────────────────────
MECI_LABELS = {
    "ethylene":     {},
    "butadiene_s0": {},
    "butadiene_s1": {},
    "benzene_s0":   {},
    "benzene_s1":   {},
}
MECI_ANNOTATE = True   # set False to hide all MECI labels

# ── Manual axis limits (None = auto-zoom to MECI range + PAD_FRAC) ───────────
MANUAL_LIMITS = {
    "ethylene":     None,
    "butadiene_s0": None,
    "butadiene_s1": None,
    "benzene_s0":   None,
    "benzene_s1":   None,
}
PAD_FRAC = 0.20

# ── Appearance ────────────────────────────────────────────────────────────────
DENSITY_CMAP  = "rocket"
SCATTER_S     = 6
SCATTER_ALPHA = 0.6
MECI_S        = 120   # size of the most-populated MECI star
MECI_S_MIN    = 30    # minimum star size (least-populated MECI)
BASIN_S       = 150
LABEL_SIZE    = 11
TICK_SIZE     = 9
ANNOT_SIZE    = 6

# ── Figure size and output ────────────────────────────────────────────────────
FIG_W       = 4.5
FIG_H       = 4.5
OUTPUT_DPI  = 150
OUTPUT_FILE = str(BASE_DIR / "seam_basin" / "combined_scatter_dm1_dm2.png")
"""

CELL_HELPERS = """\
def load_dataset(name):
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    out = BASE_DIR / "seam_basin" / name
    emb_spawn = np.load(out / "spawn_embedding_3d.npy")
    emb_meci  = np.load(out / "meci_embedding_3d.npy")

    # KDE: load cache or refit
    kde_caches = sorted(out.glob("kde_cache_bw*.pkl"))
    if kde_caches:
        with open(kde_caches[0], "rb") as f:
            cache = pickle.load(f)
        kde           = cache["kde"]
        spawn_density = cache["spawn_density"]
        print(f"{name}: loaded KDE cache")
    else:
        print(f"{name}: no KDE cache, refitting...")
        bw_grid = GridSearchCV(KernelDensity(), {"bandwidth": np.logspace(-1, 1, 20)}, cv=5)
        bw_grid.fit(emb_spawn)
        kde = bw_grid.best_estimator_
        d   = np.exp(kde.score_samples(emb_spawn))
        spawn_density = d / d.sum()
        cache = {"kde": kde, "bw": kde.bandwidth, "spawn_density": spawn_density}
        with open(out / f"kde_cache_bw{kde.bandwidth:.4f}.pkl", "wb") as f:
            pickle.dump(cache, f)
        print(f"{name}: KDE fitted and cached")

    # Grid: load cache or recompute
    N_GRID = DATASET_PARAMS[name].get("n_grid", 200)
    grid_caches = sorted(out.glob(f"kde_grid_cache_bw*_n{N_GRID}.npy"))
    axes_caches = sorted(out.glob(f"kde_grid_axes_bw*_n{N_GRID}.pkl"))
    if grid_caches and axes_caches:
        zz3d = np.load(grid_caches[0])
        with open(axes_caches[0], "rb") as f:
            axes_1d = pickle.load(f)
        print(f"{name}: loaded grid cache")
    else:
        print(f"{name}: no grid cache, recomputing...")
        axes_1d = []
        for d in range(3):
            lo, hi = emb_spawn[:, d].min(), emb_spawn[:, d].max()
            p = (hi - lo) * 0.05
            axes_1d.append(np.linspace(lo - p, hi + p, N_GRID))
        grids    = np.meshgrid(*axes_1d, indexing="ij")
        grid_pts = np.column_stack([g.ravel() for g in grids])
        zz3d     = np.exp(kde.score_samples(grid_pts)).reshape([N_GRID] * 3)
        bw       = getattr(kde, "bandwidth", "None")
        bw_str   = f"{bw:.4f}" if isinstance(bw, float) else str(bw)
        np.save(out / f"kde_grid_cache_bw{bw_str}_n{N_GRID}.npy", zz3d)
        with open(out / f"kde_grid_axes_bw{bw_str}_n{N_GRID}.pkl", "wb") as f:
            pickle.dump(axes_1d, f)
        print(f"{name}: grid computed and cached")

    metrics    = pd.read_csv(out / "meci_metrics_grid_3d.csv")
    meci_names = metrics["meci"].tolist()
    return emb_spawn, emb_meci, spawn_density, zz3d, axes_1d, meci_names


def _norm(name):
    return str(name).strip().lower().replace('_', '').replace(' ', '')


def load_meci_population(dataset_name):
    '''Read FC_MECI_stats.xlsx; return {normalised_meci_name: population}.'''
    path = BASE_DIR / "data" / dataset_name / "FC_MECI_stats.xlsx"
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    with zipfile.ZipFile(path) as z:
        ss_root = ET.fromstring(z.read('xl/sharedStrings.xml'))
        strings = [si.find(f'{{{ns}}}t').text for si in ss_root.findall(f'{{{ns}}}si')]
        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = []
        for row in sheet.findall(f'.//{{{ns}}}row'):
            cells = []
            for c in row.findall(f'{{{ns}}}c'):
                t = c.get('t', '')
                v = c.find(f'{{{ns}}}v')
                if v is None:
                    cells.append('')
                elif t == 's':
                    cells.append(strings[int(v.text)])
                else:
                    cells.append(v.text)
            if any(cells):
                rows.append(cells)
    header = [h.lower() for h in rows[0]]
    pop_col = header.index('pop')
    result = {}
    for row in rows[1:]:
        if len(row) <= pop_col:
            continue
        name = row[0].strip()
        pop_str = row[pop_col].strip() if row[pop_col] else ''
        if not name or pop_str in ('-', '', 'None', None):
            continue
        try:
            result[_norm(name)] = float(pop_str)
        except ValueError:
            pass
    return result


def meci_sizes(meci_names, pop_dict):
    '''Return per-MECI marker sizes scaled by population.'''
    pops = np.array([pop_dict.get(_norm(m), 0.0) for m in meci_names])
    max_pop = pops.max()
    if max_pop <= 0:
        return np.full(len(meci_names), MECI_S)
    return MECI_S_MIN + (MECI_S - MECI_S_MIN) * pops / max_pop


def find_modes(zz3d, axes_1d, min_peak_dist, peak_threshold, **kwargs):
    fp        = np.ones([min_peak_dist * 2 + 1] * 3)
    local_max = (maximum_filter(zz3d, footprint=fp) == zz3d)
    local_max &= zz3d >= peak_threshold * zz3d.max()
    peak_idx  = np.argwhere(local_max)
    return np.array([[axes_1d[d][idx[d]] for d in range(3)] for idx in peak_idx])


def draw_panel(ax, emb_spawn, emb_meci, spawn_density, modes, meci_names,
               title, name, xi=0, yi=1):
    sc = ax.scatter(
        emb_spawn[:, xi], emb_spawn[:, yi],
        c=spawn_density, s=SCATTER_S, alpha=SCATTER_ALPHA,
        cmap=DENSITY_CMAP, rasterized=True,
    )
    pop_dict = load_meci_population(name)
    sizes = meci_sizes(meci_names, pop_dict)
    for i in range(len(meci_names)):
        ax.scatter(
            emb_meci[i, xi], emb_meci[i, yi],
            marker="*", s=sizes[i], c="red", edgecolor="k", linewidths=0.5,
            zorder=5, label="MECIs" if i == 0 else "_nolegend_",
        )
    ax.scatter(
        modes[:, xi], modes[:, yi],
        marker="D", s=BASIN_S, c="yellow", edgecolor="k", linewidths=0.8,
        zorder=5, label="Density maxima",
    )
    if MECI_ANNOTATE:
        label_map = MECI_LABELS.get(name, {})
        for i, raw_name in enumerate(meci_names):
            display = label_map.get(raw_name, raw_name)
            ax.annotate(
                display,
                (emb_meci[i, xi], emb_meci[i, yi]),
                fontsize=ANNOT_SIZE, ha="left", va="bottom",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            )
    manual = MANUAL_LIMITS.get(name)
    if manual is not None:
        ax.set_xlim(manual[0], manual[1])
        ax.set_ylim(manual[2], manual[3])
    else:
        xlo, xhi = emb_meci[:, xi].min(), emb_meci[:, xi].max()
        ylo, yhi = emb_meci[:, yi].min(), emb_meci[:, yi].max()
        xpad = (xhi - xlo) * PAD_FRAC or 1.0
        ypad = (yhi - ylo) * PAD_FRAC or 1.0
        ax.set_xlim(xlo - xpad, xhi + xpad)
        ax.set_ylim(ylo - ypad, yhi + ypad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("densMAP 1", fontsize=LABEL_SIZE)
    ax.set_ylabel("densMAP 2", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title(title, fontsize=LABEL_SIZE + 1, fontweight="bold")
    return sc
"""

CELL_PLOT = """\
fig, axes = plt.subplots(2, 3, figsize=(FIG_W * 3, FIG_H * 2), constrained_layout=True)
axes_flat  = axes.flatten()
last_sc    = None

for i, name in enumerate(DATASET_ORDER):
    ax = axes_flat[i]
    emb_spawn, emb_meci, spawn_density, zz3d, axes_1d, meci_names = load_dataset(name)
    modes   = find_modes(zz3d, axes_1d, **DATASET_PARAMS[name])
    last_sc = draw_panel(ax, emb_spawn, emb_meci, spawn_density, modes,
                         meci_names, title=TITLES[name], name=name)

# ── Legend panel (bottom-right) ───────────────────────────────────────────────
ax_leg = axes_flat[5]
ax_leg.set_axis_off()

legend_handles = [
    plt.scatter([], [], marker="*", s=MECI_S,     c="red",    edgecolor="k", linewidths=0.5, label="MECI (high pop.)"),
    plt.scatter([], [], marker="*", s=MECI_S_MIN, c="red",    edgecolor="k", linewidths=0.5, label="MECI (low pop.)"),
    plt.scatter([], [], marker="D", s=BASIN_S,    c="yellow", edgecolor="k", linewidths=0.8, label="Density maxima"),
]
ax_leg.legend(
    handles=legend_handles,
    fontsize=LABEL_SIZE + 1,
    loc="upper center",
    frameon=True,
    framealpha=0.9,
)

cbar = fig.colorbar(last_sc, ax=ax_leg, orientation="horizontal",
                    shrink=0.7, pad=0.05, location="bottom")
cbar.set_label("KDE density", fontsize=LABEL_SIZE)
cbar.ax.tick_params(labelsize=TICK_SIZE)

fig.savefig(OUTPUT_FILE, dpi=OUTPUT_DPI, bbox_inches="tight")
print(f"Saved -> {OUTPUT_FILE}")
plt.show()
"""

def make_cell(cell_id, source):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": [
        make_cell("c1_imports", CELL_IMPORTS),
        make_cell("c2_config",  CELL_CONFIG),
        make_cell("c3_helpers", CELL_HELPERS),
        make_cell("c4_plot",    CELL_PLOT),
    ],
}

out = "plotting_notebooks/combined_scatter.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Written -> {out}")
