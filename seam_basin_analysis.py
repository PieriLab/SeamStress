"""
seam_basin_analysis.py

Seam basin analysis: projects SP geometries and MECIs into a shared densMAP
embedding (2D or 3D), estimates the density landscape via KDE, finds density
maxima, and computes per-MECI quantitative metrics:

    rho_m  : KDE density at the embedded MECI position
    d_m    : distance from MECI to the nearest density maximum
    N_k    : population of the basin nearest to the MECI

All output filenames are suffixed with _2d or _3d so runs never overwrite
each other.

Outputs (2D run, --n-components 2)
-----------------------------------
  spawn_embedding_2d.npy
  meci_embedding_2d.npy
  meci_metrics_meanshift_2d.csv
  meci_metrics_grid_2d.csv
  basin_summary_meanshift_2d.csv
  basin_summary_grid_2d.csv
  density_scatter_2d.png
  density_contours_2d.png

Outputs (3D run, --n-components 3)
-----------------------------------
  spawn_embedding_3d.npy
  meci_embedding_3d.npy
  meci_metrics_grid_3d.csv          (mean shift not run in 3D)
  basin_summary_grid_3d.csv
  density_scatter_3d.png            (3 pairwise projections)

Usage
-----
  python seam_basin_analysis.py -d ethylene -o seam_basin/ethylene
  python seam_basin_analysis.py -d ethylene -o seam_basin/ethylene --n-components 3
"""

import os
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.ndimage import maximum_filter
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from umap import UMAP


# ── Dataset defaults ────────────────────────────────────────────────────────

DATASET_PATHS = {
    "ethylene": {
        "spawn_dir": "data/ethylene/aligned_spawns/multi_ref",
        "meci_dir":  "data/ethylene/mecis_aligned",
    },
    "butadiene_s0": {
        "spawn_dir": "data/butadiene_s0/aligned_spawns/multi_ref",
        "meci_dir":  "data/butadiene_s0/mecis_aligned",
    },
    "butadiene_s1": {
        "spawn_dir": "data/butadiene_s1/aligned_spawns/multi_ref",
        "meci_dir":  "data/butadiene_s1/mecis_aligned",
    },
    "benzene_s0": {
        "spawn_dir": "data/benzene_s0/aligned_spawns/single_ref",
        "meci_dir":  "data/benzene_s0/mecis_aligned",
    },
    "benzene_s1": {
        "spawn_dir": "data/benzene_s1/aligned_spawns/single_ref",
        "meci_dir":  "data/benzene_s1/mecis_aligned",
    },
}

COMP_LABELS = {2: ["densMAP 1", "densMAP 2"],
               3: ["densMAP 1", "densMAP 2", "densMAP 3"]}


# ── IO ───────────────────────────────────────────────────────────────────────

def load_xyz_folder(folder):
    folder = Path(folder)
    files = sorted(folder.glob("*.xyz"))
    if not files:
        raise ValueError(f"No .xyz files found in {folder}")
    coords, names = [], []
    n_atoms = None
    for f in files:
        with open(f) as fh:
            lines = fh.readlines()
        na = int(lines[0].strip())
        if n_atoms is None:
            n_atoms = na
        elif na != n_atoms:
            raise ValueError(f"Inconsistent atom count in {f}")
        xyz = np.array([list(map(float, l.split()[1:4])) for l in lines[2:2 + na]])
        coords.append(xyz.reshape(-1))
        names.append(f.name)
    return np.array(coords), names


# ── Bandwidth selection ──────────────────────────────────────────────────────

def select_bandwidth(X, mode):
    if mode == "manual":
        return 0.9
    elif mode == "cv":
        params = {"bandwidth": np.logspace(-2, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params, cv=5)
        grid.fit(X)
        bw = grid.best_params_["bandwidth"]
        print(f"  CV-selected bandwidth: {bw:.4f}")
        return bw
    elif mode == "scott":
        n, d = X.shape
        return n ** (-1.0 / (d + 4))
    else:
        raise ValueError(f"Unknown bandwidth mode: {mode}")


# ── KDE ─────────────────────────────────────────────────────────────────────

def fit_kde(X, bandwidth):
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(X)
    return kde


def kde_density_normalised(kde, X):
    d = np.exp(kde.score_samples(X))
    return d / d.sum()


# ── Mean shift (2D only) ─────────────────────────────────────────────────────

def mean_shift(X, bandwidth, tol=1e-3, max_iter=300, min_members=5):
    N = len(X)
    modes = []
    for i in range(N):
        x = X[i].copy()
        for _ in range(max_iter):
            diff = X - x
            w = np.exp(-np.sum(diff**2, axis=1) / (2 * bandwidth**2))
            x_new = (w[:, None] * X).sum(0) / w.sum()
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        modes.append(x)
    modes = np.array(modes)

    unique_modes = []
    for m in modes:
        if not unique_modes:
            unique_modes.append(m)
        else:
            dists = np.linalg.norm(np.array(unique_modes) - m, axis=1)
            if dists.min() >= bandwidth / 2:
                unique_modes.append(m)
    unique_modes = np.array(unique_modes)

    assignments = np.argmin(
        np.linalg.norm(X[:, None, :] - unique_modes[None, :, :], axis=2), axis=1
    )
    valid = [i for i in range(len(unique_modes))
             if np.sum(assignments == i) >= min_members]
    if not valid:
        raise RuntimeError(
            "No density maxima survived the minimum-members filter. "
            "Try reducing --min-members or adjusting the bandwidth."
        )
    valid_modes = unique_modes[valid]
    assignments = np.argmin(
        np.linalg.norm(X[:, None, :] - valid_modes[None, :, :], axis=2), axis=1
    )
    populations = {k: int(np.sum(assignments == k)) for k in range(len(valid_modes))}
    return valid_modes, assignments, populations


# ── Grid-based peak finding (nD) ─────────────────────────────────────────────

def find_maxima_grid(emb_spawn, kde, n_grid, min_distance, threshold_rel):
    """
    Find local maxima of the KDE surface on a regular grid.
    Works for both 2D and 3D embeddings.

    For 3D use a coarser grid (e.g. n_grid=80) to keep memory manageable.
    """
    ndim = emb_spawn.shape[1]
    pad = 0.05

    axes = []
    for d in range(ndim):
        lo, hi = emb_spawn[:, d].min(), emb_spawn[:, d].max()
        p = (hi - lo) * pad
        axes.append(np.linspace(lo - p, hi + p, n_grid))

    # Build grid
    grids = np.meshgrid(*axes, indexing="ij")           # each (n_grid,)*ndim
    grid_pts = np.column_stack([g.ravel() for g in grids])

    zz = np.exp(kde.score_samples(grid_pts)).reshape([n_grid] * ndim)

    footprint = np.ones([min_distance * 2 + 1] * ndim)
    local_max = (maximum_filter(zz, footprint=footprint) == zz)
    local_max &= zz >= threshold_rel * zz.max()

    peak_idx = np.argwhere(local_max)   # shape (M, ndim) — indices per axis
    peaks = np.array([[axes[d][idx[d]] for d in range(ndim)]
                      for idx in peak_idx])

    return peaks, zz, axes


# ── Per-MECI metrics helper ───────────────────────────────────────────────────

def compute_meci_metrics(emb_meci, emb_spawn, kde, modes, populations):
    # rho_m
    norm_factor = np.exp(kde.score_samples(emb_spawn)).sum()
    rho_m = np.exp(kde.score_samples(emb_meci)) / norm_factor

    # d_m and nearest basin
    dists = np.linalg.norm(emb_meci[:, None, :] - modes[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    d_m = dists[np.arange(len(emb_meci)), nearest]
    nk = [populations[b] for b in nearest]

    return rho_m, nearest, d_m, nk


def basin_populations(emb_spawn, modes):
    assignments = np.argmin(
        np.linalg.norm(emb_spawn[:, None, :] - modes[None, :, :], axis=2), axis=1
    )
    populations = {k: int(np.sum(assignments == k)) for k in range(len(modes))}
    return assignments, populations


# ── Plots ────────────────────────────────────────────────────────────────────

def _scatter_panel(ax, emb, meci_emb, modes, density, meci_names,
                   xi, yi, label_xi, label_yi, title, pad_frac=0.2):
    sc = ax.scatter(emb[:, xi], emb[:, yi], c=density, s=6,
                    cmap="viridis", alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Normalised KDE density")
    ax.scatter(meci_emb[:, xi], meci_emb[:, yi],
               marker="*", s=200, c="red", edgecolor="k", zorder=5, label="MECIs")
    ax.scatter(modes[:, xi], modes[:, yi],
               marker="D", s=100, c="yellow", edgecolor="k", zorder=5,
               label="Density maxima")
    for i, name in enumerate(meci_names):
        ax.annotate(name.replace(".xyz", ""), (meci_emb[i, xi], meci_emb[i, yi]),
                    fontsize=7, ha="left", va="bottom")
    ax.set_xlabel(label_xi, fontsize=12)
    ax.set_ylabel(label_yi, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # Zoom to range spanned by MECIs only, with padding
    xlo, xhi = meci_emb[:, xi].min(), meci_emb[:, xi].max()
    ylo, yhi = meci_emb[:, yi].min(), meci_emb[:, yi].max()
    xpad = (xhi - xlo) * pad_frac or 1.0
    ypad = (yhi - ylo) * pad_frac or 1.0
    ax.set_xlim(xlo - xpad, xhi + xpad)
    ax.set_ylim(ylo - ypad, yhi + ypad)


def plot_2d(emb_spawn, emb_meci, modes_ms, modes_grid, spawn_density,
            meci_names, kde, out_dir, tag):
    labels = COMP_LABELS[2]

    # ── Scatter ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, modes, title in zip(axes,
                                 [modes_ms, modes_grid],
                                 [f"Mean shift  ({len(modes_ms)} maxima)",
                                  f"Grid-based  ({len(modes_grid)} maxima)"]):
        _scatter_panel(ax, emb_spawn, emb_meci, modes, spawn_density,
                       meci_names, 0, 1, labels[0], labels[1], title)
    plt.tight_layout()
    p = out_dir / f"density_scatter_{tag}.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")

    # ── Contours ──
    x0, x1 = emb_spawn[:, 0].min(), emb_spawn[:, 0].max()
    y0, y1 = emb_spawn[:, 1].min(), emb_spawn[:, 1].max()
    px, py = (x1 - x0) * 0.05, (y1 - y0) * 0.05
    xx, yy = np.meshgrid(np.linspace(x0 - px, x1 + px, 200),
                         np.linspace(y0 - py, y1 + py, 200))
    zz = np.exp(kde.score_samples(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, modes, title in zip(axes,
                                 [modes_ms, modes_grid],
                                 [f"Mean shift  ({len(modes_ms)} maxima)",
                                  f"Grid-based  ({len(modes_grid)} maxima)"]):
        cf = ax.contourf(xx, yy, zz, levels=50, cmap="viridis")
        ax.contour(xx, yy, zz, levels=10, colors="k", linewidths=0.4)
        plt.colorbar(cf, ax=ax, label="KDE density")
        ax.scatter(emb_meci[:, 0], emb_meci[:, 1],
                   marker="*", s=200, c="red", edgecolor="k", zorder=5, label="MECIs")
        ax.scatter(modes[:, 0], modes[:, 1],
                   marker="D", s=100, c="yellow", edgecolor="k", zorder=5,
                   label="Density maxima")
        for i, name in enumerate(meci_names):
            ax.annotate(name.replace(".xyz", ""), emb_meci[i],
                        fontsize=7, ha="left", va="bottom")
        ax.set_xlabel(labels[0], fontsize=12); ax.set_ylabel(labels[1], fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold"); ax.legend(fontsize=9)
    plt.tight_layout()
    p = out_dir / f"density_contours_{tag}.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")


def plot_3d_projections(emb_spawn, emb_meci, modes, spawn_density, meci_names, out_dir, tag):
    labels = COMP_LABELS[3]
    pairs = list(itertools.combinations(range(3), 2))   # (0,1),(0,2),(1,2)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (xi, yi) in zip(axes, pairs):
        _scatter_panel(ax, emb_spawn, emb_meci, modes, spawn_density,
                       meci_names, xi, yi,
                       labels[xi], labels[yi],
                       f"{labels[xi]} vs {labels[yi]}  ({len(modes)} maxima)")
    plt.tight_layout()
    p = out_dir / f"density_scatter_{tag}.png"
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  Saved → {p}")


# ── Main pipeline ────────────────────────────────────────────────────────────

def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    nc  = args.n_components
    tag = f"{nc}d"

    # ── Load data ──
    print("Loading spawns...")
    X_spawn, spawn_names = load_xyz_folder(args.spawn_dir)
    print(f"  {len(X_spawn)} spawns loaded.")
    print("Loading MECIs...")
    X_meci, meci_names = load_xyz_folder(args.meci_dir)
    print(f"  {len(X_meci)} MECIs loaded.")

    # ── Scale (fit on spawns only) ──
    scaler = StandardScaler().fit(X_spawn)
    X_all_scaled = np.vstack([scaler.transform(X_spawn), scaler.transform(X_meci)])

    # ── densMAP embedding ──
    emb_path_spawn = out_dir / f"spawn_embedding_{tag}.npy"
    emb_path_meci  = out_dir / f"meci_embedding_{tag}.npy"

    print(f"Fitting densMAP ({nc}D) on spawns + MECIs jointly...")
    n_neighbors = min(args.n_neighbors, len(X_all_scaled) - 1)
    densmap_model = UMAP(
        densmap=True,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=nc,
        random_state=42,
    )
    emb_all   = densmap_model.fit_transform(X_all_scaled)
    emb_spawn = emb_all[:len(X_spawn)]
    emb_meci  = emb_all[len(X_spawn):]

    np.save(emb_path_spawn, emb_spawn)
    np.save(emb_path_meci,  emb_meci)
    print(f"  Saved embeddings → {emb_path_spawn}, {emb_path_meci}")

    # ── KDE ──
    print(f"Selecting bandwidth (mode={args.bandwidth_mode})...")
    bw  = select_bandwidth(emb_spawn, args.bandwidth_mode)
    kde = fit_kde(emb_spawn, bw)
    spawn_density = kde_density_normalised(kde, emb_spawn)

    # ── Method 1: mean shift (2D only) ──
    if nc == 2:
        print("Running mean shift...")
        modes_ms, _, pops_ms = mean_shift(
            emb_spawn, bandwidth=bw, min_members=args.min_members)
        print(f"  Mean shift: {len(modes_ms)} maxima")
        _, pops_ms = basin_populations(emb_spawn, modes_ms)   # re-derive clean dict
        rho_m, nb, dm, nk = compute_meci_metrics(emb_meci, emb_spawn, kde, modes_ms, pops_ms)
        _save_results(out_dir, tag, "meanshift", meci_names, emb_meci,
                      rho_m, nb, dm, nk, modes_ms, pops_ms)

    # ── Method 2: grid-based ──
    n_grid = args.grid_size_3d if nc == 3 else args.grid_size
    print(f"Running grid-based peak finding ({n_grid}^{nc} grid)...")
    modes_grid, _, _ = find_maxima_grid(
        emb_spawn, kde,
        n_grid=n_grid,
        min_distance=args.min_peak_distance,
        threshold_rel=args.peak_threshold,
    )
    print(f"  Grid-based: {len(modes_grid)} maxima")
    _, pops_grid = basin_populations(emb_spawn, modes_grid)
    rho_m, nb, dm, nk = compute_meci_metrics(emb_meci, emb_spawn, kde, modes_grid, pops_grid)
    _save_results(out_dir, tag, "grid", meci_names, emb_meci,
                  rho_m, nb, dm, nk, modes_grid, pops_grid)

    if nc == 2:
        print(f"\n  Comparison: mean shift={len(modes_ms)}, grid={len(modes_grid)}")

    # ── Plots ──
    if nc == 2:
        plot_2d(emb_spawn, emb_meci, modes_ms, modes_grid, spawn_density,
                meci_names, kde, out_dir, tag)
    else:
        plot_3d_projections(emb_spawn, emb_meci, modes_grid, spawn_density,
                            meci_names, out_dir, tag)


def _save_results(out_dir, tag, method, meci_names, emb_meci,
                  rho_m, nearest, d_m, nk, modes, populations):
    nc = emb_meci.shape[1]
    coord_cols = {f"emb_{c}": emb_meci[:, i]
                  for i, c in enumerate(["x", "y", "z"][:nc])}
    meci_df = pd.DataFrame({
        "meci": [n.replace(".xyz", "") for n in meci_names],
        **coord_cols,
        "rho_m": rho_m,
        "nearest_basin": nearest,
        "d_m": d_m,
        "basin_N_k": nk,
    })
    path = out_dir / f"meci_metrics_{method}_{tag}.csv"
    meci_df.to_csv(path, index=False)
    print(f"\nMECI metrics ({method}, {tag}) → {path}")
    print(meci_df.to_string(index=False))

    centroid_cols = {f"centroid_{c}": modes[:, i]
                     for i, c in enumerate(["x", "y", "z"][:nc])}
    basin_df = pd.DataFrame({
        "basin_id": list(populations.keys()),
        "N_k":      list(populations.values()),
        **centroid_cols,
    })
    path = out_dir / f"basin_summary_{method}_{tag}.csv"
    basin_df.to_csv(path, index=False)
    print(f"Basin summary ({method}, {tag}) → {path}")
    print(basin_df.to_string(index=False))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Seam basin analysis: densMAP embedding + KDE + peak finding"
    )
    parser.add_argument("-d", "--dataset", choices=list(DATASET_PATHS.keys()))
    parser.add_argument("--spawn-dir")
    parser.add_argument("--meci-dir")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--n-components", type=int, default=2, choices=[2, 3],
                        help="densMAP dimensionality (default: 2)")
    parser.add_argument("--bandwidth-mode", choices=["manual", "cv", "scott"],
                        default="cv")
    parser.add_argument("--min-members", type=int, default=5,
                        help="Min spawns per mode for mean shift (default: 5)")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--grid-size", type=int, default=300,
                        help="Grid resolution for 2D peak finding (default: 300)")
    parser.add_argument("--grid-size-3d", type=int, default=80,
                        help="Grid resolution per axis for 3D peak finding (default: 80)")
    parser.add_argument("--min-peak-distance", type=int, default=10,
                        help="Min separation between peaks in grid cells (default: 10)")
    parser.add_argument("--peak-threshold", type=float, default=0.01,
                        help="Discard peaks below this fraction of global max (default: 0.01)")
    args = parser.parse_args()

    if args.dataset:
        defaults = DATASET_PATHS[args.dataset]
        if not args.spawn_dir:
            args.spawn_dir = defaults["spawn_dir"]
        if not args.meci_dir:
            args.meci_dir = defaults["meci_dir"]

    if not args.spawn_dir or not args.meci_dir:
        parser.error("Provide --dataset or both --spawn-dir and --meci-dir")

    run(args)


if __name__ == "__main__":
    main()
