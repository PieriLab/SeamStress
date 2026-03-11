"""
Evaluate and compare three master-reference strategies for each dataset.

Three modes compared:
  current_single_ref  — align all spawns to the current reference (see DATASETS)
  multi_ref           — align each spawn to its closest MECI centroid
  meci_medoid         — align all spawns to the medoid of the MECI set

All MECIs are first brute-force aligned to the current reference to establish
a common atom ordering and orientation frame.  The medoid is then the MECI
that minimises the mean pairwise RMSD to all other aligned MECIs.

Metrics reported per mode:
  • RMSD distribution: mean, std, median, max, 95th percentile
  • Total coordinate variance (trace of covariance of stacked aligned coords)
  • Fraction of PCA variance explained by PC1 and PC2

Plots produced per dataset:
  • RMSD distributions (histogram + KDE) for all three modes
  • Pairwise MECI RMSD matrix (heatmap)
  • PCA scatter for all three modes
  • Cumulative PCA variance for all three modes

Usage:
    uv run python evaluate_reference.py
    uv run python evaluate_reference.py -d benzene_s0
"""

import argparse
import ast
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from tqdm import tqdm

from seamstress.alignment import (
    _search_bruteforce_elementwise,
    kabsch_align_rmsd,
)
from seamstress.geometry import read_all_geometries, read_xyz_file


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS = {
    "benzene_s0": {
        "spawns":    "data/benzene_s0/spawns",
        "reference": "data/benzene_s0/mecis/Type_2.xyz",
        "mecis":     "data/benzene_s0/mecis",
    },
    "benzene_s1": {
        "spawns":    "data/benzene_s1/spawns",
        "reference": "data/benzene_s1/mecis/Type_3.xyz",
        "mecis":     "data/benzene_s1/mecis",
    },
    "ethylene": {
        "spawns":    "data/ethylene/spawns",
        "reference": "data/ethylene/mecis/Twist.xyz",
        "mecis":     "data/ethylene/mecis",
    },
    "butadiene_s0": {
        "spawns":    "data/butadiene_s0/spawns",
        "reference": "data/butadiene_s0/mecis/type12.xyz",
        "mecis":     "data/butadiene_s0/mecis",
    },
    "butadiene_s1": {
        "spawns":    "data/butadiene_s1/spawns",
        "reference": "data/butadiene_s1/mecis/type7.xyz",
        "mecis":     "data/butadiene_s1/mecis",
    },
}

ALLOW_REFLECTION = True


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_permutations(csv_path: Path) -> dict[str, tuple[int, ...]]:
    perms = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            perms[row["filename"]] = ast.literal_eval(row["bf_perm"])
    return perms


# ---------------------------------------------------------------------------
# MECI alignment helpers
# ---------------------------------------------------------------------------

def align_mecis_to_reference(mecis, reference):
    """
    Brute-force align every MECI to the master reference.
    Returns list of (filename, aligned_coords) in reference atom ordering.
    """
    aligned = []
    for meci in tqdm(mecis, desc="Aligning MECIs to reference", leave=False):
        if meci.filename == reference.filename:
            aligned.append((meci.filename, reference.coordinates.copy()))
        else:
            _, _, _, _, coords, _, _ = _search_bruteforce_elementwise(
                reference.coordinates, meci.coordinates,
                reference.atoms, meci.atoms, ALLOW_REFLECTION,
            )
            aligned.append((meci.filename, coords))
    return aligned


def compute_meci_rmsd_matrix(aligned_mecis, ref_atoms):
    """
    Pairwise Kabsch RMSD between all aligned MECIs (atom ordering already fixed).
    Returns (names, N×N matrix).
    """
    names  = [name for name, _ in aligned_mecis]
    coords = [c    for _, c    in aligned_mecis]
    n      = len(coords)
    mat    = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            _, rmsd = kabsch_align_rmsd(
                coords[i], coords[j],
                ref_atoms, ref_atoms,
                use_all_atoms=True,
                allow_reflection=ALLOW_REFLECTION,
            )
            mat[i, j] = mat[j, i] = rmsd

    return names, mat


def find_medoid(names, rmsd_matrix):
    """Return (index, name) of the MECI with smallest mean RMSD to all others."""
    mean_rmsds = rmsd_matrix.mean(axis=1)
    idx = int(np.argmin(mean_rmsds))
    return idx, names[idx]


# ---------------------------------------------------------------------------
# Spawn alignment
# ---------------------------------------------------------------------------

def align_spawns_to_single_ref(spawns, reference, permutations):
    """Kabsch-align all spawns to one reference.  Returns (rmsds, coord_matrix)."""
    rmsds  = []
    coords = []

    for tgt in tqdm(spawns, desc="  aligning spawns", leave=False):
        perm = permutations[tgt.filename]
        reordered = tgt.coordinates[list(perm)]
        aligned, rmsd = kabsch_align_rmsd(
            reference, reordered,
            None, None,
            use_all_atoms=True,
            allow_reflection=ALLOW_REFLECTION,
        )
        rmsds.append(rmsd)
        coords.append(aligned.ravel())

    return np.array(rmsds), np.array(coords)


def align_spawns_multi_ref(spawns, aligned_mecis, permutations):
    """Align each spawn to its closest MECI.  Returns (rmsds, coord_matrix, closest_meci_names)."""
    meci_names  = [name for name, _ in aligned_mecis]
    meci_coords = [c    for _, c    in aligned_mecis]

    rmsds       = []
    coords      = []
    closest     = []

    for tgt in tqdm(spawns, desc="  aligning spawns", leave=False):
        perm = permutations[tgt.filename]
        reordered = tgt.coordinates[list(perm)]

        best_rmsd    = float("inf")
        best_aligned = None
        best_name    = None

        for name, meci_c in zip(meci_names, meci_coords):
            aligned, rmsd = kabsch_align_rmsd(
                meci_c, reordered,
                None, None,
                use_all_atoms=True,
                allow_reflection=ALLOW_REFLECTION,
            )
            if rmsd < best_rmsd:
                best_rmsd    = rmsd
                best_aligned = aligned
                best_name    = name

        rmsds.append(best_rmsd)
        coords.append(best_aligned.ravel())
        closest.append(best_name)

    return np.array(rmsds), np.array(coords), closest


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmsd_stats(rmsds: np.ndarray) -> dict:
    return {
        "mean":   float(np.mean(rmsds)),
        "std":    float(np.std(rmsds)),
        "median": float(np.median(rmsds)),
        "max":    float(np.max(rmsds)),
        "p95":    float(np.percentile(rmsds, 95)),
    }


def total_variance(coord_matrix: np.ndarray) -> float:
    """Trace of the covariance matrix of the (N, 3*n_atoms) coordinate matrix."""
    centred = coord_matrix - coord_matrix.mean(axis=0)
    cov     = centred.T @ centred / (len(coord_matrix) - 1)
    return float(np.trace(cov))


def pca_variance_ratios(coord_matrix: np.ndarray, n_components: int = 10) -> np.ndarray:
    pca = PCA(n_components=min(n_components, coord_matrix.shape[1]))
    pca.fit(coord_matrix)
    return pca.explained_variance_ratio_


def pca_embed(coord_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(coord_matrix)
    return embedding, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rmsd_distributions(modes: dict, out_path: Path, dataset_name: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"current_single_ref": "#2196F3", "multi_ref": "#F44336", "meci_medoid": "#4CAF50"}

    for label, data in modes.items():
        rmsds = data["rmsds"]
        x = np.linspace(0, rmsds.max() * 1.05, 400)
        kde = gaussian_kde(rmsds, bw_method=0.15)
        ax.fill_between(x, kde(x), alpha=0.25, color=colors[label])
        ax.plot(x, kde(x), color=colors[label], lw=2,
                label=f"{label}  (μ={data['stats']['mean']:.3f}, σ={data['stats']['std']:.3f})")

    ax.set_xlabel("RMSD (Å)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{dataset_name} — RMSD distributions by reference strategy", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_meci_rmsd_heatmap(names, rmsd_matrix, medoid_name, out_path: Path, dataset_name: str):
    n    = len(names)
    lbls = [n.replace(".xyz", "") for n in names]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.35), max(5, n * 0.3)))
    im = ax.imshow(rmsd_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="RMSD (Å)")

    ax.set_xticks(range(n)); ax.set_xticklabels(lbls, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(lbls, fontsize=7)

    medoid_idx = names.index(medoid_name)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(plt.Rectangle((medoid_idx - 0.5, -0.5), 1, n,
                                fill=False, edgecolor="red", lw=2, clip_on=False))
    ax.add_patch(plt.Rectangle((-0.5, medoid_idx - 0.5), n, 1,
                                fill=False, edgecolor="red", lw=2, clip_on=False))

    ax.set_title(f"{dataset_name} — pairwise MECI RMSD matrix  (medoid = {medoid_name.replace('.xyz','')})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_pca_scatter(modes: dict, out_path: Path, dataset_name: str):
    colors = {"current_single_ref": "#2196F3", "multi_ref": "#F44336", "meci_medoid": "#4CAF50"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (label, data) in zip(axes, modes.items()):
        emb, var = pca_embed(data["coords"])
        sc = ax.scatter(emb[:, 0], emb[:, 1],
                        c=data["rmsds"], cmap="plasma", s=4, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="RMSD (Å)")
        ax.set_title(f"{label}\nPC1 {var[0]*100:.1f}%  PC2 {var[1]*100:.1f}%", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.suptitle(f"{dataset_name} — PCA of aligned coordinates (coloured by RMSD)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cumulative_variance(modes: dict, out_path: Path, dataset_name: str):
    colors = {"current_single_ref": "#2196F3", "multi_ref": "#F44336", "meci_medoid": "#4CAF50"}
    fig, ax = plt.subplots(figsize=(7, 4))

    for label, data in modes.items():
        ratios = pca_variance_ratios(data["coords"], n_components=10)
        ax.plot(np.arange(1, len(ratios) + 1), np.cumsum(ratios) * 100,
                marker="o", color=colors[label], label=label, lw=2)

    ax.set_xlabel("Number of PCs", fontsize=12)
    ax.set_ylabel("Cumulative variance explained (%)", fontsize=12)
    ax.set_title(f"{dataset_name} — Cumulative PCA variance by reference strategy", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_summary_table(modes: dict, total_vars: dict):
    col_w = [20, 8, 8, 8, 8, 8, 14]
    header = ["Mode", "Mean", "Std", "Median", "Max", "P95", "Total Var"]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print()
    print(sep)
    print(fmt.format(*header))
    print(sep)
    for label, data in modes.items():
        s = data["stats"]
        print(fmt.format(
            label,
            f"{s['mean']:.4f}",
            f"{s['std']:.4f}",
            f"{s['median']:.4f}",
            f"{s['max']:.4f}",
            f"{s['p95']:.4f}",
            f"{total_vars[label]:.4f}",
        ))
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(dataset_name: str, cfg: dict):
    from collections import Counter

    plots_dir = Path(f"reports/{dataset_name}_reference_eval")
    plots_dir.mkdir(parents=True, exist_ok=True)

    spawns_dir = Path(cfg["spawns"])
    ref_path   = Path(cfg["reference"])
    mecis_dir  = Path(cfg["mecis"])
    csv_path   = Path(f"reports/{dataset_name}.csv")

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}")
    print(f"Ref     : {ref_path}")

    missing = [p for p in (spawns_dir, ref_path, mecis_dir, csv_path) if not p.exists()]
    if missing:
        for m in missing:
            print(f"  MISSING: {m}")
        print("  Skipping.")
        return None

    reference = read_xyz_file(ref_path)
    spawns    = read_all_geometries(spawns_dir)
    perms     = load_permutations(csv_path)
    mecis     = read_all_geometries(mecis_dir)
    print(f"Spawns  : {len(spawns)}  |  MECIs: {len(mecis)}")

    # --- align all MECIs to master reference ---
    print("Aligning MECIs to master reference (brute-force)...")
    aligned_mecis = align_mecis_to_reference(mecis, reference)

    # --- pairwise MECI RMSD matrix → medoid ---
    print("Computing pairwise MECI RMSD matrix...")
    meci_names, rmsd_matrix = compute_meci_rmsd_matrix(aligned_mecis, reference.atoms)
    medoid_idx, medoid_name = find_medoid(meci_names, rmsd_matrix)
    medoid_coords           = aligned_mecis[medoid_idx][1]

    mean_rmsds = rmsd_matrix.mean(axis=1)
    cur_idx    = meci_names.index(reference.filename)
    print(f"  Medoid      : {medoid_name}  (mean RMSD = {mean_rmsds[medoid_idx]:.4f} Å)")
    print(f"  Current ref : {reference.filename}  (mean RMSD = {mean_rmsds[cur_idx]:.4f} Å)")
    if medoid_name == reference.filename:
        print("  → Current reference IS the medoid.")
    else:
        print(f"  → Medoid differs from current reference!")

    # --- three alignment modes ---
    modes = {}

    print("[1/3] current_single_ref ...")
    rmsds, coords = align_spawns_to_single_ref(spawns, reference.coordinates, perms)
    modes["current_single_ref"] = {"rmsds": rmsds, "coords": coords,
                                   "stats": rmsd_stats(rmsds)}

    print("[2/3] multi_ref ...")
    rmsds, coords, closest = align_spawns_multi_ref(spawns, aligned_mecis, perms)
    modes["multi_ref"] = {"rmsds": rmsds, "coords": coords,
                          "stats": rmsd_stats(rmsds)}
    top5 = Counter(closest).most_common(5)
    print(f"  Top-5 chosen MECIs: {top5}")

    print("[3/3] meci_medoid ...")
    rmsds, coords = align_spawns_to_single_ref(spawns, medoid_coords, perms)
    modes["meci_medoid"] = {"rmsds": rmsds, "coords": coords,
                            "stats": rmsd_stats(rmsds)}

    # --- metrics ---
    total_vars = {label: total_variance(data["coords"]) for label, data in modes.items()}
    print_summary_table(modes, total_vars)

    # --- plots ---
    print("Generating plots...")
    plot_rmsd_distributions(modes, plots_dir / "rmsd_distributions.png", dataset_name)
    plot_meci_rmsd_heatmap(meci_names, rmsd_matrix, medoid_name,
                           plots_dir / "meci_rmsd_heatmap.png", dataset_name)
    plot_pca_scatter(modes, plots_dir / "pca_scatter.png", dataset_name)
    plot_cumulative_variance(modes, plots_dir / "pca_cumulative_variance.png", dataset_name)
    print(f"  Written to {plots_dir}/")

    return {
        "dataset":      dataset_name,
        "current_ref":  reference.filename,
        "medoid":       medoid_name,
        "same":         medoid_name == reference.filename,
        "cur_mean_rmsd_to_mecis": float(mean_rmsds[cur_idx]),
        "med_mean_rmsd_to_mecis": float(mean_rmsds[medoid_idx]),
        "modes":        modes,
        "total_vars":   total_vars,
    }


def print_cross_dataset_summary(results: list):
    print(f"\n{'='*60}")
    print("CROSS-DATASET SUMMARY")
    col_w = [14, 14, 14, 6, 8, 8, 8, 8]
    header = ["Dataset", "Current ref", "Medoid", "Same?",
              "cur μ", "med μ", "cur Var", "med Var"]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    print(sep)
    print(fmt.format(*header))
    print(sep)
    for r in results:
        cur_mean = r["modes"]["current_single_ref"]["stats"]["mean"]
        med_mean = r["modes"]["meci_medoid"]["stats"]["mean"]
        cur_var  = r["total_vars"]["current_single_ref"]
        med_var  = r["total_vars"]["meci_medoid"]
        print(fmt.format(
            r["dataset"],
            r["current_ref"].replace(".xyz", ""),
            r["medoid"].replace(".xyz", ""),
            "YES" if r["same"] else "NO",
            f"{cur_mean:.4f}",
            f"{med_mean:.4f}",
            f"{cur_var:.3f}",
            f"{med_var:.3f}",
        ))
    print(sep)
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-d", "--dataset",
        choices=list(DATASETS.keys()),
        default=None,
        help="Evaluate only this dataset (default: all)",
    )
    args = parser.parse_args()

    datasets_to_run = (
        {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    )

    all_results = []
    for dataset_name, cfg in datasets_to_run.items():
        result = run_dataset(dataset_name, cfg)
        if result is not None:
            all_results.append(result)

    if len(all_results) > 1:
        print_cross_dataset_summary(all_results)


if __name__ == "__main__":
    main()
