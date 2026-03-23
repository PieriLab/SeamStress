import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from seamstress.analysis import coords_to_flat_cartesian,coords_to_inverse_distance_matrix,coords_to_inverse_eigenvalues,coords_to_soap,coords_to_mbtr
# coords_to_flat_cartesian, coords_to_inverse_distance_matrix, coords_to_inverse_eigenvalues
# coords_to_soap, coords_to_mbtr

STRATEGY_COLORS = {
    "cartesian": "#2196F3",
    "inv_dist": "#FF5722",
    "inv_eig": "#4CAF50",
    "soap": "#FFC107",
    "mbtr": "#9C27B0",
}
STRATEGY_LABELS = {
    "cartesian": "Flattened Cartesian",
    "inv_dist": "Inverse Distance Matrix",
    "inv_eig": "Inverse-Distance Eigenvalues",
    "soap": "SOAP",
    "mbtr": "MBTR",
}

# ── Load geometries ──────────────────────────────
def load_geometries(aligned_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    from seamstress.geometry import read_xyz_file
    paths = sorted(aligned_dir.glob("*.xyz"))
    coords_list = []
    atoms_list = []
    names = []
    for p in paths:
        geom = read_xyz_file(p)
        coords_list.append(geom.coordinates)
        atoms_list.append(geom.atoms)
        names.append(p.stem)
    return np.array(coords_list), atoms_list[0], names  # assume same atoms per geometry

# ── Feature-based distance matrices ───────────────────────
def compute_feature_distances(coords: np.ndarray, atoms: list[str], features: list[str]) -> dict[str, np.ndarray]:
    matrices = {}
    for feat in features:
        if feat == "cartesian":
            X = coords_to_flat_cartesian(coords)
        elif feat == "inv_dist":
            X = coords_to_inverse_distance_matrix(coords)
        elif feat == "inv_eig":
            X = coords_to_inverse_eigenvalues(coords, atoms=atoms)
        elif feat == "soap":
            X = coords_to_soap(coords, atoms=atoms)
        elif feat == "mbtr":
            X = coords_to_mbtr(coords, atoms=atoms)
        else:
            raise ValueError(f"Unknown feature type: {feat}")
        D = squareform(pdist(X, metric="euclidean"))
        matrices[feat] = D
    return matrices

# ── Comparison metrics ───────────────────────────────
def knn_overlap(D1: np.ndarray, D2: np.ndarray, k: int) -> float:
    n = D1.shape[0]
    k = min(k, n - 1)
    nn1 = np.argsort(D1, axis=1)[:, 1: k + 1]
    nn2 = np.argsort(D2, axis=1)[:, 1: k + 1]
    return float(np.mean([len(set(nn1[i]) & set(nn2[i])) / k for i in range(n)]))

def compare_matrices(D_opt: np.ndarray, D_feat: np.ndarray, label: str) -> dict:
    n = D_opt.shape[0]
    triu = np.triu_indices(n, k=1)
    d_opt, d_f = D_opt[triu], D_feat[triu]
    spearman = spearmanr(d_opt, d_f).statistic
    pearson = pearsonr(d_opt, d_f).statistic
    frob_abs = float(np.linalg.norm(D_opt - D_feat, ord="fro"))
    frob_rel = frob_abs / float(np.linalg.norm(D_opt, ord="fro"))
    mse = float(np.mean((d_opt - d_f) ** 2))
    k_vals = {
        "k5pct": max(1, int(0.05 * n)),
        "k10pct": max(1, int(0.10 * n)),
        "k20pct": max(1, int(0.20 * n)),
    }
    knn = {tag: knn_overlap(D_opt, D_feat, k) for tag, k in k_vals.items()}
    return {
        "strategy": label,
        "n_sample": n,
        "spearman": round(spearman, 6),
        "pearson": round(pearson, 6),
        "frobenius_abs": round(frob_abs, 6),
        "frobenius_rel": round(frob_rel, 6),
        "mse": round(mse, 8),
        **{f"knn_overlap_{tag}": round(v, 6) for tag, v in knn.items()},
        **{f"k_{tag}": k_vals[tag] for tag in k_vals},
    }

# ── Plots ───────────────────────────────────────────────
def scatter_plot(D_opt, D_feat, feat_name, out_path: Path, metrics: dict):
    triu = np.triu_indices(D_opt.shape[0], k=1)
    x, y = D_opt[triu], D_feat[triu]
    color = STRATEGY_COLORS[feat_name]
    label = STRATEGY_LABELS[feat_name]
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x, y, s=6, alpha=0.4, color=color, linewidths=0)
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="Identity")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r"$D^{\mathrm{opt}}_{ij}$ (Å)")
    ax.set_ylabel(r"$D^{(\mathrm{feat})}_{ij}$ (Å)")
    ax.set_title(f"{label}\nSpearman={metrics['spearman']:.4f} Frob={metrics['frobenius_rel']:.2%}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def knn_bar_plot(metrics_list: list[dict], out_path: Path):
    k_tags = ["k5pct","k10pct","k20pct"]
    k_labels = ["k = 5%","k = 10%","k = 20%"]
    x = np.arange(len(k_tags))
    width = 0.15
    fig, ax = plt.subplots(figsize=(7,4))
    for i, m in enumerate(metrics_list):
        vals = [m[f"knn_overlap_{t}"] for t in k_tags]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=STRATEGY_LABELS[m["strategy"]], color=STRATEGY_COLORS[m["strategy"]], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(k_labels)
    ax.set_ylim(0,1)
    ax.set_ylabel("k-NN overlap")
    ax.set_title("Local neighbourhood preservation vs $D^{\\mathrm{opt}}$")
    ax.legend(fontsize=9)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ── Runner ──────────────────────────────────────────────
def run_feature_comparison(aligned_dir: Path, dopt_csv: Path, features: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all geometries in aligned folder
    coords, atoms, names = load_geometries(aligned_dir)

    # Load D_opt and intersect with available geometries
    df_dopt = pd.read_csv(dopt_csv, index_col=0)
    common_names = [n for n in names if n in df_dopt.index]
    if len(common_names) < len(names):
        print(f"Warning: D_opt only has {len(common_names)} of {len(names)} geometries — using subset")
    # subset coords and names
    idx_map = [names.index(n) for n in common_names]
    coords = coords[idx_map]
    names = common_names
    D_opt = df_dopt.loc[names, names].values

    # Compute feature distance matrices
    matrices = compute_feature_distances(coords, atoms, features)
    rows = []
    for feat, D_feat in matrices.items():
        metrics = compare_matrices(D_opt, D_feat, label=feat)
        rows.append(metrics)
        print(f"[{feat}] Spearman={metrics['spearman']:.4f} Frobenius={metrics['frobenius_rel']:.2%}")
        # Scatter plot
        scatter_plot(D_opt, D_feat, feat, output_dir / f"scatter_{feat}.png", metrics)

    # k-NN overlap bar plot
    knn_bar_plot(rows, output_dir / "knn_overlap.png")

    # Save metrics
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "feature_metrics.csv", index=False)
    print(f"Feature comparison saved → {output_dir / 'feature_metrics.csv'}")
    return df

# ── CLI ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Feature-based comparison to D_opt")
    parser.add_argument("--aligned-dir", type=Path, required=True, help="Path to aligned XYZ geometries")
    parser.add_argument("--dopt-csv", type=Path, required=True, help="Path to D_opt CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("feature_evaluation"))
    parser.add_argument("--features", nargs="+", default=["cartesian","inv_dist","inv_eig","soap","mbtr"])
    args = parser.parse_args()
    
    run_feature_comparison(args.aligned_dir, args.dopt_csv, args.features, args.output_dir)

if __name__ == "__main__":
    main()