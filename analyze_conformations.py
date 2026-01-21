#!/usr/bin/env python3
"""
Dimensionality reduction and clustering analysis of aligned molecular conformations.

Uses PCA for variance analysis and t-SNE for 2D visualization.
Uses K-means for clustering.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Map SMILES to human-readable centroid names
SMILES_TO_NAME = {
    "[H]C([H])=C([H])[H]": "Twist (ethylene)",
    "[H][C]C([H])([H])[H]": "Ethylidene",
    "[H+].[H][C-]=C([H])[H]": "H dissociation",
}

# Map centroid filenames to SMILES
CENTROID_TO_SMILES = {
    "twist": "[H]C([H])=C([H])[H]",
    "Ethylidene": "[H][C]C([H])([H])[H]",
    "H_dissociation": "[H+].[H][C-]=C([H])[H]",
}


def load_centroids(centroids_dir: Path) -> dict[str, np.ndarray]:
    """
    Load centroid structures from xyz files.

    Returns:
        Dictionary mapping SMILES to centroid coordinates (n_atoms, 3)
    """
    centroids = {}

    if not centroids_dir.exists():
        return centroids

    for xyz_file in centroids_dir.glob("*.xyz"):
        name = xyz_file.stem
        if name not in CENTROID_TO_SMILES:
            continue

        smiles = CENTROID_TO_SMILES[name]

        with open(xyz_file) as f:
            lines = f.readlines()

        n_atoms = int(lines[0].strip())
        coords = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            coords.append([x, y, z])

        centroids[smiles] = np.array(coords)

    return centroids


def load_family_geometries(family_dir: Path) -> tuple[np.ndarray, list[str], list[float], str]:
    """
    Load all aligned geometries from a family directory.

    Returns:
        coords: Array of shape (n_molecules, n_atoms, 3)
        filenames: List of source filenames
        rmsds: List of RMSD values from headers
        smiles: SMILES string for this family
    """
    coords_list = []
    filenames = []
    rmsds = []
    smiles = ""

    xyz_files = sorted(family_dir.glob("*.xyz"))

    for xyz_file in xyz_files:
        with open(xyz_file) as f:
            lines = f.readlines()

        n_atoms = int(lines[0].strip())
        header = lines[1].strip()

        # Extract RMSD from header
        rmsd_match = re.search(r"RMSD:\s*([\d.]+)", header)
        rmsd = float(rmsd_match.group(1)) if rmsd_match else 0.0

        # Extract SMILES from header (format: "Family_N SMILES | RMSD: ...")
        if not smiles:
            smiles_match = re.search(r"Family_\d+\s+(\S+)", header)
            if smiles_match:
                smiles = smiles_match.group(1)

        # Parse coordinates
        atoms_coords = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms_coords.append([x, y, z])

        coords_list.append(atoms_coords)
        filenames.append(xyz_file.stem)
        rmsds.append(rmsd)

    return np.array(coords_list), filenames, rmsds, smiles


def get_display_name(smiles: str, family_name: str) -> str:
    """Get display name for a family - centroid name if known, else SMILES."""
    if smiles in SMILES_TO_NAME:
        return SMILES_TO_NAME[smiles]
    return smiles if smiles else family_name


def coords_to_features(coords: np.ndarray) -> np.ndarray:
    """
    Convert coordinate array to feature matrix.

    Args:
        coords: Shape (n_molecules, n_atoms, 3)

    Returns:
        features: Shape (n_molecules, n_atoms * 3)
    """
    n_molecules = coords.shape[0]
    return coords.reshape(n_molecules, -1)


def run_pca_analysis(features: np.ndarray, n_components: int = 10) -> dict:
    """
    Run PCA and return results.
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Run PCA
    n_comp = min(n_components, features.shape[1], features.shape[0])
    pca = PCA(n_components=n_comp)
    pca_coords = pca.fit_transform(features_scaled)

    return {
        "coords": pca_coords,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "components": pca.components_,
        "pca": pca,
        "scaler": scaler,
    }


def run_tsne_analysis(
    features: np.ndarray,
    perplexity: float = 30.0,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Run t-SNE dimensionality reduction.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Adjust perplexity for small datasets
    perp = min(perplexity, (features.shape[0] - 1) / 3)

    reducer = TSNE(
        n_components=n_components,
        perplexity=perp,
        random_state=random_state,
        max_iter=1000,
    )

    return reducer.fit_transform(features_scaled)


def run_kmeans_clustering(
    features: np.ndarray,
    n_clusters: int = 5,
) -> np.ndarray:
    """
    Run K-means clustering.

    Returns:
        labels: Cluster labels
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    clusterer = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    return clusterer.fit_predict(features_scaled)


def analyze_family(
    family_dir: Path,
    output_dir: Path,
    family_name: str,
    centroids: dict[str, np.ndarray] | None = None,
) -> dict:
    """
    Run full analysis on a single family.
    """
    # Load data
    coords, filenames, rmsds, smiles = load_family_geometries(family_dir)
    n_molecules = len(filenames)

    display_name = get_display_name(smiles, family_name)

    print(f"\n{'='*60}")
    print(f"Analyzing {display_name}")
    print(f"{'='*60}")

    if n_molecules < 5:
        print(f"  Skipping: only {n_molecules} molecules (need >= 5)")
        return None

    print(f"  Loaded {n_molecules} molecules")

    # Check if we have a centroid for this family
    centroid_coords = None
    if centroids and smiles in centroids:
        centroid_coords = centroids[smiles]
        print(f"  Found centroid for {display_name}")

    # Convert to features
    features = coords_to_features(coords)
    print(f"  Feature dimension: {features.shape[1]}")

    # If we have a centroid, add it to features for t-SNE (will be last row)
    if centroid_coords is not None:
        centroid_features = centroid_coords.reshape(1, -1)
        features_with_centroid = np.vstack([features, centroid_features])
    else:
        features_with_centroid = features

    # PCA analysis (on data without centroid to avoid biasing)
    print("  Running PCA...")
    pca_results = run_pca_analysis(features)

    # Project centroid into PCA space if available
    centroid_pca = None
    if centroid_coords is not None:
        centroid_scaled = pca_results['scaler'].transform(centroid_features)
        centroid_pca = pca_results['pca'].transform(centroid_scaled)

    print(f"  Variance explained by first 3 PCs: {pca_results['cumulative_variance'][2]*100:.1f}%")
    print(f"  Variance explained by first 5 PCs: {pca_results['cumulative_variance'][min(4, len(pca_results['cumulative_variance'])-1)]*100:.1f}%")

    # t-SNE analysis (include centroid so it gets embedded)
    print("  Running t-SNE...")
    tsne_all = run_tsne_analysis(features_with_centroid)

    # Separate centroid from data points
    if centroid_coords is not None:
        tsne_coords = tsne_all[:-1]
        centroid_tsne = tsne_all[-1:]
    else:
        tsne_coords = tsne_all
        centroid_tsne = None

    # K-means clustering (on t-SNE coordinates for better results)
    print("  Running K-means...")
    # Estimate number of clusters: sqrt(n/2) is a common heuristic
    n_clusters_est = max(3, min(10, int(np.sqrt(n_molecules / 2))))
    cluster_labels = run_kmeans_clustering(
        tsne_coords,
        n_clusters=n_clusters_est,
    )

    n_clusters = len(set(cluster_labels))
    n_noise = 0  # K-means doesn't have noise points
    print(f"  Found {n_clusters} clusters")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{display_name} ({n_molecules} conformations)", fontsize=14)

    # Plot 1: PCA variance explained
    ax1 = axes[0, 0]
    n_show = min(10, len(pca_results['explained_variance_ratio']))
    ax1.bar(range(1, n_show + 1), pca_results['explained_variance_ratio'][:n_show] * 100)
    ax1.plot(range(1, n_show + 1), pca_results['cumulative_variance'][:n_show] * 100, 'ro-')
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("PCA Variance Explained")
    ax1.legend(["Cumulative", "Individual"])

    # Plot 2: PCA projection (PC1 vs PC2) colored by RMSD
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(
        pca_results['coords'][:, 0],
        pca_results['coords'][:, 1],
        c=rmsds,
        cmap='viridis',
        alpha=0.7,
        s=20,
    )
    # Plot centroid as star
    if centroid_pca is not None:
        ax2.scatter(
            centroid_pca[0, 0],
            centroid_pca[0, 1],
            marker='*',
            s=400,
            c='red',
            edgecolors='black',
            linewidths=1,
            zorder=10,
            label='Centroid',
        )
        ax2.legend(loc='upper right')
    plt.colorbar(scatter2, ax=ax2, label="RMSD")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("PCA Projection (colored by RMSD)")

    # Plot 3: t-SNE colored by RMSD
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(
        tsne_coords[:, 0],
        tsne_coords[:, 1],
        c=rmsds,
        cmap='viridis',
        alpha=0.7,
        s=20,
    )
    # Plot centroid as star
    if centroid_tsne is not None:
        ax3.scatter(
            centroid_tsne[0, 0],
            centroid_tsne[0, 1],
            marker='*',
            s=400,
            c='red',
            edgecolors='black',
            linewidths=1,
            zorder=10,
            label='Centroid',
        )
        ax3.legend(loc='upper right')
    plt.colorbar(scatter3, ax=ax3, label="RMSD")
    ax3.set_xlabel("t-SNE1")
    ax3.set_ylabel("t-SNE2")
    ax3.set_title("t-SNE Projection (colored by RMSD)")

    # Plot 4: t-SNE colored by cluster
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(
        tsne_coords[:, 0],
        tsne_coords[:, 1],
        c=cluster_labels,
        cmap='tab20',
        alpha=0.7,
        s=20,
    )
    # Plot centroid as star
    if centroid_tsne is not None:
        ax4.scatter(
            centroid_tsne[0, 0],
            centroid_tsne[0, 1],
            marker='*',
            s=400,
            c='red',
            edgecolors='black',
            linewidths=1,
            zorder=10,
            label='Centroid',
        )
        ax4.legend(loc='upper right')
    ax4.set_xlabel("t-SNE1")
    ax4.set_ylabel("t-SNE2")
    ax4.set_title(f"t-SNE Projection ({n_clusters} clusters)")

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"{family_name}_analysis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {plot_path}")

    return {
        "family": family_name,
        "display_name": display_name,
        "smiles": smiles,
        "n_molecules": n_molecules,
        "pca_variance": pca_results['cumulative_variance'],
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "tsne_coords": tsne_coords,
        "cluster_labels": cluster_labels,
        "rmsds": rmsds,
        "filenames": filenames,
    }


def analyze_combined(
    aligned_dir: Path,
    output_dir: Path,
    centroids: dict[str, np.ndarray] | None = None,
) -> dict:
    """
    Run analysis on all families combined.
    """
    print(f"\n{'='*60}")
    print("Combined Analysis (All Families)")
    print(f"{'='*60}")

    all_coords = []
    all_filenames = []
    all_rmsds = []
    all_families = []
    all_display_names = []
    all_smiles = []
    family_indices = []

    family_dirs = sorted(aligned_dir.glob("family_*"))

    for i, family_dir in enumerate(family_dirs):
        if not family_dir.is_dir():
            continue

        coords, filenames, rmsds, smiles = load_family_geometries(family_dir)

        if len(filenames) == 0:
            continue

        display_name = get_display_name(smiles, family_dir.name)

        all_coords.append(coords)
        all_filenames.extend(filenames)
        all_rmsds.extend(rmsds)
        all_families.extend([family_dir.name] * len(filenames))
        all_display_names.extend([display_name] * len(filenames))
        all_smiles.extend([smiles] * len(filenames))
        family_indices.extend([i] * len(filenames))

    if not all_coords:
        print("  No data found!")
        return None

    coords = np.vstack(all_coords)
    n_molecules = len(all_filenames)
    print(f"  Loaded {n_molecules} molecules from {len(family_dirs)} families")

    # Convert to features
    features = coords_to_features(coords)

    # Add centroids to feature set for t-SNE embedding
    centroid_info = []  # List of (smiles, family_idx, features)
    if centroids:
        unique_families = sorted(set(all_families))
        for smiles, centroid_coords in centroids.items():
            # Find which family this centroid belongs to
            for idx, fam in enumerate(unique_families):
                fam_idx = all_families.index(fam)
                if all_smiles[fam_idx] == smiles:
                    centroid_features = centroid_coords.reshape(1, -1)
                    centroid_info.append((smiles, idx, centroid_features))
                    break

    # Combine features with centroids for t-SNE
    if centroid_info:
        centroid_features_all = np.vstack([c[2] for c in centroid_info])
        features_with_centroids = np.vstack([features, centroid_features_all])
        n_centroids = len(centroid_info)
        print(f"  Added {n_centroids} centroids for embedding")
    else:
        features_with_centroids = features
        n_centroids = 0

    # PCA
    print("  Running PCA...")
    pca_results = run_pca_analysis(features)

    # Project centroids into PCA space
    centroid_pca_coords = []
    if centroid_info:
        for smiles, family_idx, cent_feat in centroid_info:
            cent_scaled = pca_results['scaler'].transform(cent_feat)
            cent_pca = pca_results['pca'].transform(cent_scaled)
            centroid_pca_coords.append((family_idx, cent_pca[0]))

    # t-SNE (with centroids included)
    print("  Running t-SNE...")
    tsne_all = run_tsne_analysis(features_with_centroids)

    # Separate data and centroid coordinates
    tsne_coords = tsne_all[:n_molecules]
    centroid_tsne_coords = []
    if n_centroids > 0:
        for i, (smiles, family_idx, _) in enumerate(centroid_info):
            centroid_tsne_coords.append((family_idx, tsne_all[n_molecules + i]))

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Combined Analysis ({n_molecules} conformations)", fontsize=14)

    # t-SNE colored by family
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        tsne_coords[:, 0],
        tsne_coords[:, 1],
        c=family_indices,
        cmap='tab10',
        alpha=0.5,
        s=10,
    )

    # Plot centroids as stars with matching family colors
    for family_idx, tsne_pt in centroid_tsne_coords:
        color = plt.cm.tab10(family_idx / 10)
        ax1.scatter(
            tsne_pt[0],
            tsne_pt[1],
            marker='*',
            s=500,
            c=[color],
            edgecolors='black',
            linewidths=1.5,
            zorder=10,
        )

    ax1.set_xlabel("t-SNE1")
    ax1.set_ylabel("t-SNE2")
    ax1.set_title("t-SNE (colored by family)")

    # Add legend with display names
    # Create mapping from family_dir to display name
    unique_families = sorted(set(all_families))
    unique_display = []
    for fam in unique_families:
        idx = all_families.index(fam)
        unique_display.append(all_display_names[idx])

    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=plt.cm.tab10(i/10), markersize=8, label=d)
               for i, d in enumerate(unique_display)]
    # Add star to legend for centroids
    handles.append(plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='gray', markeredgecolor='black',
                   markersize=12, label='Centroid'))
    ax1.legend(handles=handles, loc='upper right', fontsize=7)

    # t-SNE colored by RMSD
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        tsne_coords[:, 0],
        tsne_coords[:, 1],
        c=all_rmsds,
        cmap='viridis',
        alpha=0.5,
        s=10,
        vmax=np.percentile(all_rmsds, 95),  # Cap colorbar at 95th percentile
    )

    # Plot centroids as stars on RMSD plot too
    for family_idx, tsne_pt in centroid_tsne_coords:
        color = plt.cm.tab10(family_idx / 10)
        ax2.scatter(
            tsne_pt[0],
            tsne_pt[1],
            marker='*',
            s=500,
            c=[color],
            edgecolors='black',
            linewidths=1.5,
            zorder=10,
        )

    plt.colorbar(scatter2, ax=ax2, label="RMSD")
    ax2.set_xlabel("t-SNE1")
    ax2.set_ylabel("t-SNE2")
    ax2.set_title("t-SNE (colored by RMSD)")

    plt.tight_layout()

    plot_path = output_dir / "combined_analysis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {plot_path}")

    return {
        "n_molecules": n_molecules,
        "n_families": len(family_dirs),
        "tsne_coords": tsne_coords,
        "families": all_families,
        "rmsds": all_rmsds,
    }


def main():
    aligned_dir = Path("aligned_output")
    output_dir = Path("analysis_output")
    centroids_dir = Path("centroids")
    output_dir.mkdir(exist_ok=True)

    print("Conformation Analysis: PCA + t-SNE + K-means")
    print("=" * 60)

    # Load centroids
    centroids = load_centroids(centroids_dir)
    if centroids:
        print(f"Loaded {len(centroids)} centroid structures")

    # Analyze each family
    family_results = {}
    family_dirs = sorted(aligned_dir.glob("family_*"))

    for family_dir in family_dirs:
        if not family_dir.is_dir():
            continue

        result = analyze_family(family_dir, output_dir, family_dir.name, centroids)
        if result:
            family_results[family_dir.name] = result

    # Combined analysis
    combined_result = analyze_combined(aligned_dir, output_dir, centroids)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nAnalyzed {len(family_results)} families")
    print(f"Output saved to: {output_dir}/")

    print("\nPer-family results:")
    print(f"{'Name':<25} {'N':<8} {'Var(3PC)':<10} {'Clusters':<10}")
    print("-" * 55)
    for name, res in family_results.items():
        var_3pc = res['pca_variance'][2] * 100 if len(res['pca_variance']) > 2 else 0
        disp = res.get('display_name', name)[:24]
        print(f"{disp:<25} {res['n_molecules']:<8} {var_3pc:<10.1f}% {res['n_clusters']:<10}")

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
