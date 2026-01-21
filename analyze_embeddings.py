#!/usr/bin/env python3
"""
Comprehensive dimensionality reduction analysis with PCA, UMAP, and Diffusion Maps.

Generates comparison plots for all three methods with centroid markers.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from umap import UMAP

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
    """Load centroid structures from xyz files."""
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
    """Load all aligned geometries from a family directory."""
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

        rmsd_match = re.search(r"RMSD:\s*([\d.]+)", header)
        rmsd = float(rmsd_match.group(1)) if rmsd_match else 0.0

        if not smiles:
            smiles_match = re.search(r"Family_\d+\s+(\S+)", header)
            if smiles_match:
                smiles = smiles_match.group(1)

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
    """Get display name for a family."""
    if smiles in SMILES_TO_NAME:
        return SMILES_TO_NAME[smiles]
    return smiles if smiles else family_name


def coords_to_features(coords: np.ndarray) -> np.ndarray:
    """Convert coordinate array to feature matrix."""
    n_molecules = coords.shape[0]
    return coords.reshape(n_molecules, -1)


def run_pca(features: np.ndarray, scaler: StandardScaler) -> tuple[np.ndarray, PCA]:
    """Run PCA on scaled features."""
    features_scaled = scaler.transform(features)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(features_scaled)
    return coords, pca


def run_umap(features: np.ndarray, scaler: StandardScaler, n_neighbors: int = 15) -> np.ndarray:
    """Run UMAP on scaled features."""
    features_scaled = scaler.transform(features)
    n_neighbors = min(n_neighbors, features.shape[0] - 1)
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
    return reducer.fit_transform(features_scaled)


def run_tsne(features: np.ndarray, scaler: StandardScaler, perplexity: float = 30.0) -> np.ndarray:
    """Run t-SNE on scaled features."""
    features_scaled = scaler.transform(features)
    perp = min(perplexity, (features.shape[0] - 1) / 3)
    reducer = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    return reducer.fit_transform(features_scaled)


def run_diffusion_map(features: np.ndarray, scaler: StandardScaler, n_neighbors: int = 10) -> np.ndarray:
    """Run Diffusion Map (via Spectral Embedding) on scaled features."""
    features_scaled = scaler.transform(features)
    n_neighbors = min(n_neighbors, features.shape[0] - 1)
    embedding = SpectralEmbedding(
        n_components=2,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        random_state=42,
    )
    return embedding.fit_transform(features_scaled)


def analyze_family(
    family_dir: Path,
    output_dir: Path,
    family_name: str,
    centroids: dict[str, np.ndarray] | None = None,
) -> dict | None:
    """Run PCA, UMAP, and Diffusion Map analysis on a single family."""
    coords, filenames, rmsds, smiles = load_family_geometries(family_dir)
    n_molecules = len(filenames)
    display_name = get_display_name(smiles, family_name)

    print(f"\n{'='*60}")
    print(f"Analyzing {display_name}")
    print(f"{'='*60}")

    if n_molecules < 10:
        print(f"  Skipping: only {n_molecules} molecules (need >= 10)")
        return None

    print(f"  Loaded {n_molecules} molecules")

    # Check for centroid
    centroid_coords = None
    if centroids and smiles in centroids:
        centroid_coords = centroids[smiles]
        print(f"  Found centroid")

    # Convert to features
    features = coords_to_features(coords)

    # Add centroid to features for embedding
    if centroid_coords is not None:
        centroid_features = centroid_coords.reshape(1, -1)
        features_with_centroid = np.vstack([features, centroid_features])
    else:
        features_with_centroid = features

    # Fit scaler on data (without centroid)
    scaler = StandardScaler()
    scaler.fit(features)

    # Run embeddings on data + centroid
    print("  Running PCA...")
    pca_coords_all, pca = run_pca(features_with_centroid, scaler)

    print("  Running t-SNE...")
    tsne_coords_all = run_tsne(features_with_centroid, scaler)

    print("  Running UMAP...")
    umap_coords_all = run_umap(features_with_centroid, scaler)

    print("  Running Diffusion Map...")
    dm_coords_all = run_diffusion_map(features_with_centroid, scaler)

    # Separate centroid from data
    if centroid_coords is not None:
        pca_coords, pca_centroid = pca_coords_all[:-1], pca_coords_all[-1]
        tsne_coords, tsne_centroid = tsne_coords_all[:-1], tsne_coords_all[-1]
        umap_coords, umap_centroid = umap_coords_all[:-1], umap_coords_all[-1]
        dm_coords, dm_centroid = dm_coords_all[:-1], dm_coords_all[-1]
    else:
        pca_coords, pca_centroid = pca_coords_all, None
        tsne_coords, tsne_centroid = tsne_coords_all, None
        umap_coords, umap_centroid = umap_coords_all, None
        dm_coords, dm_centroid = dm_coords_all, None

    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{display_name} ({n_molecules} conformations)", fontsize=14)

    # PCA
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(pca_coords[:, 0], pca_coords[:, 1], c=rmsds, cmap='viridis', alpha=0.7, s=20)
    if pca_centroid is not None:
        ax1.scatter(pca_centroid[0], pca_centroid[1], marker='*', s=400, c='red',
                   edgecolors='black', linewidths=1.5, zorder=10, label='Centroid')
        ax1.legend(loc='upper right')
    plt.colorbar(scatter1, ax=ax1, label="RMSD")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("PCA")

    # t-SNE
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=rmsds, cmap='viridis', alpha=0.7, s=20)
    if tsne_centroid is not None:
        ax2.scatter(tsne_centroid[0], tsne_centroid[1], marker='*', s=400, c='red',
                   edgecolors='black', linewidths=1.5, zorder=10, label='Centroid')
        ax2.legend(loc='upper right')
    plt.colorbar(scatter2, ax=ax2, label="RMSD")
    ax2.set_xlabel("t-SNE1")
    ax2.set_ylabel("t-SNE2")
    ax2.set_title("t-SNE")

    # UMAP
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(umap_coords[:, 0], umap_coords[:, 1], c=rmsds, cmap='viridis', alpha=0.7, s=20)
    if umap_centroid is not None:
        ax3.scatter(umap_centroid[0], umap_centroid[1], marker='*', s=400, c='red',
                   edgecolors='black', linewidths=1.5, zorder=10, label='Centroid')
        ax3.legend(loc='upper right')
    plt.colorbar(scatter3, ax=ax3, label="RMSD")
    ax3.set_xlabel("UMAP1")
    ax3.set_ylabel("UMAP2")
    ax3.set_title("UMAP")

    # Diffusion Map
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(dm_coords[:, 0], dm_coords[:, 1], c=rmsds, cmap='viridis', alpha=0.7, s=20)
    if dm_centroid is not None:
        ax4.scatter(dm_centroid[0], dm_centroid[1], marker='*', s=400, c='red',
                   edgecolors='black', linewidths=1.5, zorder=10, label='Centroid')
        ax4.legend(loc='upper right')
    plt.colorbar(scatter4, ax=ax4, label="RMSD")
    ax4.set_xlabel("DM1")
    ax4.set_ylabel("DM2")
    ax4.set_title("Diffusion Map")

    plt.tight_layout()
    plot_path = output_dir / f"{family_name}_embeddings.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    return {"family": family_name, "display_name": display_name, "n_molecules": n_molecules}


def analyze_combined(
    aligned_dir: Path,
    output_dir: Path,
    centroids: dict[str, np.ndarray] | None = None,
) -> dict | None:
    """Run combined analysis on all families."""
    print(f"\n{'='*60}")
    print("Combined Analysis (All Families)")
    print(f"{'='*60}")

    all_coords = []
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
        all_rmsds.extend(rmsds)
        all_families.extend([family_dir.name] * len(filenames))
        all_display_names.extend([display_name] * len(filenames))
        all_smiles.extend([smiles] * len(filenames))
        family_indices.extend([i] * len(filenames))

    if not all_coords:
        print("  No data found!")
        return None

    coords = np.vstack(all_coords)
    n_molecules = len(all_rmsds)
    print(f"  Loaded {n_molecules} molecules")

    features = coords_to_features(coords)

    # Collect centroids for this data
    centroid_info = []
    if centroids:
        unique_families = sorted(set(all_families))
        for smiles, centroid_coords in centroids.items():
            for idx, fam in enumerate(unique_families):
                fam_idx = all_families.index(fam)
                if all_smiles[fam_idx] == smiles:
                    centroid_features = centroid_coords.reshape(1, -1)
                    centroid_info.append((smiles, idx, centroid_features))
                    break

    # Combine features with centroids
    if centroid_info:
        centroid_features_all = np.vstack([c[2] for c in centroid_info])
        features_with_centroids = np.vstack([features, centroid_features_all])
        n_centroids = len(centroid_info)
        print(f"  Added {n_centroids} centroids")
    else:
        features_with_centroids = features
        n_centroids = 0

    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(features)

    # Run embeddings
    print("  Running PCA...")
    pca_all, _ = run_pca(features_with_centroids, scaler)

    print("  Running t-SNE...")
    tsne_all = run_tsne(features_with_centroids, scaler)

    print("  Running UMAP...")
    umap_all = run_umap(features_with_centroids, scaler)

    print("  Running Diffusion Map...")
    dm_all = run_diffusion_map(features_with_centroids, scaler)

    # Separate data and centroids
    pca_coords = pca_all[:n_molecules]
    tsne_coords = tsne_all[:n_molecules]
    umap_coords = umap_all[:n_molecules]
    dm_coords = dm_all[:n_molecules]

    centroid_pca = [(centroid_info[i][1], pca_all[n_molecules + i]) for i in range(n_centroids)]
    centroid_tsne = [(centroid_info[i][1], tsne_all[n_molecules + i]) for i in range(n_centroids)]
    centroid_umap = [(centroid_info[i][1], umap_all[n_molecules + i]) for i in range(n_centroids)]
    centroid_dm = [(centroid_info[i][1], dm_all[n_molecules + i]) for i in range(n_centroids)]

    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Combined Analysis ({n_molecules} conformations)", fontsize=14)

    # Unique families for legend
    unique_families = sorted(set(all_families))
    unique_display = []
    for fam in unique_families:
        idx = all_families.index(fam)
        unique_display.append(all_display_names[idx])

    # PCA
    ax1 = axes[0, 0]
    ax1.scatter(pca_coords[:, 0], pca_coords[:, 1], c=family_indices, cmap='tab10', alpha=0.5, s=10)
    for family_idx, pt in centroid_pca:
        color = plt.cm.tab10(family_idx / 10)
        ax1.scatter(pt[0], pt[1], marker='*', s=500, c=[color], edgecolors='black', linewidths=1.5, zorder=10)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("PCA")

    # t-SNE
    ax2 = axes[0, 1]
    ax2.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=family_indices, cmap='tab10', alpha=0.5, s=10)
    for family_idx, pt in centroid_tsne:
        color = plt.cm.tab10(family_idx / 10)
        ax2.scatter(pt[0], pt[1], marker='*', s=500, c=[color], edgecolors='black', linewidths=1.5, zorder=10)
    ax2.set_xlabel("t-SNE1")
    ax2.set_ylabel("t-SNE2")
    ax2.set_title("t-SNE")

    # UMAP
    ax3 = axes[1, 0]
    ax3.scatter(umap_coords[:, 0], umap_coords[:, 1], c=family_indices, cmap='tab10', alpha=0.5, s=10)
    for family_idx, pt in centroid_umap:
        color = plt.cm.tab10(family_idx / 10)
        ax3.scatter(pt[0], pt[1], marker='*', s=500, c=[color], edgecolors='black', linewidths=1.5, zorder=10)
    ax3.set_xlabel("UMAP1")
    ax3.set_ylabel("UMAP2")
    ax3.set_title("UMAP")

    # Diffusion Map
    ax4 = axes[1, 1]
    ax4.scatter(dm_coords[:, 0], dm_coords[:, 1], c=family_indices, cmap='tab10', alpha=0.5, s=10)
    for family_idx, pt in centroid_dm:
        color = plt.cm.tab10(family_idx / 10)
        ax4.scatter(pt[0], pt[1], marker='*', s=500, c=[color], edgecolors='black', linewidths=1.5, zorder=10)
    ax4.set_xlabel("DM1")
    ax4.set_ylabel("DM2")
    ax4.set_title("Diffusion Map")

    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i/10),
               markersize=8, label=d) for i, d in enumerate(unique_display)]
    handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                   markeredgecolor='black', markersize=12, label='Centroid'))
    fig.legend(handles=handles, loc='upper right', fontsize=7, bbox_to_anchor=(0.99, 0.95))

    plt.tight_layout()
    plot_path = output_dir / "combined_embeddings.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

    return {"n_molecules": n_molecules}


def main():
    aligned_dir = Path("aligned_output")
    output_dir = Path("analysis_output")
    centroids_dir = Path("centroids")
    output_dir.mkdir(exist_ok=True)

    print("Embedding Analysis: PCA + UMAP + Diffusion Map")
    print("=" * 60)

    centroids = load_centroids(centroids_dir)
    if centroids:
        print(f"Loaded {len(centroids)} centroid structures")

    # Analyze main families (1, 2, 3 have centroids)
    for family_name in ["family_1", "family_2", "family_3"]:
        family_dir = aligned_dir / family_name
        if family_dir.exists():
            analyze_family(family_dir, output_dir, family_name, centroids)

    # Combined analysis
    analyze_combined(aligned_dir, output_dir, centroids)

    print("\n" + "=" * 60)
    print("Done! Plots saved to analysis_output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
