#!/usr/bin/env python3
"""
Complete embedding analysis with explosion filtering.

Generates:
- Combined all-families plot (most important!) with 4 methods
- Per-family 2D and 3D plots
- PCA/t-SNE/UMAP/DM movies
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
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

CENTROID_TO_SMILES = {
    "twist": "[H]C([H])=C([H])[H]",
    "Ethylidene": "[H][C]C([H])([H])[H]",
    "H_dissociation": "[H+].[H][C-]=C([H])[H]",
}

ATOM_ORDER = ['C', 'H', 'H', 'C', 'H', 'H']


def load_centroids(centroids_dir: Path) -> dict:
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
        atoms = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            atoms.append(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            coords.append([x, y, z])

        centroids[smiles] = {'coords': np.array(coords), 'atoms': atoms, 'name': name}
    return centroids


def max_pairwise_distance(coords: np.ndarray) -> float:
    """Calculate maximum distance between any two atoms."""
    return pdist(coords).max()


def load_family_geometries(
    family_dir: Path,
    max_distance_threshold: float | None = None,
) -> tuple[np.ndarray, list[str], list[float], str, int]:
    """Load aligned geometries with optional explosion filtering."""
    coords_list = []
    filenames = []
    rmsds = []
    smiles = ""
    n_filtered = 0

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

        atoms_coords = np.array(atoms_coords)

        if max_distance_threshold is not None:
            max_dist = max_pairwise_distance(atoms_coords)
            if max_dist > max_distance_threshold:
                n_filtered += 1
                continue

        coords_list.append(atoms_coords)
        filenames.append(xyz_file.stem)
        rmsds.append(rmsd)

    if len(coords_list) == 0:
        return np.array([]), filenames, rmsds, smiles, n_filtered

    return np.array(coords_list), filenames, rmsds, smiles, n_filtered


def get_display_name(smiles: str, family_name: str) -> str:
    if smiles in SMILES_TO_NAME:
        return SMILES_TO_NAME[smiles]
    return smiles if smiles else family_name


def coords_to_features(coords: np.ndarray) -> np.ndarray:
    return coords.reshape(coords.shape[0], -1)


# Embedding functions
def run_pca(features: np.ndarray, scaler: StandardScaler, n_comp: int = 2):
    features_scaled = scaler.transform(features)
    pca = PCA(n_components=min(n_comp, features.shape[1], features.shape[0]))
    coords = pca.fit_transform(features_scaled)
    return coords, pca


def run_tsne(features: np.ndarray, scaler: StandardScaler, n_comp: int = 2):
    features_scaled = scaler.transform(features)
    perp = min(30.0, (features.shape[0] - 1) / 3)
    reducer = TSNE(n_components=n_comp, perplexity=perp, random_state=42, max_iter=1000)
    return reducer.fit_transform(features_scaled)


def run_umap(features: np.ndarray, scaler: StandardScaler, n_comp: int = 2):
    features_scaled = scaler.transform(features)
    n_neighbors = min(15, features.shape[0] - 1)
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=n_comp, random_state=42)
    return reducer.fit_transform(features_scaled)


def run_diffusion_map(features: np.ndarray, scaler: StandardScaler, n_comp: int = 2):
    features_scaled = scaler.transform(features)
    n_neighbors = min(10, features.shape[0] - 1)
    embedding = SpectralEmbedding(
        n_components=n_comp,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        random_state=42,
    )
    return embedding.fit_transform(features_scaled)


def create_pca_movie(pca, scaler, centroid_coords, atoms, output_path, n_frames=21, amplitude=3.0):
    """Create XYZ movie showing PCA component motion."""
    components = pca.components_
    centroid_flat = centroid_coords.reshape(1, -1)
    centroid_scaled = scaler.transform(centroid_flat)

    for pc_idx in range(min(3, components.shape[0])):
        pc_vector = components[pc_idx]
        movie_path = output_path.parent / f"{output_path.stem}_PC{pc_idx+1}_movie.xyz"

        with open(movie_path, 'w') as f:
            for frame_idx, t in enumerate(np.linspace(-amplitude, amplitude, n_frames)):
                displaced_scaled = centroid_scaled + t * pc_vector
                displaced = scaler.inverse_transform(displaced_scaled)
                displaced_coords = displaced.reshape(-1, 3)

                f.write(f"{len(atoms)}\n")
                f.write(f"PC{pc_idx+1} frame {frame_idx+1}/{n_frames}, t={t:.2f}\n")
                for atom, coord in zip(atoms, displaced_coords):
                    f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")


def create_embedding_movie(embedding_coords, original_coords, atoms, output_path, axis, method_name, n_frames=21):
    """Create XYZ movie by traversing embedding axis."""
    axis_values = embedding_coords[:, axis]
    min_val, max_val = axis_values.min(), axis_values.max()
    sample_points = np.linspace(min_val, max_val, n_frames)

    movie_path = output_path.parent / f"{output_path.stem}_{method_name}{axis+1}_movie.xyz"

    with open(movie_path, 'w') as f:
        for frame_idx, target_val in enumerate(sample_points):
            distances = np.abs(axis_values - target_val)
            nearest_idx = np.argmin(distances)
            geom_coords = original_coords[nearest_idx]

            f.write(f"{len(atoms)}\n")
            f.write(f"{method_name} axis {axis+1}, frame {frame_idx+1}/{n_frames}, val={target_val:.2f}\n")
            for atom, coord in zip(atoms, geom_coords):
                f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")


def analyze_combined(
    aligned_dir: Path,
    output_dir: Path,
    centroids: dict,
    threshold: float | None,
    families: list[str],
):
    """
    MAIN ANALYSIS: All families combined in one plot with 4 embedding methods.
    Centroids shown as stars with matching family colors.
    """
    print("\n  === COMBINED ALL-FAMILIES ANALYSIS ===")

    # Load all families
    all_features = []
    all_rmsds = []
    all_family_idx = []
    all_display_names = []
    all_smiles = []
    family_info = {}  # Store per-family info

    for fam_idx, family_name in enumerate(families):
        family_dir = aligned_dir / family_name
        if not family_dir.exists():
            continue

        coords, filenames, rmsds, smiles, n_filtered = load_family_geometries(family_dir, threshold)
        if len(coords) == 0:
            continue

        display_name = get_display_name(smiles, family_name)
        features = coords_to_features(coords)

        family_info[family_name] = {
            'smiles': smiles,
            'display_name': display_name,
            'n_molecules': len(coords),
            'fam_idx': fam_idx,
        }

        all_features.append(features)
        all_rmsds.extend(rmsds)
        all_family_idx.extend([fam_idx] * len(coords))
        all_display_names.append(display_name)
        all_smiles.append(smiles)

        print(f"    {display_name}: {len(coords)} molecules")

    if not all_features:
        print("    No data!")
        return

    # Combine all features
    features_combined = np.vstack(all_features)
    n_total = features_combined.shape[0]
    print(f"    Total: {n_total} molecules")

    # Add centroids to features
    centroid_info = []  # (smiles, fam_idx, features)
    for family_name, info in family_info.items():
        smiles = info['smiles']
        if smiles in centroids:
            cent_feat = centroids[smiles]['coords'].reshape(1, -1)
            centroid_info.append((smiles, info['fam_idx'], cent_feat, info['display_name']))

    if centroid_info:
        centroid_features = np.vstack([c[2] for c in centroid_info])
        features_with_centroids = np.vstack([features_combined, centroid_features])
        n_centroids = len(centroid_info)
        print(f"    Added {n_centroids} centroids")
    else:
        features_with_centroids = features_combined
        n_centroids = 0

    # Fit scaler on data only
    scaler = StandardScaler()
    scaler.fit(features_combined)

    # Run 2D embeddings
    print("    Running PCA...")
    pca_all, _ = run_pca(features_with_centroids, scaler, 2)
    print("    Running t-SNE...")
    tsne_all = run_tsne(features_with_centroids, scaler, 2)
    print("    Running UMAP...")
    umap_all = run_umap(features_with_centroids, scaler, 2)
    print("    Running Diffusion Map...")
    dm_all = run_diffusion_map(features_with_centroids, scaler, 2)

    # Separate data and centroids
    pca_data = pca_all[:n_total]
    tsne_data = tsne_all[:n_total]
    umap_data = umap_all[:n_total]
    dm_data = dm_all[:n_total]

    if n_centroids > 0:
        centroid_pca = [(centroid_info[i][1], pca_all[n_total + i], centroid_info[i][3]) for i in range(n_centroids)]
        centroid_tsne = [(centroid_info[i][1], tsne_all[n_total + i], centroid_info[i][3]) for i in range(n_centroids)]
        centroid_umap = [(centroid_info[i][1], umap_all[n_total + i], centroid_info[i][3]) for i in range(n_centroids)]
        centroid_dm = [(centroid_info[i][1], dm_all[n_total + i], centroid_info[i][3]) for i in range(n_centroids)]
    else:
        centroid_pca = centroid_tsne = centroid_umap = centroid_dm = []

    # Create 2x2 plot - THE MAIN GRAPH
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    thresh_str = "No Filter" if threshold is None else f"Max {threshold} Å"
    fig.suptitle(f"All Families Combined ({n_total} molecules) - {thresh_str}", fontsize=16, fontweight='bold')

    family_idx_array = np.array(all_family_idx)

    for ax, data, centroid_list, title, labels in [
        (axes[0, 0], pca_data, centroid_pca, "PCA", ("PC1", "PC2")),
        (axes[0, 1], tsne_data, centroid_tsne, "t-SNE", ("t-SNE1", "t-SNE2")),
        (axes[1, 0], umap_data, centroid_umap, "UMAP", ("UMAP1", "UMAP2")),
        (axes[1, 1], dm_data, centroid_dm, "Diffusion Map", ("DM1", "DM2")),
    ]:
        # Plot data points colored by family
        scatter = ax.scatter(
            data[:, 0], data[:, 1],
            c=family_idx_array,
            cmap='tab10',
            alpha=0.5,
            s=15,
        )

        # Plot centroids as stars with matching colors
        for fam_idx, pt, name in centroid_list:
            color = plt.cm.tab10(fam_idx / 10)
            ax.scatter(
                pt[0], pt[1],
                marker='*',
                s=600,
                c=[color],
                edgecolors='black',
                linewidths=2,
                zorder=10,
            )

        ax.set_xlabel(labels[0], fontsize=12)
        ax.set_ylabel(labels[1], fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Create legend
    unique_fam_idx = sorted(set(all_family_idx))
    handles = []
    for fam_idx in unique_fam_idx:
        color = plt.cm.tab10(fam_idx / 10)
        name = all_display_names[fam_idx]
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, markersize=10, label=name))

    # Add centroid marker to legend
    handles.append(plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='gray', markeredgecolor='black',
                   markersize=15, label='Centroid'))

    fig.legend(handles=handles, loc='center right', fontsize=11,
               bbox_to_anchor=(0.99, 0.5), frameon=True)

    plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    plot_path = output_dir / "combined_all_families.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {plot_path}")


def analyze_family(
    family_dir: Path,
    output_dir: Path,
    family_name: str,
    centroids: dict,
    threshold: float | None,
):
    """Run per-family analysis (2D, 3D plots and movies)."""
    coords, filenames, rmsds, smiles, n_filtered = load_family_geometries(family_dir, threshold)
    n_molecules = len(filenames)
    display_name = get_display_name(smiles, family_name)

    print(f"\n  {display_name}")
    if n_filtered > 0:
        print(f"    Filtered {n_filtered} exploded geometries")

    if n_molecules < 15:
        print(f"    Skipping: only {n_molecules} molecules")
        return

    print(f"    {n_molecules} molecules")

    centroid_data = centroids.get(smiles)
    if centroid_data:
        centroid_features = centroid_data['coords'].reshape(1, -1)
        atoms = centroid_data['atoms']
    else:
        centroid_features = None
        atoms = ATOM_ORDER

    features = coords_to_features(coords)
    scaler = StandardScaler()
    scaler.fit(features)

    if centroid_features is not None:
        features_all = np.vstack([features, centroid_features])
    else:
        features_all = features

    # 2D embeddings
    pca_2d, _ = run_pca(features_all, scaler, 2)
    tsne_2d = run_tsne(features_all, scaler, 2)
    umap_2d = run_umap(features_all, scaler, 2)
    dm_2d = run_diffusion_map(features_all, scaler, 2)

    # 3D embeddings
    pca_3d, _ = run_pca(features_all, scaler, 3)
    tsne_3d = run_tsne(features_all, scaler, 3)
    umap_3d = run_umap(features_all, scaler, 3)
    dm_3d = run_diffusion_map(features_all, scaler, 3)

    # Separate centroid
    if centroid_features is not None:
        pca_2d_data, pca_2d_cent = pca_2d[:-1], pca_2d[-1]
        tsne_2d_data, tsne_2d_cent = tsne_2d[:-1], tsne_2d[-1]
        umap_2d_data, umap_2d_cent = umap_2d[:-1], umap_2d[-1]
        dm_2d_data, dm_2d_cent = dm_2d[:-1], dm_2d[-1]
        pca_3d_data, pca_3d_cent = pca_3d[:-1], pca_3d[-1]
        tsne_3d_data, tsne_3d_cent = tsne_3d[:-1], tsne_3d[-1]
        umap_3d_data, umap_3d_cent = umap_3d[:-1], umap_3d[-1]
        dm_3d_data, dm_3d_cent = dm_3d[:-1], dm_3d[-1]
    else:
        pca_2d_data, pca_2d_cent = pca_2d, None
        tsne_2d_data, tsne_2d_cent = tsne_2d, None
        umap_2d_data, umap_2d_cent = umap_2d, None
        dm_2d_data, dm_2d_cent = dm_2d, None
        pca_3d_data, pca_3d_cent = pca_3d, None
        tsne_3d_data, tsne_3d_cent = tsne_3d, None
        umap_3d_data, umap_3d_cent = umap_3d, None
        dm_3d_data, dm_3d_cent = dm_3d, None

    # 2D Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{display_name} ({n_molecules} molecules)", fontsize=14)

    for ax, data, cent, title, labels in [
        (axes[0, 0], pca_2d_data, pca_2d_cent, "PCA", ("PC1", "PC2")),
        (axes[0, 1], tsne_2d_data, tsne_2d_cent, "t-SNE", ("t-SNE1", "t-SNE2")),
        (axes[1, 0], umap_2d_data, umap_2d_cent, "UMAP", ("UMAP1", "UMAP2")),
        (axes[1, 1], dm_2d_data, dm_2d_cent, "Diffusion Map", ("DM1", "DM2")),
    ]:
        sc = ax.scatter(data[:, 0], data[:, 1], c=rmsds, cmap='viridis', alpha=0.7, s=20)
        if cent is not None:
            ax.scatter(cent[0], cent[1], marker='*', s=400, c='red',
                      edgecolors='black', linewidths=1.5, zorder=10, label='Centroid')
            ax.legend(loc='upper right')
        plt.colorbar(sc, ax=ax, label="RMSD")
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_dir / f"{family_name}_2d.png", dpi=150)
    plt.close()

    # 3D Plot
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"{display_name} ({n_molecules} molecules) - 3D", fontsize=14)

    for idx, (data, cent, title, labels) in enumerate([
        (pca_3d_data, pca_3d_cent, "PCA", ("PC1", "PC2", "PC3")),
        (tsne_3d_data, tsne_3d_cent, "t-SNE", ("t-SNE1", "t-SNE2", "t-SNE3")),
        (umap_3d_data, umap_3d_cent, "UMAP", ("UMAP1", "UMAP2", "UMAP3")),
        (dm_3d_data, dm_3d_cent, "Diffusion Map", ("DM1", "DM2", "DM3")),
    ]):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=rmsds, cmap='viridis', alpha=0.6, s=15)
        if cent is not None:
            ax.scatter(cent[0], cent[1], cent[2], marker='*', s=300, c='red',
                      edgecolors='black', linewidths=1, zorder=10)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.set_title(title)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label='RMSD')
    plt.savefig(output_dir / f"{family_name}_3d.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Movies
    movie_base = output_dir / family_name

    if centroid_data:
        pca_data_only = PCA(n_components=min(5, features.shape[1]))
        pca_data_only.fit(scaler.transform(features))
        create_pca_movie(pca_data_only, scaler, centroid_data['coords'], atoms, movie_base)

        var_str = ", ".join([f"PC{i+1}:{v*100:.1f}%" for i, v in enumerate(pca_data_only.explained_variance_ratio_[:5])])
        print(f"    PCA variance: {var_str}")

    for method, emb_3d in [('tSNE', tsne_3d_data), ('UMAP', umap_3d_data), ('DM', dm_3d_data)]:
        for axis in range(3):
            create_embedding_movie(emb_3d, coords, atoms, movie_base, axis, method)


def main():
    aligned_dir = Path("aligned_output")
    base_output_dir = Path("analysis_output")
    centroids_dir = Path("centroids")

    # Only two thresholds: no filter and 5.0 Å
    THRESHOLDS = [None, 5.0]

    print("=" * 70)
    print("Complete Embedding Analysis")
    print("=" * 70)

    centroids = load_centroids(centroids_dir)
    print(f"Loaded {len(centroids)} centroid structures")

    families = ["family_1", "family_2", "family_3"]

    for threshold in THRESHOLDS:
        if threshold is None:
            thresh_name = "no_filter"
            print(f"\n{'='*70}")
            print(f"Threshold: No filtering")
            print(f"{'='*70}")
        else:
            thresh_name = f"max_{threshold:.1f}A"
            print(f"\n{'='*70}")
            print(f"Threshold: {threshold} Å max pairwise distance")
            print(f"{'='*70}")

        output_dir = base_output_dir / thresh_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # MOST IMPORTANT: Combined all-families analysis
        analyze_combined(aligned_dir, output_dir, centroids, threshold, families)

        # Per-family analysis
        for family_name in families:
            family_dir = aligned_dir / family_name
            if family_dir.exists():
                analyze_family(family_dir, output_dir, family_name, centroids, threshold)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"\nOutput structure:")
    print(f"  analysis_output/no_filter/")
    print(f"  analysis_output/max_5.0A/")
    print("\nMost important file:")
    print("  combined_all_families.png - All families in one plot with 4 methods")


if __name__ == "__main__":
    main()
