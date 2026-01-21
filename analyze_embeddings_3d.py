#!/usr/bin/env python3
"""
3D embedding analysis with PCA, t-SNE, UMAP, and Diffusion Maps.
Also generates PCA component movies showing molecular motions.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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

ATOM_ORDER = ['C', 'H', 'H', 'C', 'H', 'H']  # Standard ethylene atom order


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
    from scipy.spatial.distance import pdist
    return pdist(coords).max()


def load_family_geometries(
    family_dir: Path,
    max_distance_threshold: float | None = None,
) -> tuple[np.ndarray, list[str], list[float], str, int]:
    """
    Load all aligned geometries from a family directory.

    Args:
        family_dir: Path to family directory
        max_distance_threshold: If set, filter out geometries where max pairwise
                               distance exceeds this value (in Angstroms).
                               Typical values: 5-6 Å for ethylene.

    Returns:
        coords, filenames, rmsds, smiles, n_filtered
    """
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

        # Filter exploded geometries
        if max_distance_threshold is not None:
            max_dist = max_pairwise_distance(atoms_coords)
            if max_dist > max_distance_threshold:
                n_filtered += 1
                continue

        coords_list.append(atoms_coords)
        filenames.append(xyz_file.stem)
        rmsds.append(rmsd)

    return np.array(coords_list), filenames, rmsds, smiles, n_filtered


def get_display_name(smiles: str, family_name: str) -> str:
    if smiles in SMILES_TO_NAME:
        return SMILES_TO_NAME[smiles]
    return smiles if smiles else family_name


def coords_to_features(coords: np.ndarray) -> np.ndarray:
    n_molecules = coords.shape[0]
    return coords.reshape(n_molecules, -1)


def run_pca_3d(features: np.ndarray, scaler: StandardScaler) -> tuple[np.ndarray, PCA]:
    """Run PCA with 3 components."""
    features_scaled = scaler.transform(features)
    pca = PCA(n_components=min(3, features.shape[1], features.shape[0]))
    coords = pca.fit_transform(features_scaled)
    return coords, pca


def run_tsne_3d(features: np.ndarray, scaler: StandardScaler, perplexity: float = 30.0) -> np.ndarray:
    """Run t-SNE with 3 components."""
    features_scaled = scaler.transform(features)
    perp = min(perplexity, (features.shape[0] - 1) / 3)
    reducer = TSNE(n_components=3, perplexity=perp, random_state=42, max_iter=1000)
    return reducer.fit_transform(features_scaled)


def run_umap_3d(features: np.ndarray, scaler: StandardScaler, n_neighbors: int = 15) -> np.ndarray:
    """Run UMAP with 3 components."""
    features_scaled = scaler.transform(features)
    n_neighbors = min(n_neighbors, features.shape[0] - 1)
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=3, random_state=42)
    return reducer.fit_transform(features_scaled)


def run_diffusion_map_3d(features: np.ndarray, scaler: StandardScaler, n_neighbors: int = 10) -> np.ndarray:
    """Run Diffusion Map with 3 components."""
    features_scaled = scaler.transform(features)
    n_neighbors = min(n_neighbors, features.shape[0] - 1)
    embedding = SpectralEmbedding(
        n_components=3,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        random_state=42,
    )
    return embedding.fit_transform(features_scaled)


def create_pca_movie(
    pca: PCA,
    scaler: StandardScaler,
    centroid_coords: np.ndarray,
    atoms: list[str],
    output_dir: Path,
    family_name: str,
    n_frames: int = 21,
    amplitude: float = 3.0,
) -> None:
    """
    Create XYZ movie files showing PCA component motions.

    For each principal component, creates a trajectory that goes from
    -amplitude*std to +amplitude*std along that component direction.
    """
    # Get the PCA components (loadings) - these are in scaled space
    # Shape: (n_components, n_features) where n_features = n_atoms * 3
    components = pca.components_

    # Scale the centroid to the same space PCA was fitted in
    centroid_flat = centroid_coords.reshape(1, -1)
    centroid_scaled = scaler.transform(centroid_flat)

    # For each component, create a movie
    for pc_idx in range(min(3, components.shape[0])):
        pc_vector = components[pc_idx]  # Shape: (n_features,)

        # The component is in scaled space, need to convert displacements back
        # Displacement in original space = displacement in scaled space / scale
        # Since we're applying to scaled coords and inverting, we work in scaled space

        movie_path = output_dir / f"{family_name}_PC{pc_idx+1}_movie.xyz"

        with open(movie_path, 'w') as f:
            # Go from -amplitude to +amplitude along this PC
            for frame_idx, t in enumerate(np.linspace(-amplitude, amplitude, n_frames)):
                # Apply displacement in scaled space
                displaced_scaled = centroid_scaled + t * pc_vector

                # Transform back to original space
                displaced = scaler.inverse_transform(displaced_scaled)
                displaced_coords = displaced.reshape(-1, 3)

                # Write XYZ frame
                f.write(f"{len(atoms)}\n")
                f.write(f"PC{pc_idx+1} frame {frame_idx+1}/{n_frames}, t={t:.2f}\n")
                for atom, coord in zip(atoms, displaced_coords):
                    f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")

        print(f"    Created {movie_path.name}")


def plot_3d_embedding(ax, coords, colors, centroid=None, centroid_color='red', title='', cmap='viridis'):
    """Plot 3D scatter with optional centroid star."""
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=colors, cmap=cmap, alpha=0.6, s=15)
    if centroid is not None:
        ax.scatter(centroid[0], centroid[1], centroid[2],
                  marker='*', s=300, c=centroid_color, edgecolors='black', linewidths=1, zorder=10)
    ax.set_title(title)
    return scatter


def analyze_family_3d(
    family_dir: Path,
    output_dir: Path,
    family_name: str,
    centroids: dict | None = None,
    max_distance_threshold: float | None = 5.0,
) -> dict | None:
    """Run 3D embedding analysis on a single family."""
    coords, filenames, rmsds, smiles, n_filtered = load_family_geometries(
        family_dir, max_distance_threshold
    )
    n_molecules = len(filenames)
    display_name = get_display_name(smiles, family_name)

    print(f"\n{'='*60}")
    print(f"Analyzing {display_name}")
    print(f"{'='*60}")

    if n_filtered > 0:
        print(f"  Filtered {n_filtered} exploded geometries (max dist > {max_distance_threshold} Å)")

    if n_molecules < 15:
        print(f"  Skipping: only {n_molecules} molecules (need >= 15 for 3D)")
        return None

    print(f"  Loaded {n_molecules} molecules")

    # Check for centroid
    centroid_data = None
    if centroids and smiles in centroids:
        centroid_data = centroids[smiles]
        print(f"  Found centroid: {centroid_data['name']}")

    features = coords_to_features(coords)

    # Add centroid to features
    if centroid_data is not None:
        centroid_features = centroid_data['coords'].reshape(1, -1)
        features_with_centroid = np.vstack([features, centroid_features])
    else:
        features_with_centroid = features

    # Fit scaler on data only
    scaler = StandardScaler()
    scaler.fit(features)

    # Run 3D embeddings
    print("  Running PCA (3D)...")
    pca_all, pca = run_pca_3d(features_with_centroid, scaler)

    print("  Running t-SNE (3D)...")
    tsne_all = run_tsne_3d(features_with_centroid, scaler)

    print("  Running UMAP (3D)...")
    umap_all = run_umap_3d(features_with_centroid, scaler)

    print("  Running Diffusion Map (3D)...")
    dm_all = run_diffusion_map_3d(features_with_centroid, scaler)

    # Separate centroid
    if centroid_data is not None:
        pca_coords, pca_cent = pca_all[:-1], pca_all[-1]
        tsne_coords, tsne_cent = tsne_all[:-1], tsne_all[-1]
        umap_coords, umap_cent = umap_all[:-1], umap_all[-1]
        dm_coords, dm_cent = dm_all[:-1], dm_all[-1]
    else:
        pca_coords, pca_cent = pca_all, None
        tsne_coords, tsne_cent = tsne_all, None
        umap_coords, umap_cent = umap_all, None
        dm_coords, dm_cent = dm_all, None

    # Create 2x2 3D plot
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"{display_name} ({n_molecules} conformations) - 3D Embeddings", fontsize=14)

    # PCA 3D
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    scatter1 = plot_3d_embedding(ax1, pca_coords, rmsds, pca_cent, title='PCA')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')

    # t-SNE 3D
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_3d_embedding(ax2, tsne_coords, rmsds, tsne_cent, title='t-SNE')
    ax2.set_xlabel('t-SNE1')
    ax2.set_ylabel('t-SNE2')
    ax2.set_zlabel('t-SNE3')

    # UMAP 3D
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    plot_3d_embedding(ax3, umap_coords, rmsds, umap_cent, title='UMAP')
    ax3.set_xlabel('UMAP1')
    ax3.set_ylabel('UMAP2')
    ax3.set_zlabel('UMAP3')

    # Diffusion Map 3D
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    plot_3d_embedding(ax4, dm_coords, rmsds, dm_cent, title='Diffusion Map')
    ax4.set_xlabel('DM1')
    ax4.set_ylabel('DM2')
    ax4.set_zlabel('DM3')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(scatter1, cax=cbar_ax, label='RMSD')

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plot_path = output_dir / f"{family_name}_3d_embeddings.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    # Create PCA component movies if we have a centroid
    if centroid_data is not None:
        print("  Creating PCA component movies...")
        # Re-fit PCA on data only (not centroid) to get proper components
        pca_data = PCA(n_components=min(5, features.shape[1]))
        pca_data.fit(scaler.transform(features))

        create_pca_movie(
            pca_data, scaler,
            centroid_data['coords'],
            centroid_data['atoms'],
            output_dir,
            family_name,
        )

        # Print variance explained
        print(f"  PCA variance explained:")
        for i, var in enumerate(pca_data.explained_variance_ratio_[:5]):
            print(f"    PC{i+1}: {var*100:.1f}%")

    return {"family": family_name, "display_name": display_name, "n_molecules": n_molecules}


def create_embedding_movie(
    embedding_coords: np.ndarray,
    original_coords: np.ndarray,
    atoms: list[str],
    output_path: Path,
    axis: int,
    method_name: str,
    n_frames: int = 21,
) -> None:
    """
    Create XYZ movie by interpolating along an embedding axis.

    For non-linear methods (t-SNE, UMAP), we find the nearest actual
    geometry for each point along the axis.
    """
    # Get range of the specified axis
    axis_values = embedding_coords[:, axis]
    min_val, max_val = axis_values.min(), axis_values.max()

    # Sample points along this axis
    sample_points = np.linspace(min_val, max_val, n_frames)

    with open(output_path, 'w') as f:
        for frame_idx, target_val in enumerate(sample_points):
            # Find geometry with closest value on this axis
            distances = np.abs(axis_values - target_val)
            nearest_idx = np.argmin(distances)

            # Get the actual geometry
            geom_coords = original_coords[nearest_idx]

            # Write XYZ frame
            f.write(f"{len(atoms)}\n")
            f.write(f"{method_name} axis {axis+1}, frame {frame_idx+1}/{n_frames}, "
                   f"val={target_val:.2f}, geom_idx={nearest_idx}\n")
            for atom, coord in zip(atoms, geom_coords):
                f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")


def create_all_embedding_movies(
    coords: np.ndarray,
    atoms: list[str],
    pca_coords: np.ndarray,
    tsne_coords: np.ndarray,
    umap_coords: np.ndarray,
    dm_coords: np.ndarray,
    output_dir: Path,
    family_name: str,
) -> None:
    """Create movies for all embedding methods and axes."""
    methods = [
        ('tSNE', tsne_coords),
        ('UMAP', umap_coords),
        ('DM', dm_coords),
    ]

    for method_name, emb_coords in methods:
        for axis in range(3):
            output_path = output_dir / f"{family_name}_{method_name}{axis+1}_movie.xyz"
            create_embedding_movie(
                emb_coords, coords, atoms, output_path, axis, method_name
            )
            print(f"    Created {output_path.name}")


def main():
    aligned_dir = Path("aligned_output")
    output_dir = Path("analysis_output")
    centroids_dir = Path("centroids")
    output_dir.mkdir(exist_ok=True)

    # Filter threshold: molecules with max pairwise distance > this are excluded
    # Set to None to disable filtering
    MAX_DISTANCE_THRESHOLD = 5.0  # Angstroms (ethylene is ~3-4 Å normally)

    print("3D Embedding Analysis: PCA + t-SNE + UMAP + Diffusion Map")
    print("=" * 60)
    if MAX_DISTANCE_THRESHOLD:
        print(f"Filtering exploded geometries (max pairwise dist > {MAX_DISTANCE_THRESHOLD} Å)")

    centroids = load_centroids(centroids_dir)
    if centroids:
        print(f"Loaded {len(centroids)} centroid structures")

    # Analyze main families with centroids
    for family_name in ["family_1", "family_2", "family_3"]:
        family_dir = aligned_dir / family_name
        if family_dir.exists():
            result = analyze_family_3d(
                family_dir, output_dir, family_name, centroids, MAX_DISTANCE_THRESHOLD
            )

            # Also create t-SNE/UMAP/DM movies
            if result is not None:
                print("  Creating t-SNE/UMAP/DM axis movies...")
                # Reload data for movie creation (with same filter)
                coords, filenames, rmsds, smiles, _ = load_family_geometries(
                    family_dir, MAX_DISTANCE_THRESHOLD
                )
                features = coords_to_features(coords)

                # Get atoms from centroid if available
                if centroids and smiles in centroids:
                    atoms = centroids[smiles]['atoms']
                else:
                    atoms = ATOM_ORDER

                scaler = StandardScaler()
                scaler.fit(features)

                # Run embeddings (without centroid this time)
                pca_coords, _ = run_pca_3d(features, scaler)
                tsne_coords = run_tsne_3d(features, scaler)
                umap_coords = run_umap_3d(features, scaler)
                dm_coords = run_diffusion_map_3d(features, scaler)

                create_all_embedding_movies(
                    coords, atoms,
                    pca_coords, tsne_coords, umap_coords, dm_coords,
                    output_dir, family_name
                )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - *_3d_embeddings.png: 3D scatter plots")
    print("  - *_PC*_movie.xyz: PCA component motions (smooth interpolation)")
    print("  - *_tSNE*_movie.xyz: t-SNE axis traversal (nearest geometries)")
    print("  - *_UMAP*_movie.xyz: UMAP axis traversal (nearest geometries)")
    print("  - *_DM*_movie.xyz: Diffusion Map axis traversal (nearest geometries)")
    print("\nTo view movies in VMD:")
    print("  vmd analysis_output/family_2_tSNE1_movie.xyz")


if __name__ == "__main__":
    main()
