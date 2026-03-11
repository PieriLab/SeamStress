"""Dimensionality reduction analysis module."""

import colorsys
import json
import re
from pathlib import Path
import os 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import pandas as pd 
from ase import Atoms
from dscribe.descriptors import SOAP, MBTR
import seaborn as sns
from tqdm import tqdm 
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr
from seamstress.alignment import weighted_rmsd



# OPTIONAL: Map SMILES to human-readable names for display
SMILES_TO_NAME = {
    "[H+].[H][C-]=C([H])[H]": "H dissociation",
    "[H]C([H])=C([H])[H]": "Twist (ethylene)",
    "[H][C]C([H])([H])[H]": "Ethylidene",
    "[H][C][H].[H][C][H]": "2xCH2 radicals",
    "[H]C#C[H].[H][H]": "Acetylene + H2",
    "C.C.[HH].[HH].[HH].[HH]": "Full dissociation",
    "[H+].[H]C([H])([H])[C-3]": "H+ + CH3C-",
    "[H][C+]([H])[H].[H][C-3]": "CH3+ + CH-",
    "[H+].[H+].[H]C#C[H]": "2H+ + Acetylene",
}

# SMILES that correspond to actual MECIs from raw_centroids (not just family centroids)
MECI_SMILES = {
    "[H+].[H][C-]=C([H])[H]",  # H dissociation
    "[H]C([H])=C([H])[H]",      # Twist (ethylene)
    "[H][C]C([H])([H])[H]",     # Ethylidene
}

FAMILY_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]




def load_cpt_colormap(cpt_path: Path, name: str = 'custom_cpt') -> LinearSegmentedColormap:
    """
    Load a GMT CPT (Color Palette Table) file and convert to matplotlib colormap.

    Args:
        cpt_path: Path to the .cpt file
        name: Name for the colormap

    Returns:
        LinearSegmentedColormap object
    """
    colors = []
    positions = []

    with open(cpt_path) as f:
        for line in f:
            line = line.strip()
            # Skip comments and special lines
            if not line or line.startswith('#') or line.startswith('B') or line.startswith('F'):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            # Parse start position and color
            try:
                pos_start = float(parts[0])
                color_start = parts[1]
                pos_end = float(parts[2])
                color_end = parts[3]

                # Parse HSV color (format: H-S-V)
                def parse_hsv(hsv_str):
                    h, s, v = map(float, hsv_str.split('-'))
                    # Normalize hue to [0, 1] range
                    h = h / 360.0
                    # Convert HSV to RGB
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    return (r, g, b)

                rgb_start = parse_hsv(color_start)
                rgb_end = parse_hsv(color_end)

                # Normalize positions to [0, 1]
                # CPT uses -1 to 1 range
                norm_pos_start = (pos_start + 1) / 2
                norm_pos_end = (pos_end + 1) / 2

                # Add both colors
                if not positions or positions[-1] != norm_pos_start:
                    positions.append(norm_pos_start)
                    colors.append(rgb_start)
                positions.append(norm_pos_end)
                colors.append(rgb_end)

            except (ValueError, IndexError):
                continue

    # Create colormap
    cmap = LinearSegmentedColormap.from_list(name, list(zip(positions, colors)))
    return cmap


def load_centroids_from_aligned_output(aligned_dir: Path) -> list:
    """
    Load centroids or reference structures from aligned_output/family_* folders.

    Returns:
        List of dictionaries with keys: coords, species, name, smiles, family_name, is_meci, meci_number
    """
    centroids = []
    if not aligned_dir.exists():
        return centroids

    # First check for aligned_centroids.xyz at root (multi-family mode)
    aligned_centroids_file = aligned_dir / "aligned_centroids.xyz"
    if aligned_centroids_file.exists():
        with open(aligned_centroids_file) as f:
            lines = f.readlines()

        idx = 0
        meci_num = 1
        while idx < len(lines):
            n_atoms = int(lines[idx].strip())
            header = lines[idx + 1].strip()

            # Extract family name and SMILES
            family_match = re.search(r"(Family_\d+)\s+(\S+)", header)
            if family_match:
                family_name = family_match.group(1).lower()
                smiles = family_match.group(2)
            else:
                family_name = "unknown"
                smiles = "Unknown"

            coords = []
            species = []
            for i in range(idx + 2, idx + 2 + n_atoms):
                parts = lines[i].split()
                species.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

            # Extract centroid label if available
            filename_match = re.search(r"Centroid\s+(\S+\.xyz)", header)
            centroid_label = filename_match.group(1).replace('.xyz', '') if filename_match else f"centroid{meci_num}"

            centroids.append({
                'coords': np.array(coords),
                'species': species,
                'name': f"{family_name}_{centroid_label}",
                'smiles': smiles,
                'family_name': family_name,
                'is_meci': True,
                'meci_number': meci_num
            })

            idx += 2 + n_atoms
            meci_num += 1

        return centroids

    # Otherwise, check individual family folders
    for family_dir in sorted(aligned_dir.glob("family_*")):
        # Check centroids.xyz first
        centroids_file = family_dir / "centroids.xyz"
        if centroids_file.exists():
            with open(centroids_file) as f:
                lines = f.readlines()
            idx = 0
            centroid_num = 1
            while idx < len(lines):
                n_atoms = int(lines[idx].strip())
                header = lines[idx + 1].strip()

                smiles_match = re.search(r"Family_\d+\s+(\S+)", header)
                smiles = smiles_match.group(1) if smiles_match else "Unknown"

                coords = []
                species = []
                for i in range(idx + 2, idx + 2 + n_atoms):
                    parts = lines[i].split()
                    species.append(parts[0])
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

                centroids.append({
                    'coords': np.array(coords),
                    'species': species,
                    'name': f"{family_dir.name}_centroid{centroid_num}",
                    'smiles': smiles,
                    'family_name': family_dir.name,
                    'is_meci': True,
                    'meci_number': centroid_num
                })

                idx += 2 + n_atoms
                centroid_num += 1
            continue

        # Fallback to reference.xyz
        ref_file = family_dir / "reference.xyz"
        if not ref_file.exists():
            continue

        with open(ref_file) as f:
            lines = f.readlines()
        n_atoms = int(lines[0].strip())
        header = lines[1].strip()

        smiles_match = re.search(r"Family_\d+\s+(\S+)", header)
        if not smiles_match:
            continue
        smiles = smiles_match.group(1)

        coords = []
        species = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            species.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        centroids.append({
            'coords': np.array(coords),
            'species': species,
            'name': family_dir.name,
            'smiles': smiles,
            'family_name': family_dir.name,
            'is_meci': False,
            'meci_number': None
        })

    return centroids


def max_pairwise_distance(coords: np.ndarray) -> float:
    """Calculate maximum pairwise distance in a geometry."""
    return pdist(coords).max()

def compute_rmsd_matrix(
    aligned_dir: Path,
    centroids: list,
    output_dir : Path,
) -> None:
    """
    Compute full RMSD matrix between:
      - all aligned geometries (n)
      - all centroids (m)

    Produces (n + m) x (n + m) symmetric matrix and saves as CSV.

    Rows/columns are labeled by:
        family_X/filename.xyz   for geometries
        centroid_name           for centroids
    """

    print("\nComputing RMSD matrix...")

    all_coords = []
    labels = []

    family_dirs = sorted(aligned_dir.glob("family_*"))

    for family_dir in family_dirs:
        print(family_dir)
        for xyz_file in sorted(family_dir.glob("*.xyz")):
            if xyz_file.name == "reference.xyz":
                continue
            if xyz_file.name == f"{family_dir.name}.xyz":
                continue
            if xyz_file.name == f"{family_dir.name}":
                continue
            with open(xyz_file) as f:
                lines = f.readlines()

            n_atoms = int(lines[0].strip())
            coords = []

            for i in range(2, 2 + n_atoms):
                parts = lines[i].split()
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

            coords = np.array(coords)

            all_coords.append(coords)
            labels.append(f"{xyz_file.name}")

    n = len(all_coords)

    for centroid in centroids:
        all_coords.append(centroid["coords"])
        labels.append(centroid["name"])

    m = len(centroids)

    if n + m == 0:
        print("  No geometries or centroids found.")
        return

    print(f"  {n} geometries")
    print(f"  {m} centroids")
    print(f"  Matrix size: {n+m} x {n+m}")

   
    total = n + m
    rmsd_matrix = np.zeros((total, total))

    def rmsd(A, B):
        return np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1)))

    for i in range(total):
        for j in range(i, total):
            value = rmsd(all_coords[i], all_coords[j])
            rmsd_matrix[i, j] = value
            rmsd_matrix[j, i] = value

    df = pd.DataFrame(rmsd_matrix, index=labels, columns=labels)

    output_path = output_dir / "rmsd_matrix.csv"
    df.to_csv(output_path)

    print(f"  Saved RMSD matrix to: {output_path}\n")

def load_family_geometries(family_dir: Path, max_distance_threshold: float | None = None):
    """
    Load all geometries and species from a family directory.

    Returns:
        coords_list: list of ndarray of shape (n_atoms, 3)
        species_list: list of list of atomic symbols (length n_atoms)
        filenames: list of filenames (without extension)
        rmsds: list of RMSDs
        smiles: SMILES string from header
        n_filtered: number of geometries filtered due to max_distance_threshold
    """
    coords_list, species_list, filenames, rmsds = [], [], [], []
    smiles = ""
    n_filtered = 0

    for xyz_file in sorted(family_dir.glob("*.xyz")):
        if xyz_file.name == "reference.xyz":
            continue

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
        atoms_species = []
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            atoms_species.append(parts[0])
            atoms_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        atoms_coords = np.array(atoms_coords)

        if max_distance_threshold is not None and max_pairwise_distance(atoms_coords) > max_distance_threshold:
            n_filtered += 1
            continue

        coords_list.append(atoms_coords)
        species_list.append(atoms_species)
        filenames.append(xyz_file.stem)
        rmsds.append(rmsd)

    if len(coords_list) == 0:
        return np.array([]), [], [], [], smiles, n_filtered

    return np.array(coords_list), species_list, filenames, rmsds, smiles, n_filtered


def get_display_name(smiles: str, family_name: str) -> str:
    """Get human-readable name for a SMILES string."""
    return SMILES_TO_NAME.get(smiles, smiles if smiles else family_name)





def coords_to_soap(coords: np.ndarray, species, r_cut=5.0, n_max=8, l_max=6, average="inner") -> np.ndarray:
    """Convert 3D coordinates to SOAP features."""
    n_samples, n_atoms, _ = coords.shape

    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average=average,
        sparse=False
    )

    features = []
    for i in range(n_samples):
        atoms = Atoms(symbols=species[:n_atoms], positions=coords[i])
        features.append(soap.create(atoms))

    return np.array(features)


def coords_to_mbtr(coords: np.ndarray, species, normalization="none") -> np.ndarray:
    """
    Convert 3D coordinates (NumPy array) to MBTR features, one sample at a time.
    
    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n_samples, n_atoms, 3)
    species : list
        List of atomic symbols corresponding to each atom
    normalization : str
        Normalization mode for MBTR ("none", "l2", etc.)
    
    Returns
    -------
    np.ndarray
        Array of MBTR feature vectors, shape (n_samples, feature_length)
    """
    n_samples, n_atoms, _ = coords.shape
    features = []

    for i in range(n_samples):

   
        atoms = Atoms(symbols=species[:n_atoms], positions=coords[i])
        atoms.set_momenta(None)
        atoms.set_velocities(None)
        
        mbtr = MBTR(
            geometry={"function": "distance"},
            grid={"min": 0.5, "max": 5.0, "sigma": 0.1, "n": 50},
            weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
            normalization=normalization,
            species=species[:n_atoms],  # explicitly provide species
            periodic=False
        )
        
        feature = mbtr.create([atoms])  # note the list wrapper
        features.append(feature)

    return np.array(features)


def coords_to_inverse_eigenvalues(coords: np.ndarray) -> np.ndarray:
    """Convert 3D coordinates to inverse-distance eigenvalue features."""
    n_samples = coords.shape[0]

    features = []
    for i in range(n_samples):
        dist_matrix = squareform(pdist(coords[i]))
        inv_dist_matrix = 1.0 / (dist_matrix + 1e-8)

        eigenvals = np.linalg.eigvals(inv_dist_matrix)
        eigenvals = np.sort(eigenvals)[::-1]

        features.append(eigenvals.real)

    return np.array(features)


def coords_to_inverse_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Convert 3D coordinates to flattened inverse distance matrix features.
    
    Parameters:
        coords: np.ndarray of shape (n_samples, n_atoms, 3)
    
    Returns:
        np.ndarray of shape (n_samples, n_atoms*(n_atoms-1)//2)
        Flattened upper-triangle inverse distance matrix for each sample.
    """
    n_samples = coords.shape[0]
    features = []

    for i in range(n_samples):
        # Compute pairwise distances between atoms in sample i
        dist_matrix = squareform(pdist(coords[i]))  # shape (n_atoms, n_atoms)

        # Avoid division by zero for self-distances
        np.fill_diagonal(dist_matrix, np.inf)

        # Take inverse distances and flatten upper triangle only
        inv_dist = 1 / dist_matrix
        upper_tri = inv_dist[np.triu_indices_from(inv_dist, k=1)]
        features.append(upper_tri)

    return np.array(features)


def coords_to_flat_cartesian(coords: np.ndarray) -> np.ndarray:
    """Convert 3D coordinates to flattened Cartesian features."""
    return coords.reshape(coords.shape[0], -1)




def reduce_features(features: np.ndarray, method: str, n_components: int = 2):
    """
    Reduce features using a specified dimensionality reduction method.
    Features are internally scaled before reduction.

    Args:
        features: Feature matrix (samples × features)
        method: Reduction method name: "pca", "tsne", "umap", "dm"
        n_components: Target number of dimensions

    Returns:
        Reduced feature matrix (samples × n_components)
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    method = method.lower()
    if method == "pca":
        pca = PCA(n_components=min(n_components, X_scaled.shape[1], X_scaled.shape[0]))
        return pca.fit_transform(X_scaled)

    elif method == "tsne":
        perp = min(30.0, max(1, (X_scaled.shape[0] - 1) / 3))
        return TSNE(
            n_components=n_components,
            perplexity=perp,
            random_state=42,
            max_iter=1000
        ).fit_transform(X_scaled)

    elif method == "umap":
        n_neighbors = min(15, X_scaled.shape[0] - 1)
        return UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            n_components=n_components,
            random_state=42
        ).fit_transform(X_scaled)

    elif method in ["dm", "diffusion_map"]:
        n_neighbors = min(10, X_scaled.shape[0] - 1)
        emb = SpectralEmbedding(
            n_components=n_components + 1,  # skip first eigenvector later
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
            random_state=42
        )
        return emb.fit_transform(X_scaled)[:, 1:]  # skip first trivial eigenvector

    else:
        raise ValueError(f"Unknown reduction method: {method}")



def compute_stress(D_feat, D_true):
    """Compute normalized stress between distance matrices"""
    triu_idx = np.triu_indices_from(D_feat, k=1)
    diff = D_feat[triu_idx] - D_true[triu_idx]
    num = np.sum(diff**2)
    denom = np.sum(D_true[triu_idx]**2)
    return np.sqrt(num / denom)

def compute_continuity(X_high, X_low, k=10):
    """
    Compute continuity metric.
    Measures how many true neighbors from high-D are lost in low-D.
    """

    n = X_high.shape[0]

    nbrs_high = NearestNeighbors(n_neighbors=k + 1).fit(X_high)
    nbrs_low = NearestNeighbors(n_neighbors=k + 1).fit(X_low)

    high_neighbors = nbrs_high.kneighbors(return_distance=False)[:, 1:]
    low_neighbors = nbrs_low.kneighbors(return_distance=False)[:, 1:]

    continuity_sum = 0

    for i in range(n):

        high_set = set(high_neighbors[i])
        low_set = set(low_neighbors[i])

        missing = high_set - low_set

        if len(missing) == 0:
            continue

        low_ranking = nbrs_low.kneighbors(
            X_low[i].reshape(1, -1),
            n_neighbors=n,
            return_distance=False
        )[0]

        for j in missing:

            rank = np.where(low_ranking == j)[0][0]

            if rank > k:
                continuity_sum += rank - k

    normalization = 2 / (n * k * (2 * n - 3 * k - 1))

    return 1 - normalization * continuity_sum


def full_embedding_analysis(X_high, X_low, k=10):
    """
    Compute unsupervised embedding quality metrics.
    """

    trust = trustworthiness(X_high, X_low, n_neighbors=k)

    continuity = compute_continuity(X_high, X_low, k)

    dist_high = pdist(X_high)
    dist_low = pdist(X_low)

    pearson_corr = pearsonr(dist_high, dist_low)[0]
    spearman_corr = spearmanr(dist_high, dist_low)[0]

    return {
        "trustworthiness": trust,
        "continuity": continuity,
        "pearson_dist_corr": pearson_corr,
        "spearman_dist_corr": spearman_corr,
    }



def save_static_plots(result, method_name, feature_name, threshold_name, output_dir, use_smiles=False):
    """
    Generate and save static plots as PNG and SVG.

    Args:
        result: Dictionary containing embedding data, families, filenames, centroids
        method_name: Name of the dimensionality reduction method (pca, tsne, umap, dm)
        feature_name: Name of the feature type (cartesian_aligned, inverse_distance)
        threshold_name: Name of the threshold (no_filter, max_5.0A)
        output_dir: Base output directory
        use_smiles: If True, use SMILES strings in legend; if False, use display names
    """
    # Load PRL style
    style_path = Path(__file__).parent / "styles" / "prl.mplstyle"
    plt.style.use(str(style_path))

    # Create threshold subfolder
    threshold_dir = output_dir / threshold_name
    threshold_dir.mkdir(parents=True, exist_ok=True)

    # Get data
    coords = result[method_name]
    filenames = result['filenames']
    family_idx = result['family_idx']
    families = result['families']
    centroids = result['centroids'].get(method_name, [])

    # Calculate total points for percentages
    total_points = len(coords)

    # Axis labels
    axis_labels = {
        'pca': ['PC1', 'PC2'],
        'tsne': ['t-SNE 1', 't-SNE 2'],
        'umap': ['UMAP 1', 'UMAP 2'],
        'dm': ['DM 2', 'DM 3']
    }
    labels = axis_labels.get(method_name, ['Dim 1', 'Dim 2'])

    # Get colors from colormap (equally distributed)
    n_families = len(families)
    # Load sealand colormap from cpt-city (GMT)
    cpt_path = Path(__file__).parent / "styles" / "GMT_sealand.cpt"
    cmap = load_cpt_colormap(cpt_path, name='sealand')
    # Sample colors evenly across the colormap
    if n_families == 1:
        colors = [cmap(0.5)]
    else:
        colors = [cmap(i / (n_families - 1)) for i in range(n_families)]

    # Create figure (override style's 3.4x3.4 for better visibility with multiple families)
    fig, ax = plt.subplots(figsize=(7, 5.25))

    # Plot each family
    for i, fam in enumerate(families):
        color = colors[i]
        mask = [j for j, idx in enumerate(family_idx) if idx == i]

        if mask:
            x = [coords[j][0] for j in mask]
            y = [coords[j][1] for j in mask]

            # Calculate percentage
            percentage = (fam['count'] / total_points) * 100

            # Choose label based on use_smiles flag
            if use_smiles:
                label = f"{fam['smiles']} ({percentage:.1f}%)"
            else:
                label = f"{fam['name']} ({percentage:.1f}%)"

            ax.scatter(x, y, c=[color], label=label,
                      s=30, alpha=0.7, edgecolors='black', linewidths=0.3)

    # Plot centroids as stars (only user-provided centroids)
    for centroid in centroids:
        # Only plot stars for user-provided centroids, not auto-generated references
        if not centroid.get('is_meci', False):
            continue

        color = colors[centroid['idx']]
        # Use the base filename as label (e.g., "meci1", "meci2")
        centroid_label = centroid['name'].split('_')[-1]  # Extract last part (e.g., "meci1" from "family_1_meci1")

        ax.scatter([centroid['x']], [centroid['y']],
                  marker='*', s=400, c=[color],
                  edgecolors='black', linewidths=1.2,
                  label=centroid_label,
                  zorder=10)

    # Labels and title
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    method_display = method_name.upper()
    feature_display = feature_name.replace('_', ' ').title()
    threshold_display = threshold_name.replace('_', ' ').title()

    title = f"{feature_display} - {method_display}\n"
    title += f"Threshold: {threshold_display} | {len(coords)} molecules"
    ax.set_title(title, fontweight='bold')

    # Legend (PRL style has frameon=False by default)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    # Tight layout
    plt.tight_layout()

    # Save as PNG and SVG
    base_name = f"{feature_name}_{method_name}"
    png_path = threshold_dir / f"{base_name}.png"
    svg_path = threshold_dir / f"{base_name}.svg"

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)

    print(f"        Saved: {png_path.relative_to(output_dir)}")
    print(f"        Saved: {svg_path.relative_to(output_dir)}")



def convert_numpy(obj):
    """
    Recursively convert NumPy arrays and scalars to native Python types
    for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def generate_html(all_data, output_path):
    """Generate HTML explorer with dropdowns for all features and all DR methods."""

    # Extract feature types from all_data keys
    feature_types = sorted(set(k.split("_", 1)[1] for k in all_data.keys()))
    dr_methods = ["pca", "tsne", "umap", "dm"]

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>SeamStress - Dimensionality Reduction Explorer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .controls {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .controls label {{ margin-right: 10px; font-weight: bold; }}
        .controls select {{ padding: 8px 12px; font-size: 14px; margin-right: 20px; border-radius: 4px; border: 1px solid #ccc; }}
        #plot {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        .info {{ background: #e7f3ff; padding: 10px; border-radius: 4px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <h1>SeamStress - Dimensionality Reduction Explorer</h1>
    <div class="info">Hover over points to see xyz filename. Click legend items to show/hide families.</div>
    <div class="controls">
        <label>Feature Type:</label>
        <select id="feature">
            {"".join([f'<option value="{ft}">{ft}</option>' for ft in feature_types])}
        </select>
        <label>Method:</label>
        <select id="method">
            {"".join([f'<option value="{m}">{m.upper()}</option>' for m in dr_methods])}
        </select>
    </div>
    <div id="plot"></div>

    <script>
    const DATA = {json.dumps(all_data)};
    const COLORS = {json.dumps(FAMILY_COLORS)};
    const AXIS_LABELS = {{
        'pca': ['PC1', 'PC2'],
        'tsne': ['t-SNE1', 't-SNE2'],
        'umap': ['UMAP1', 'UMAP2'],
        'dm': ['DM2', 'DM3']
    }};

    function updatePlot() {{
        const feature = document.getElementById('feature').value;
        const method = document.getElementById('method').value;
        const keyPrefix = Object.keys(DATA).find(k => k.endsWith("_" + feature));
        if (!keyPrefix) {{
            document.getElementById('plot').innerHTML = '<p>No data for this feature</p>';
            return;
        }}
        const data = DATA[keyPrefix];
        const coords = data[method];
        if (!coords) {{
            document.getElementById('plot').innerHTML = '<p>No data for this method</p>';
            return;
        }}
        const filenames = data.filenames;
        const familyIdx = data.family_idx;
        const families = data.families;
        const centroids = data.centroids[method] || [];

        const traces = [];

        families.forEach((fam, i) => {{
            const mask = familyIdx.map((idx, j) => idx === i ? j : -1).filter(j => j >= 0);
            traces.push({{
                x: mask.map(j => coords[j][0]),
                y: mask.map(j => coords[j][1]),
                mode: 'markers',
                type: 'scatter',
                name: fam.name + ' (' + fam.count + ')',
                text: mask.map(j => filenames[j]),
                hovertemplate: '%{{text}}<extra>' + fam.name + '</extra>',
                marker: {{ color: COLORS[i], size: 8, opacity: 0.6 }}
            }});
        }});

        centroids.forEach(c => {{
            traces.push({{
                x: [c.x],
                y: [c.y],
                mode: 'markers',
                type: 'scatter',
                name: 'Centroid: ' + c.name,
                text: ['CENTROID: ' + c.name],
                hovertemplate: '%{{text}}<extra></extra>',
                marker: {{ color: COLORS[c.idx] || 'black', size: 24, symbol: 'star', line: {{ color: 'black', width: 2 }} }},
                showlegend: false
            }});
        }});

        const labels = AXIS_LABELS[method];
        const layout = {{
            title: feature + ' - ' + method.toUpperCase() + ' (' + coords.length + ' molecules)',
            xaxis: {{ title: labels[0] }},
            yaxis: {{ title: labels[1] }},
            height: 800,
            width: 1200,
            hovermode: 'closest',
            legend: {{ x: 1.02, y: 1 }}
        }};

        Plotly.newPlot('plot', traces, layout);
    }}

    document.getElementById('feature').addEventListener('change', updatePlot);
    document.getElementById('method').addEventListener('change', updatePlot);

    updatePlot();
    </script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"\nSaved: {output_path}")


def results_dict_to_df(results):
    """
    Convert nested results dict (feature -> method -> n_dim -> metrics) to flat DataFrame.
    """
    rows = []
    for feature, methods in results.items():
        for method, dims in methods.items():
            for n_dim, metrics in dims.items():
                row = {
                    "feature": feature,
                    "reduction": method,
                    "n_components": n_dim
                }
                # Ensure all expected metrics exist
                expected_metrics = ["trustworthiness", "continuity", "pearson_dist_corr", "spearman_dist_corr","stress"]
                for m in expected_metrics:
                    row[m] = metrics.get(m, float("nan"))
                rows.append(row)
    return pd.DataFrame(rows)


def plot_metrics_heatmap(df, metric, save_folder="plots", save_png=True, save_svg=True):


    os.makedirs(save_folder, exist_ok=True)
    
    if metric not in df.columns:
        print(f"Warning: metric '{metric}' not in DataFrame. Skipping heatmap.")
        return

    # Combine feature + method to avoid duplicate indices
    df['feature_reduction'] = df['feature'].astype(str) + '-' + df['reduction'].astype(str)
    
    # Pivot table: rows = feature_reduction, columns = n_components
    heatmap_data = df.pivot(index='feature_reduction', columns='n_components', values=metric)
    
    if heatmap_data.empty:
        print(f"No data to plot for metric '{metric}'. Skipping heatmap.")
        return
    
    # Ensure values are Python floats
    heatmap_data = heatmap_data.astype(float)
    
    # Create figure with constrained_layout to avoid colorbar warning
    plt.figure(figsize=(10, max(6, 0.5*len(heatmap_data))), constrained_layout=True)
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar=True,
        linewidths=0.5
    )
    
    plt.title(f"{metric.replace('_',' ').title()} Across Features, Reductions, and Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Feature-Reduction")
    
    # Save files
    if save_png:
        path_png = os.path.join(save_folder, f"{metric}_heatmap.png")
        plt.savefig(path_png, dpi=300)
        print(f"Saved heatmap PNG: {path_png}")
    if save_svg:
        path_svg = os.path.join(save_folder, f"{metric}_heatmap.svg")
        plt.savefig(path_svg)
        print(f"Saved heatmap SVG: {path_svg}")
    
    plt.close()

import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

def compute_stress(D_feat, D_true):
    """Compute normalized stress between distance matrices"""
    triu_idx = np.triu_indices_from(D_feat, k=1)
    diff = D_feat[triu_idx] - D_true[triu_idx]
    num = np.sum(diff**2)
    denom = np.sum(D_true[triu_idx]**2)
    return np.sqrt(num / denom)


def run_analysis(
    aligned_dir: Path,
    output_dir: Path,
    families_to_include=None,
    target_dims=[2],
    filter_outliers: bool = False,
    max_distance_threshold: float | None = None,
):
    """
    Run dimensionality reduction analysis on aligned geometries using load_family_geometries,
    """
    use_smiles_in_legend = True
    FEATURE_TYPES = {
        "SOAP": coords_to_soap,
        "inv_eigenval": coords_to_inverse_eigenvalues,
        "inverse_dist_matrix": coords_to_inverse_distance_matrix,
        "flatten_cartesian": coords_to_flat_cartesian
    }
    METHODS = ["pca", "tsne", "umap", "dm"]
    expected_metrics = ["trustworthiness", "continuity", "pearson_dist_corr", "spearman_dist_corr", "stress"]

    # ----------------------------
    # Discover families
    # ----------------------------
    all_families = sorted([f.name for f in aligned_dir.glob("family_*")])
    families = [f for f in all_families if f in families_to_include] if families_to_include else all_families
    if not families:
        print("No families to analyze")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load structures & species
    # ----------------------------
    structures, species_list, names, family_idx, families_info = [], [], [], [], []
    for i, fam_name in enumerate(families):
        fam_path = aligned_dir / fam_name
        coords_list, species_sublist, filenames, rmsds, smiles, n_filtered = load_family_geometries(
            fam_path,
            max_distance_threshold=max_distance_threshold if filter_outliers else None
        )
        if len(coords_list) == 0:
            continue
        structures.extend(coords_list)
        species_list.extend(species_sublist)
        names.extend(filenames)
        family_idx.extend([i] * len(coords_list))
        families_info.append({"name": fam_name, "count": len(coords_list), "smiles": smiles})

    if not structures:
        print("No structures to analyze after filtering.")
        return

    # ----------------------------
    # Load centroids & species
    # ----------------------------
    centroids = load_centroids_from_aligned_output(aligned_dir)

    # ----------------------------
    # Analysis loop
    # ----------------------------
    all_data, results = {}, {}

    for feature_name, feature_func in FEATURE_TYPES.items():
        print(f"\n=== Feature: {feature_name} ===")

        # Feature computation
        if feature_name in ["SOAP", "MBTR"]:
            X = np.vstack([feature_func(np.expand_dims(coords, 0), species)
                           for coords, species in zip(structures, species_list)])
            X_centroids = np.vstack([feature_func(np.expand_dims(c["coords"], 0), c["species"])
                                     for c in centroids]) if centroids else None
        else:
            X = np.vstack([feature_func(np.expand_dims(coords, 0)) for coords in structures])
            X_centroids = np.vstack([feature_func(np.expand_dims(c["coords"], 0)) for c in centroids]) if centroids else None

        result = {"filenames": names, "family_idx": family_idx, "families": families_info, "centroids": {}}
        results[feature_name] = {}

        for method in METHODS:
            print(f"  {method}")
            results[feature_name][method] = {}

            for n_dim in target_dims:
                if n_dim > X.shape[1]:
                    continue
                if method == "tsne" and n_dim > 3:
                    print(f"    Skipping TSNE for {n_dim} dimensions (Barnes-Hut limit)")
                    continue

                try:
                    X_reduced = reduce_features(X, method, n_components=n_dim)
                    metrics = full_embedding_analysis(X, X_reduced)

                    # ----------------------------
                    # Compute stress
                    # ----------------------------
                    D_high = squareform(pdist(X, metric='euclidean'))
                    D_low = squareform(pdist(X_reduced, metric='euclidean'))
                    metrics['stress'] = compute_stress(D_low, D_high)

                except Exception as e:
                    print(f"    Warning: Could not compute {method} for {n_dim} dims: {e}")
                    metrics = {m: float("nan") for m in expected_metrics}

                # Ensure all metrics are present
                for m in expected_metrics:
                    metrics.setdefault(m, float("nan"))

                results[feature_name][method][n_dim] = metrics

            # 2D embeddings for plotting centroids
            if X_centroids is not None:
                X_full = np.vstack([X, X_centroids])
                X_full_reduced = reduce_features(X_full, method, n_components=2)
                n_data = X.shape[0]
                centroid_coords = X_full_reduced[n_data:]
                centroid_list = []
                for j, c in enumerate(centroids):
                    centroid_list.append({
                        "x": float(centroid_coords[j][0]),
                        "y": float(centroid_coords[j][1]),
                        "name": c["name"],
                        "idx": family_idx[j] if j < len(family_idx) else None,
                        "is_meci": c["is_meci"]
                    })
                result["centroids"][method] = centroid_list
                result[method] = X_full_reduced[:n_data].tolist()

        # Save plots
        for method in METHODS:
            save_static_plots(
                result,
                method,
                feature_name,
                "filtered" if filter_outliers else "no_filter",
                output_dir,
                use_smiles=use_smiles_in_legend
            )

        all_data[f"{'filtered' if filter_outliers else 'no_filter'}_{feature_name}"] = result

    # ----------------------------
    # Save CSV, heatmaps, HTML
    # ----------------------------
    generate_html(convert_numpy(all_data), output_dir / "explorer.html")
    results_df = results_dict_to_df(results)
    results_df.to_csv(output_dir / "embedding_analysis_results.csv", index=False)
    print(f"Saved metrics CSV: {output_dir / 'embedding_analysis_results.csv'}")

    heatmap_folder = output_dir / "plots" / "heatmaps"
    heatmap_folder.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = ["trustworthiness", "continuity", "pearson_dist_corr", "spearman_dist_corr", "stress"]
    for metric in metrics_to_plot:
        plot_metrics_heatmap(results_df, metric, save_folder=heatmap_folder)

    print("Finished analysis")

from sklearn.metrics import pairwise_distances



def compare_feature_distances_to_precomputed_metrics(
    aligned_dir: Path,
    precomputed_distance_file: Path,
    families_to_include=None,
    feature_types=None,
    distance_metric="euclidean",
    output_dir: Path = None,
):
    """
    Compare feature-space distance matrices to a precomputed distance matrix,
    computing trustworthiness, continuity, Pearson, Spearman, MSE, Frobenius norm, and stress.
    """

    if feature_types is None:
        feature_types = ["SOAP", "inv_eigenval", "inverse_dist_matrix", "flatten_cartesian"]

    FEATURE_FUNCS = {
        "SOAP": coords_to_soap,
        "inv_eigenval": coords_to_inverse_eigenvalues,
        "inverse_dist_matrix": coords_to_inverse_distance_matrix,
        "flatten_cartesian": coords_to_flat_cartesian,
    }

    # ----------------------------
    # Load precomputed matrix
    # ----------------------------
    df = pd.read_csv(precomputed_distance_file, index_col=0)
    precomputed_labels_raw = df.index.tolist()
    precomputed_labels = [Path(x).stem for x in precomputed_labels_raw]
    D_pre = df.values

    # ----------------------------
    # Discover families
    # ----------------------------
    all_families = sorted([f.name for f in aligned_dir.glob("family_*")])
    if families_to_include:
        families = [f for f in all_families if f in families_to_include]
    else:
        families = all_families

    if not families:
        print("No families to analyze")
        return

    # ----------------------------
    # Load structures
    # ----------------------------
    structures, species_list, names = [], [], []
    for fam_name in families:
        fam_path = aligned_dir / fam_name
        coords_list, species_sublist, filenames, rmsds, smiles, n_filtered = \
            load_family_geometries(fam_path)
        if len(coords_list) == 0:
            continue
        structures.extend(coords_list)
        species_list.extend(species_sublist)
        names.extend([Path(f).stem for f in filenames])

    if not structures:
        print("No structures to analyze.")
        return

    # ----------------------------
    # Find intersection of labels
    # ----------------------------
    name_to_idx = {name: i for i, name in enumerate(names)}
    common_labels = [l for l in precomputed_labels if l in name_to_idx]

    if len(common_labels) == 0:
        raise ValueError("No overlapping structures between datasets.")

    print(f"Matched {len(common_labels)} structures (loaded={len(names)}, precomputed={len(precomputed_labels)})")

    # indices in structures
    structure_indices = [name_to_idx[l] for l in common_labels]
    precomputed_indices = [precomputed_labels.index(l) for l in common_labels]

    # reorder structures
    structures_ordered = [structures[i] for i in structure_indices]
    species_ordered = [species_list[i] for i in structure_indices]
    names_ordered = [names[i] for i in structure_indices]

    # filter precomputed matrix
    D_pre = D_pre[np.ix_(precomputed_indices, precomputed_indices)]

    n_structures = len(structures_ordered)
    results = []

    # ----------------------------
    # Compute metrics
    # ----------------------------
    for feature_name in feature_types:
        feature_func = FEATURE_FUNCS[feature_name]
        print(f"Computing feature: {feature_name}")

        # --------------------------------
        # Build feature representation
        # --------------------------------
        if feature_name == "flatten_cartesian":
            X = np.vstack([coords_to_flat_cartesian(np.expand_dims(coords, 0)) for coords in structures_ordered])
            D_feat = np.zeros((n_structures, n_structures))
            for i in range(n_structures):
                for j in range(i + 1, n_structures):
                    d = weighted_rmsd(
                        structures_ordered[i],
                        structures_ordered[j],
                        atoms1=species_ordered[i],
                        atoms2=species_ordered[j],
                        use_all_atoms=True,
                    )
                    D_feat[i, j] = d
                    D_feat[j, i] = d
        else:
            if feature_name in ["SOAP", "MBTR"]:
                X = np.vstack([feature_func(np.expand_dims(coords, 0), species)
                               for coords, species in zip(structures_ordered, species_ordered)])
            else:
                X = np.vstack([feature_func(np.expand_dims(coords, 0)) for coords in structures_ordered])
            D_feat = squareform(pdist(X, metric=distance_metric))

        # --------------------------------
        # Flatten upper triangle
        # --------------------------------
        triu_idx = np.triu_indices_from(D_feat, k=1)
        D_flat = D_feat[triu_idx]
        D_pre_flat = D_pre[triu_idx]

        # --------------------------------
        # Correlations
        # --------------------------------
        pearson_corr = pearsonr(D_flat, D_pre_flat)[0]
        spearman_corr = spearmanr(D_flat, D_pre_flat)[0]

        # --------------------------------
        # Error metrics
        # --------------------------------
        mse = np.mean((D_flat - D_pre_flat) ** 2)
        frob_norm = np.linalg.norm(D_feat - D_pre, ord="fro")
        stress = compute_stress(D_feat, D_pre)

        # --------------------------------
        # Trustworthiness
        # --------------------------------
        try:
            tw = trustworthiness(X, D_pre, n_neighbors=min(10, len(X)-1))
        except Exception:
            tw = np.nan

        # --------------------------------
        # Continuity
        # --------------------------------
        try:
            ranks_X = np.argsort(np.argsort(pairwise_distances(X), axis=1), axis=1)
            ranks_Y = np.argsort(np.argsort(D_pre), axis=1)
            n = X.shape[0]
            k = min(10, n - 1)
            continuity_sum = 0
            for i in range(n):
                U_i = set(np.where(ranks_Y[i] < k)[0])
                V_i = set(np.where(ranks_X[i] < k)[0])
                continuity_sum += len(U_i - V_i)
            continuity = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * continuity_sum
        except Exception:
            continuity = np.nan

        results.append({
            "feature": feature_name,
            "trustworthiness": tw,
            "continuity": continuity,
            "pearson_dist_corr": pearson_corr,
            "spearman_dist_corr": spearman_corr,
            "mse": mse,
            "frobenius_norm": frob_norm,
            "stress": stress,
        })

    results_df = pd.DataFrame(results)

    # ----------------------------
    # Save results
    # ----------------------------
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_csv = output_dir / "distance_comparison_metrics.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"Saved metrics to {results_csv}")

    return results_df