"""Dimensionality reduction analysis module."""

import colorsys
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from umap import UMAP

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
    Load reference structures from aligned_output/family_N/ folders.

    Looks for aligned_centroids.xyz at root (multi-family mode),
    or centroids.xyz in family_1 (align-all mode),
    or falls back to reference.xyz per family.

    Args:
        aligned_dir: Directory containing family subdirectories

    Returns:
        List of dictionaries with keys: coords, name, smiles, family_name, is_meci, meci_number
    """
    centroids = []
    if not aligned_dir.exists():
        return centroids

    # First check for aligned_centroids.xyz at root (multi-family mode with MECIs)
    aligned_centroids_file = aligned_dir / "aligned_centroids.xyz"
    if aligned_centroids_file.exists():
        with open(aligned_centroids_file) as f:
            lines = f.readlines()

        # Parse multi-frame XYZ file
        idx = 0
        meci_num = 1
        family_name_to_idx = {}  # Track which family each centroid belongs to

        while idx < len(lines):
            n_atoms = int(lines[idx].strip())
            header = lines[idx + 1].strip()

            # Extract family name and SMILES from header like "Family_1 [H+].[H][C-]=C([H])[H] | Aligned Reference"
            family_match = re.search(r"(Family_\d+)\s+(\S+)", header)
            if family_match:
                family_name = family_match.group(1).lower()  # family_1, family_2, etc.
                smiles = family_match.group(2)
            else:
                family_name = "unknown"
                smiles = "Unknown"

            coords = []
            for i in range(idx + 2, idx + 2 + n_atoms):
                parts = lines[i].split()
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

            # Only mark as MECI if SMILES matches one from raw_centroids
            is_meci = smiles in MECI_SMILES

            centroids.append({
                'coords': np.array(coords),
                'name': f"{family_name}_meci{meci_num}" if is_meci else f"{family_name}",
                'smiles': smiles,
                'family_name': family_name,
                'is_meci': is_meci,  # Only true for actual MECIs
                'meci_number': meci_num if is_meci else None
            })

            idx += 2 + n_atoms
            if is_meci:
                meci_num += 1  # Only increment for actual MECIs

        return centroids  # Return early if we found aligned_centroids.xyz

    # Otherwise check individual family folders
    for family_dir in sorted(aligned_dir.glob("family_*")):
        # First try to load centroids.xyz (multiple centroids - these are MECIs)
        centroids_file = family_dir / "centroids.xyz"
        if centroids_file.exists():
            with open(centroids_file) as f:
                lines = f.readlines()

            # Parse multi-frame XYZ file
            idx = 0
            centroid_num = 1
            while idx < len(lines):
                n_atoms = int(lines[idx].strip())
                header = lines[idx + 1].strip()

                smiles_match = re.search(r"Family_\d+\s+(\S+)", header)
                smiles = smiles_match.group(1) if smiles_match else "Unknown"

                coords = []
                for i in range(idx + 2, idx + 2 + n_atoms):
                    parts = lines[i].split()
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

                centroids.append({
                    'coords': np.array(coords),
                    'name': f"{family_dir.name}_centroid{centroid_num}",
                    'smiles': smiles,
                    'family_name': family_dir.name,
                    'is_meci': True,  # These are MECIs from centroids.xyz
                    'meci_number': centroid_num
                })

                idx += 2 + n_atoms
                centroid_num += 1
            continue

        # Fall back to reference.xyz (single centroid - NOT a MECI)
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
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        centroids.append({
            'coords': np.array(coords),
            'name': family_dir.name,
            'smiles': smiles,
            'family_name': family_dir.name,
            'is_meci': False,  # This is just a reference, not a MECI
            'meci_number': None
        })

    return centroids


def max_pairwise_distance(coords: np.ndarray) -> float:
    """Calculate maximum pairwise distance in a geometry."""
    return pdist(coords).max()


def load_family_geometries(family_dir: Path, max_distance_threshold: float | None = None):
    """Load all geometries from a family directory."""
    coords_list, filenames, rmsds = [], [], []
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
        for i in range(2, 2 + n_atoms):
            parts = lines[i].split()
            atoms_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        atoms_coords = np.array(atoms_coords)

        if max_distance_threshold is not None:
            if max_pairwise_distance(atoms_coords) > max_distance_threshold:
                n_filtered += 1
                continue

        coords_list.append(atoms_coords)
        filenames.append(xyz_file.stem)
        rmsds.append(rmsd)

    if len(coords_list) == 0:
        return np.array([]), filenames, rmsds, smiles, n_filtered
    return np.array(coords_list), filenames, rmsds, smiles, n_filtered


def get_display_name(smiles: str, family_name: str) -> str:
    """Get human-readable name for a SMILES string."""
    return SMILES_TO_NAME.get(smiles, smiles if smiles else family_name)


def coords_to_cartesian(coords: np.ndarray) -> np.ndarray:
    """Convert 3D coordinates to flattened Cartesian features."""
    return coords.reshape(coords.shape[0], -1)


def coords_to_inverse_distance(coords: np.ndarray) -> np.ndarray:
    """Convert 3D coordinates to inverse distance matrix features."""
    n_samples = coords.shape[0]
    features = np.array([1.0 / pdist(coords[i]) for i in range(n_samples)])
    return np.clip(features, 0, 100)


def run_pca(features, scaler, n_comp=2):
    """Run PCA dimensionality reduction."""
    scaled = scaler.transform(features)
    pca = PCA(n_components=min(n_comp, features.shape[1], features.shape[0]))
    return pca.fit_transform(scaled), pca


def run_tsne(features, scaler, n_comp=2):
    """Run t-SNE dimensionality reduction."""
    scaled = scaler.transform(features)
    perp = min(30.0, (features.shape[0] - 1) / 3)
    return TSNE(n_components=n_comp, perplexity=perp, random_state=42, max_iter=1000).fit_transform(scaled)


def run_umap(features, scaler, n_comp=2):
    """Run UMAP dimensionality reduction."""
    scaled = scaler.transform(features)
    n_neighbors = min(15, features.shape[0] - 1)
    return UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=n_comp, random_state=42).fit_transform(scaled)


def run_diffusion_map(features, scaler, n_comp=2):
    """Run Diffusion Map dimensionality reduction."""
    scaled = scaler.transform(features)
    n_neighbors = min(10, features.shape[0] - 1)
    emb = SpectralEmbedding(n_components=n_comp + 1, affinity='nearest_neighbors',
                            n_neighbors=n_neighbors, random_state=42)
    return emb.fit_transform(scaled)[:, 1:]


def compute_all_embeddings(aligned_dir, centroids, threshold, families, feature_name, feature_func):
    """Compute all embeddings and return data for JSON."""
    all_coords, all_family_idx, all_filenames, all_family_names = [], [], [], []
    families_with_data = []
    family_info = {}

    for fam_idx, family_name in enumerate(families):
        family_dir = aligned_dir / family_name
        if not family_dir.exists():
            continue
        coords, filenames, rmsds, smiles, n_filtered = load_family_geometries(family_dir, threshold)
        if len(coords) == 0:
            continue
        display_name = get_display_name(smiles, family_name)
        data_idx = len(families_with_data)
        families_with_data.append({'idx': fam_idx, 'name': display_name, 'smiles': smiles, 'count': len(coords)})
        family_info[family_name] = {'smiles': smiles, 'display_name': display_name, 'data_idx': data_idx}
        all_coords.append(coords)
        all_family_idx.extend([data_idx] * len(coords))
        all_filenames.extend([f"{family_name}/{f}.xyz" for f in filenames])
        all_family_names.extend([display_name] * len(coords))
        print(f"      {display_name}: {len(coords)}")

    if not all_coords:
        return None

    coords_combined = np.vstack(all_coords)
    features = feature_func(coords_combined)
    n_total = features.shape[0]
    print(f"      Total: {n_total} molecules, {features.shape[1]} features")

    # Centroids
    centroid_info = []
    for centroid in centroids:
        # Find matching family by family_name
        if centroid['family_name'] in family_info:
            info = family_info[centroid['family_name']]
            cent_coords = centroid['coords'].reshape(1, -1, 3)
            cent_feat = feature_func(cent_coords)
            centroid_info.append((
                info['data_idx'],
                cent_feat,
                centroid['name'],
                centroid.get('is_meci', False),
                centroid.get('meci_number', None)
            ))

    if centroid_info:
        centroid_features = np.vstack([c[1] for c in centroid_info])
        features_all = np.vstack([features, centroid_features])
        n_centroids = len(centroid_info)
    else:
        features_all = features
        n_centroids = 0

    scaler = StandardScaler()
    scaler.fit(features)

    print("      Computing PCA...")
    pca_all, _ = run_pca(features_all, scaler, 2)
    print("      Computing t-SNE...")
    tsne_all = run_tsne(features_all, scaler, 2)
    print("      Computing UMAP...")
    umap_all = run_umap(features_all, scaler, 2)
    print("      Computing Diffusion Map...")
    dm_all = run_diffusion_map(features_all, scaler, 2)

    # Build result
    result = {
        'families': families_with_data,
        'filenames': all_filenames,
        'family_idx': all_family_idx,
        'pca': pca_all[:n_total].tolist(),
        'tsne': tsne_all[:n_total].tolist(),
        'umap': umap_all[:n_total].tolist(),
        'dm': dm_all[:n_total].tolist(),
        'centroids': {},
    }

    if n_centroids > 0:
        for method, data in [('pca', pca_all), ('tsne', tsne_all), ('umap', umap_all), ('dm', dm_all)]:
            result['centroids'][method] = []
            for i, (data_idx, _, name, is_meci, meci_number) in enumerate(centroid_info):
                result['centroids'][method].append({
                    'idx': data_idx,
                    'name': name,
                    'x': float(data[n_total + i, 0]),
                    'y': float(data[n_total + i, 1]),
                    'is_meci': is_meci,
                    'meci_number': meci_number
                })

    return result


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

    # Plot centroids as stars (only MECIs)
    for centroid in centroids:
        # Only plot stars for MECIs (from centroids.xyz), not simple references
        if not centroid.get('is_meci', False):
            continue

        color = colors[centroid['idx']]
        meci_label = f"MECI{centroid.get('meci_number', '?')}"

        ax.scatter([centroid['x']], [centroid['y']],
                  marker='*', s=400, c=[color],
                  edgecolors='black', linewidths=1.2,
                  label=meci_label,
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


def generate_html(all_data, output_path):
    """Generate single HTML file with all data embedded."""
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>SeamStress - Dimensionality Reduction</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .controls { background: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .controls label { margin-right: 10px; font-weight: bold; }
        .controls select { padding: 8px 12px; font-size: 14px; margin-right: 20px; border-radius: 4px; border: 1px solid #ccc; }
        #plot { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .info { background: #e7f3ff; padding: 10px; border-radius: 4px; margin-bottom: 15px; }
    </style>
</head>
<body>
    <h1>SeamStress - Dimensionality Reduction Explorer</h1>
    <div class="info">Hover over points to see xyz filename. Click legend items to show/hide families.</div>
    <div class="controls">
        <label>Threshold:</label>
        <select id="threshold">
            <option value="no_filter">No Filter</option>
            <option value="max_5.0A">Max 5.0 A</option>
        </select>
        <label>Features:</label>
        <select id="features">
            <option value="cartesian_aligned">Cartesian (aligned)</option>
            <option value="inverse_distance">Inverse Distance (1/r)</option>
        </select>
        <label>Method:</label>
        <select id="method">
            <option value="pca">PCA</option>
            <option value="tsne">t-SNE</option>
            <option value="umap">UMAP</option>
            <option value="dm">Diffusion Map</option>
        </select>
    </div>
    <div id="plot"></div>

    <script>
    const DATA = ''' + json.dumps(all_data) + ''';
    const COLORS = ''' + json.dumps(FAMILY_COLORS) + ''';
    const AXIS_LABELS = {
        'pca': ['PC1', 'PC2'],
        'tsne': ['t-SNE1', 't-SNE2'],
        'umap': ['UMAP1', 'UMAP2'],
        'dm': ['DM2', 'DM3']
    };

    function updatePlot() {
        const threshold = document.getElementById('threshold').value;
        const features = document.getElementById('features').value;
        const method = document.getElementById('method').value;

        const key = threshold + '_' + features;
        const data = DATA[key];
        if (!data) {
            document.getElementById('plot').innerHTML = '<p>No data for this combination</p>';
            return;
        }

        const coords = data[method];
        const filenames = data.filenames;
        const familyIdx = data.family_idx;
        const families = data.families;
        const centroids = data.centroids[method] || [];

        const traces = [];

        // One trace per family
        families.forEach((fam, i) => {
            const mask = familyIdx.map((idx, j) => idx === i ? j : -1).filter(j => j >= 0);
            traces.push({
                x: mask.map(j => coords[j][0]),
                y: mask.map(j => coords[j][1]),
                mode: 'markers',
                type: 'scatter',
                name: fam.name + ' (' + fam.count + ')',
                text: mask.map(j => filenames[j]),
                hovertemplate: '%{text}<extra>' + fam.name + '</extra>',
                marker: { color: COLORS[i], size: 8, opacity: 0.6 }
            });
        });

        // Centroids
        centroids.forEach(c => {
            traces.push({
                x: [c.x],
                y: [c.y],
                mode: 'markers',
                type: 'scatter',
                name: 'Centroid: ' + c.name,
                text: ['CENTROID: ' + c.name],
                hovertemplate: '%{text}<extra></extra>',
                marker: { color: COLORS[c.idx], size: 24, symbol: 'star', line: { color: 'black', width: 2 } },
                showlegend: false
            });
        });

        const labels = AXIS_LABELS[method];
        const layout = {
            title: features.replace('_', ' ') + ' - ' + method.toUpperCase() + ' (' + coords.length + ' molecules)',
            xaxis: { title: labels[0] },
            yaxis: { title: labels[1] },
            height: 800,
            width: 1200,
            hovermode: 'closest',
            legend: { x: 1.02, y: 1 }
        };

        Plotly.newPlot('plot', traces, layout);
    }

    document.getElementById('threshold').addEventListener('change', updatePlot);
    document.getElementById('features').addEventListener('change', updatePlot);
    document.getElementById('method').addEventListener('change', updatePlot);

    updatePlot();
    </script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"\nSaved: {output_path}")


def run_analysis(
    aligned_dir: Path,
    output_dir: Path,
    families_to_include: list[str] | None = None,
    use_smiles_in_legend: bool = False
):
    """
    Run dimensionality reduction analysis on aligned geometries.

    Args:
        aligned_dir: Directory containing aligned_output/family_N/ folders
        output_dir: Directory to write analysis results
        families_to_include: Optional list of family names to include (e.g., ['family_1', 'family_2'])
                             If None, all families are included
        use_smiles_in_legend: If True, use SMILES strings in plot legends; if False, use display names
    """
    THRESHOLDS = [(None, "no_filter"), (5.0, "max_5.0A")]
    FEATURE_TYPES = [
        ("cartesian_aligned", coords_to_cartesian),
        ("inverse_distance", coords_to_inverse_distance)
    ]

    print("\nDimensionality Reduction Analysis")
    print("=" * 70)

    # Load reference structures
    centroids = load_centroids_from_aligned_output(aligned_dir)
    if centroids:
        print(f"Loaded {len(centroids)} reference structures from aligned_output/")
        print(f"  Centroids will be matched to families automatically:")
        for centroid in centroids:
            display = get_display_name(centroid['smiles'], centroid['family_name'])
            print(f"    {centroid['name']}: {display}")
    else:
        print("  No reference structures found (plots will not show centroids)")

    # Discover families dynamically
    all_families = sorted([f.name for f in aligned_dir.glob("family_*")])
    if not all_families:
        print(f"\nError: No family directories found in {aligned_dir}")
        return

    # Filter families if requested
    if families_to_include is not None:
        families = [f for f in all_families if f in families_to_include]
        excluded = set(all_families) - set(families)
        if excluded:
            print(f"\nExcluded families: {', '.join(sorted(excluded))}")
    else:
        families = all_families

    if not families:
        print("\nError: No families to analyze after filtering")
        return

    print(f"\nAnalyzing {len(families)} families: {', '.join(families)}")

    all_data = {}

    for threshold, thresh_name in THRESHOLDS:
        print(f"\n{'='*70}")
        print(f"Threshold: {thresh_name}")
        print(f"{'='*70}")

        for feature_name, feature_func in FEATURE_TYPES:
            print(f"\n  {feature_name}:")
            key = f"{thresh_name}_{feature_name}"
            result = compute_all_embeddings(
                aligned_dir, centroids, threshold, families, feature_name, feature_func
            )
            if result:
                all_data[key] = result

                # Save static plots for each method
                print(f"\n      Saving static plots:")
                for method in ['pca', 'tsne', 'umap', 'dm']:
                    save_static_plots(result, method, feature_name, thresh_name, output_dir, use_smiles_in_legend)

    output_dir.mkdir(parents=True, exist_ok=True)
    generate_html(all_data, output_dir / "explorer.html")

    print("\n" + "=" * 70)
    print(f"Done! Open {output_dir / 'explorer.html'} in your browser")
    print("=" * 70)
