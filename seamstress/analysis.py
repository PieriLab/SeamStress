"""Dimensionality reduction analysis module."""

import json
import re
from pathlib import Path

import numpy as np
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

FAMILY_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def load_centroids_from_aligned_output(aligned_dir: Path) -> dict:
    """
    Load reference structures from aligned_output/family_N/reference.xyz files.

    Args:
        aligned_dir: Directory containing family subdirectories with reference.xyz files

    Returns:
        Dictionary mapping SMILES to centroid info (coords, name)
    """
    centroids = {}
    if not aligned_dir.exists():
        return centroids

    for family_dir in sorted(aligned_dir.glob("family_*")):
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
        centroids[smiles] = {'coords': np.array(coords), 'name': family_dir.name}

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
    for fname, info in family_info.items():
        if info['smiles'] in centroids:
            cent_coords = centroids[info['smiles']]['coords'].reshape(1, -1, 3)
            cent_feat = feature_func(cent_coords)
            centroid_info.append((info['data_idx'], cent_feat, info['display_name']))

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
            for i, (data_idx, _, name) in enumerate(centroid_info):
                result['centroids'][method].append({
                    'idx': data_idx,
                    'name': name,
                    'x': float(data[n_total + i, 0]),
                    'y': float(data[n_total + i, 1]),
                })

    return result


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
    families_to_include: list[str] | None = None
):
    """
    Run dimensionality reduction analysis on aligned geometries.

    Args:
        aligned_dir: Directory containing aligned_output/family_N/ folders
        output_dir: Directory to write analysis results
        families_to_include: Optional list of family names to include (e.g., ['family_1', 'family_2'])
                             If None, all families are included
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
        print(f"  Centroids will be matched to families by SMILES automatically:")
        for smiles, info in centroids.items():
            display = get_display_name(smiles, smiles)
            print(f"    {info['name']}: {display}")
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

    output_dir.mkdir(parents=True, exist_ok=True)
    generate_html(all_data, output_dir / "explorer.html")

    print("\n" + "=" * 70)
    print(f"Done! Open {output_dir / 'explorer.html'} in your browser")
    print("=" * 70)
