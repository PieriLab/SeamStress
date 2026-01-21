# SeamStress

Molecular geometry alignment and dimensionality reduction analysis.

## Quick Start

```bash
pip install -e .
```

## Usage

### 1. Align geometries

Place your XYZ files in `data/spawns/` and reference centroid structures in `centroids/`, then run:

```bash
python main.py
```

This reads XYZ files, groups molecules by connectivity (SMILES), and aligns each geometry to its family centroid using the Kabsch algorithm with optimal atom permutation. Output goes to `aligned_output/`.

### 2. Generate interactive analysis

```bash
python analyze_all.py
```

This creates `analysis_output/explorer.html` - an interactive dashboard with:

- **Threshold filter**: No filter / Max 5.0 Å pairwise distance
- **Feature types**: Aligned Cartesian coordinates / Inverse distance matrix (1/r)
- **Dimensionality reduction**: PCA, t-SNE, UMAP, Diffusion Map

Open `explorer.html` in your browser. Hover over points to see the xyz filename.

## Project Structure

```
seamstress/         # Core package (alignment, connectivity, I/O)
centroids/          # Reference structures for alignment
data/spawns/        # Input XYZ files
aligned_output/     # Aligned geometries grouped by family
analysis_output/    # Interactive HTML dashboard
main.py             # Run alignment
analyze_all.py      # Generate dimensionality reduction dashboard
```

## Next Steps

See the [Algorithm](algorithm.md) page for a detailed explanation of the pipeline.
