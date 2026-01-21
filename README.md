# SeamStress

Molecular geometry alignment and dimensionality reduction analysis for photochemical reaction dynamics.

## What it does

1. **Groups** molecular geometries by connectivity (SMILES)
2. **Aligns** each geometry to a reference using Kabsch algorithm with optimal atom permutation
3. **Reduces** dimensionality (PCA, t-SNE, UMAP, Diffusion Map) for visualization
4. **Generates** an interactive HTML dashboard to explore the conformational landscape

## Setup

```bash
pip install -e .
```

## Usage

### 1. Align geometries

Place XYZ files in `data/spawns/` and reference structures in `centroids/`, then:

```bash
python main.py
```

Output: `aligned_output/` with geometries grouped by family.

### 2. Generate dashboard

```bash
python analyze_all.py
```

Output: `analysis_output/explorer.html` - open in browser, hover points to see xyz filenames.

## Documentation

Full algorithm description: https://pierilab.github.io/SeamStress/

To update docs locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve        # Preview at http://127.0.0.1:8000
```

Docs auto-deploy to GitHub Pages when you push changes to `docs/` or `mkdocs.yml`.

## Project structure

```
seamstress/         # Core package
centroids/          # Reference structures
data/spawns/        # Input XYZ files
docs/               # Documentation (MkDocs)
main.py             # Alignment entry point
analyze_all.py      # Dashboard generator
```
