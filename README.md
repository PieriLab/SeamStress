# SeamStress

Molecular geometry alignment and dimensionality reduction analysis for photochemical reaction dynamics.

## What it does

1. **Groups** molecular geometries by connectivity (SMILES)
2. **Aligns** each geometry to a reference using Kabsch algorithm with optimal atom permutation
3. **Reduces** dimensionality (PCA, t-SNE, UMAP, Diffusion Map) for visualization
4. **Generates** an interactive HTML dashboard to explore the conformational landscape

## Setup

```bash
uv sync
```

## Quick Start

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

## Alignment Modes

SeamStress supports two alignment workflows:

### Mode 1: Multi-Family Alignment (Default)

Automatically detects families by connectivity (SMILES) and aligns each family independently.

**Workflow:**
1. Read all XYZ files from input directory
2. Compute connectivity hash (SMILES) for each geometry
3. Group geometries into families by connectivity
4. Align each family to its centroid reference
5. Align all family centroids to master reference (largest family or user-specified)
6. Generate aligned output with family folders

**Output structure:**
```
aligned_output/
├── family_1/
│   ├── reference.xyz          # Centroid for this family
│   ├── spawn_001.xyz          # Aligned individual geometries
│   ├── spawn_002.xyz
│   └── family_1.xyz           # Combined file (all spawns)
├── family_2/
│   ├── reference.xyz
│   └── ...
└── aligned_spawns.xyz         # All families combined
```

**Analysis visualization:**
- Each family centroid plotted as a **star** (★)
- Individual spawns plotted as **dots** (●)

**Example:**
```bash
seamstress -f ./data/spawns -o ./output -c ./centroids --analyze
```

### Mode 2: Align-All-to-Centroid

Forces all geometries to align to a single reference centroid, bypassing family detection. Useful when all spawning points have the same connectivity.

**Workflow:**
1. Read all XYZ files from input directory
2. Load specified centroid reference from centroids directory
3. Align ALL spawning points to this single reference (as one family)
4. Align ALL centroids to this same reference for visualization
5. Generate aligned output in `family_1/` folder

**Output structure:**
```
aligned_output/
├── family_1/
│   ├── reference.xyz          # The specified centroid reference
│   ├── centroids.xyz          # ALL aligned centroids (multi-frame)
│   ├── spawn_001.xyz          # Aligned individual spawning points
│   ├── spawn_002.xyz
│   └── family_1.xyz           # Combined file (all spawns)
└── aligned_spawns.xyz         # All spawns combined
```

**Analysis visualization:**
- ALL aligned centroids plotted as **stars** (★)
- Individual spawns plotted as **dots** (●)

**Example:**
```bash
seamstress -f ./benzene_spawns -o ./output -c ./centroids \
  --align-all-to-centroid benzene.xyz --analyze
```

**When to use:**

- All spawning points have the same connectivity
- Want to compare different centroid structures in visualization
- Studying single-molecule conformational dynamics

## Command-Line Options

### Core Options

**`-f, --folder PATH`** (required)

- Folder containing XYZ files to analyze

**`-o, --output PATH`** (required)

- Output folder for aligned geometry files

**`-c, --centroids PATH`**

- Folder containing reference/centroid structures (XYZ files)
- Required for alignment modes

### Alignment Mode Selection

**`--align-all-to-centroid FILENAME`**

- Align ALL spawning points to a single centroid, bypassing family detection
- Example: `--align-all-to-centroid benzene.xyz`
- Requires `-c/--centroids` to specify centroid folder
- Treats all geometries as one family regardless of connectivity
- Warns if mean RMSD > 1.0 Å

**`--master-reference FILENAME`**

- Filename of centroid to use as master reference for inter-family alignment
- Example: `--master-reference ethylene.xyz`
- Only used in multi-family mode (default)
- If not specified, largest family (Family 1) is used as master

### Heavy Atom Weighting

**`--inter-family-heavy-atom-factor FACTOR`** (default: 1.0)

- Weight multiplier for heavy atoms when aligning family centroids to each other
- Only used in multi-family mode
- Use larger values (e.g., 10.0, 100.0) to prioritize heavy atoms
- Helpful when hydrogens cause centroid alignment orientation issues

**`--intra-family-heavy-atom-factor FACTOR`** (default: 1.0)

- Weight multiplier for heavy atoms when aligning molecules to their family reference
- Applied AFTER best permutation is found
- Works in both multi-family and align-all-to-centroid modes
- Use larger values (e.g., 10.0, 100.0) to prioritize heavy atoms in final alignment

### Permutation Search Options

**`--fragment-permutations`**

- Use fragment-based permutation search (treats heavy atoms + bonded H as rigid units)
- Only applicable when all heavy atoms have exactly 1 hydrogen (e.g., benzene)
- Provides ~720x speedup for benzene-like molecules
- Automatically falls back to standard mode if not applicable
- Recommended for benzene, pyridine, and similar aromatic systems

**`--no-permutations`**

- Disable permutation search (use identity permutation only)
- Much faster but may produce suboptimal alignments
- Useful for large molecules where permutation search is prohibitively expensive

### Analysis Options

**`--analyze`**

- Run dimensionality reduction analysis after alignment
- Generates interactive HTML dashboard with PCA, t-SNE, UMAP, and Diffusion Map embeddings
- Dashboard includes both Cartesian and internal coordinate representations

**`--analysis-output PATH`**

- Output directory for analysis results
- Default: `<output>/analysis`

### Advanced Options

**`--prealign-centroids-to FILENAME`**

- Pre-align all centroids to this centroid before main workflow
- Example: `--prealign-centroids-to benzene.xyz`
- Optional first step that aligns all centroid structures to a user-specified reference
- Aligned centroids saved to `output/prealigned_centroids/`

**`--no-connectivity`**

- Disable connectivity analysis (just display geometries)
- Skips SMILES computation and family detection

**`--no-automorphisms`**

- Disable automorphism computation
- Only show connectivity groups without detailed symmetry analysis

## Usage Examples

### Multi-family alignment with analysis

```bash
seamstress -f ./data/spawns -o ./results -c ./centroids --analyze
```

### Benzene spawning points with fast fragment permutations

```bash
seamstress -f ./benzene_spawns -o ./output -c ./centroids \
  --align-all-to-centroid benzene.xyz \
  --fragment-permutations \
  --analyze
```

### Heavy atom weighting for better alignment

```bash
seamstress -f ./data -o ./output -c ./centroids \
  --inter-family-heavy-atom-factor 10.0 \
  --intra-family-heavy-atom-factor 5.0 \
  --analyze
```

### Fast mode for large molecules (no permutation search)

```bash
seamstress -f ./large_molecules -o ./output -c ./centroids \
  --no-permutations \
  --analyze
```

## Documentation

Full algorithm description: <https://pierilab.github.io/SeamStress/>

To update docs locally:

```bash
uv run mkdocs serve        # Preview at http://127.0.0.1:8000
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
