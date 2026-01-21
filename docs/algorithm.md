# Algorithm Overview

This document describes the pipeline from raw molecular geometries to clustered dimensionality reduction maps.

## Pipeline Summary

```
XYZ files → Connectivity Analysis → Family Grouping → Alignment → Feature Extraction → Dim. Reduction → Visualization
```

---

## Step 1: Parse Molecular Geometries

```
INPUT: XYZ files (one per geometry)

FOR each xyz_file:
    n_atoms ← first line
    coords ← parse atom positions (n_atoms × 3 matrix)
    elements ← parse atom symbols
    STORE (elements, coords)
```

Each XYZ file contains one molecular snapshot with atomic coordinates.

---

## Step 2: Connectivity Analysis

```
FOR each geometry (elements, coords):
    mol ← build_molecule(elements, coords)

    # Infer bonds from distances
    FOR each atom pair (i, j):
        IF distance(i, j) < covalent_threshold:
            add_bond(mol, i, j)

    smiles ← canonical_smiles(mol)
    family[smiles].append(geometry)
```

The SMILES string encodes the molecular connectivity (which atoms are bonded). Geometries with the same SMILES belong to the same **family**.

---

## Step 3: Alignment (Kabsch with Permutation)

For each family, align all geometries to a reference centroid structure.

```
FOR each family:
    centroid ← load_reference_structure(family)

    FOR each geometry in family:
        best_rmsd ← infinity
        best_aligned ← None

        # Try all permutations of equivalent atoms
        FOR each permutation P of equivalent atoms:
            permuted_coords ← apply_permutation(coords, P)

            # Kabsch algorithm
            centered_ref ← centroid - centroid_mean
            centered_geo ← permuted_coords - permuted_mean

            H ← centered_geo.T @ centered_ref
            U, S, Vt ← SVD(H)
            R ← Vt.T @ U.T  # optimal rotation

            aligned ← (permuted_coords - mean) @ R + centroid_mean
            rmsd ← sqrt(mean(||aligned - centroid||²))

            IF rmsd < best_rmsd:
                best_rmsd ← rmsd
                best_aligned ← aligned

        STORE (best_aligned, best_rmsd)
```

The permutation search handles atom equivalence (e.g., swapping two hydrogens on the same carbon).

---

## Step 4: Feature Extraction

Two feature representations are available:

### Option A: Aligned Cartesian Coordinates

```
FOR each aligned geometry (n_atoms × 3):
    features ← flatten(coords)  # → vector of length 3×n_atoms
```

For ethylene (6 atoms): 18-dimensional feature vector.

### Option B: Inverse Distance Matrix

```
FOR each aligned geometry:
    features ← []
    FOR each atom pair (i, j) where i < j:
        r ← distance(atom_i, atom_j)
        features.append(1/r)

    features ← clip(features, 0, 100)  # handle near-zero distances
```

For ethylene (6 atoms): 15-dimensional feature vector (6×5/2 pairs).

---

## Step 5: Filtering (Optional)

Remove "exploded" geometries where atoms have separated too far.

```
threshold ← 5.0 Å  # max allowed pairwise distance

FOR each geometry:
    max_dist ← max(pairwise_distances(coords))
    IF max_dist > threshold:
        DISCARD geometry
```

---

## Step 6: Dimensionality Reduction

Reduce high-dimensional features to 2D for visualization.

```
# Preprocessing
scaler ← fit(all_features)  # zero mean, unit variance
scaled_features ← scaler.transform(features)

# Choose one method:
embedding ← PCA(scaled_features, n_components=2)
         OR TSNE(scaled_features, n_components=2)
         OR UMAP(scaled_features, n_components=2)
         OR DiffusionMap(scaled_features, n_components=3)[1:]  # skip trivial component
```

### Method Notes

| Method | Preserves | Best For |
|--------|-----------|----------|
| PCA | Global variance | Linear relationships |
| t-SNE | Local neighborhoods | Cluster separation |
| UMAP | Local + some global | General purpose |
| Diffusion Map | Intrinsic geometry | Reaction pathways |

---

## Step 7: Visualization

```
FOR each family:
    color ← family_color_map[family]
    PLOT points (embedding[:, 0], embedding[:, 1]) with color

    IF centroid exists:
        centroid_embedding ← reduce(centroid_features)
        PLOT star marker at centroid_embedding

ENABLE hover tooltips showing xyz filename
```

---

## Output

Interactive HTML dashboard where:

- Each point = one molecular geometry
- Color = molecular family (connectivity pattern)
- Star markers = reference centroid structures
- Hover = xyz filename for inspection
- Dropdowns = switch between threshold/features/method
