# ELISA SPAWNS

A Python tool for analyzing molecular geometries from XYZ files. It groups molecules by connectivity patterns and computes symmetry properties (automorphisms).

## Features

- **Read XYZ files**: Parse standard XYZ format molecular geometry files
- **Connectivity analysis**: Automatically determine chemical bonds from 3D coordinates
- **Family grouping**: Group molecules by their connectivity patterns (bond topology)
- **Symmetry detection**: Compute automorphisms (symmetry mappings) for molecules
- **RMSD calculations**: Find best atom mappings between similar molecules

## Installation

### Quick Install (pip)

The easiest way to install and use ELISA SPAWNS is with pip:

```bash
# Install directly from the package directory
pip install .

# Or install in editable mode for development
pip install -e .
```

After installation, the `elisa-spawns` command will be available globally.

### Alternative: Setup with uv

If you prefer using the `uv` package manager:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone or download this repository
cd ELISA_SPAWNS

# Install the package in editable mode
uv pip install -e .
```

### Prerequisites

- Python 3.12 or higher
- pip or [uv](https://docs.astral.sh/uv/) package manager

## Usage

### Command-Line Interface

Once installed, you can use the `elisa-spawns` command from anywhere:

```bash
# Basic usage - analyze and create aligned geometries
elisa-spawns -f /path/to/xyz/files -o /path/to/output/folder

# Disable automorphism computation (faster, no aligned files)
elisa-spawns -f ./molecules -o ./output --no-automorphisms

# Just display geometries without connectivity analysis
elisa-spawns -f ./molecules -o ./output --no-connectivity
```

### Command-Line Options

```
-f, --folder          (Required) Folder containing XYZ files to analyze
-o, --output          (Required) Output folder for aligned/swapped geometry files
--no-connectivity     Disable connectivity analysis
--no-automorphisms    Disable automorphism computation
-h, --help            Show help message
```

### What Gets Created

When you run `elisa-spawns -f input/ -o output/`, the tool:

1. **Prints analysis to terminal** (stdout):
   - Number of files found
   - Connectivity families discovered
   - Automorphism symmetries for each family
   - Progress as files are written

2. **Creates output folder** with one aligned file per input (same filenames):
   ```
   output/
   ├── benzene_1.xyz        # Family 1 | Reference | ...
   ├── benzene_2.xyz        # Family 1 | RMSD: 0.0234 | ...
   ├── benzene_3.xyz        # Family 1 | RMSD: 0.0156 | ...
   ├── acetic_acid_1.xyz    # Family 2 | Reference | ...
   ├── acetic_acid_2.xyz    # Family 2 | RMSD: 0.0089 | ...
   └── ...
   ```

**What each file contains:**
- **Original filename preserved**
- **Comment line** (title): Family number, RMSD (if aligned), original metadata
- **Reference molecules** (first in each family): Original geometry, unchanged
- **Other molecules**: Atoms reordered to match reference + optimal rotation/translation (Kabsch alignment)

### Prepare Your Data

Place your XYZ files in any directory, for example:
```
my_molecules/
├── molecule1.xyz
├── molecule2.xyz
└── molecule3.xyz
```

Then run:
```bash
elisa-spawns -f my_molecules -o results
```

### XYZ File Format

Your XYZ files should follow the standard format:

```
<number of atoms>
<comment line / metadata>
<element> <x> <y> <z>
<element> <x> <y> <z>
...
```

Example (`water.xyz`):
```
3
Water molecule
O    0.000000    0.000000    0.117790
H    0.000000    0.755453   -0.471161
H    0.000000   -0.755453   -0.471161
```

### Example Output

```
Reading geometries from: /path/to/data/spawns
============================================================

Found 15 geometry files

Analyzing connectivity patterns...

Found 3 unique connectivity patterns:

======================================================================

Group 1: 8 molecules
  Connectivity: C1=CC=CC=C1
  Example files: benzene_1.xyz, benzene_2.xyz, benzene_3.xyz
  ... and 5 more

Group 2: 5 molecules
  Connectivity: CC(=O)O
  Example files: acetic_acid_1.xyz, acetic_acid_2.xyz, acetic_acid_3.xyz
  ... and 2 more

Group 3: 2 molecules
  Connectivity: O
  Example files: water_1.xyz, water_2.xyz


================================================================================
AUTOMORPHISMS FOR TEMPLATE MEMBERS OF EACH FAMILY
================================================================================


Family 1: C1=CC=CC=C1 (8 molecules)

Template: benzene_1.xyz
================================================================================
Atoms: C C C C C C H H H H H H
Number of automorphisms: 12
================================================================================

Automorphism tuples:
[
  (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
  (1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6),
  ...
]
```

## Code Structure

```
elisa_spawns/
├── __init__.py           # Package initialization
├── cli.py                # Command-line interface
├── geometry.py           # Geometry data structure and XYZ reading
├── connectivity.py       # Connectivity analysis and grouping
├── automorphism.py       # Symmetry and atom mapping calculations
├── rdkit_utils.py        # Conversion to RDKit molecule format
├── alignment.py          # Kabsch alignment and RMSD calculations
├── io_utils.py           # XYZ file writing utilities
└── processor.py          # Main processing pipeline
```

## Advanced Usage

### Use as a Python Library

```python
from pathlib import Path
from elisa_spawns.geometry import read_xyz_file, read_all_geometries
from elisa_spawns.connectivity import analyze_connectivity, group_by_connectivity
from elisa_spawns.automorphism import get_automorphisms
from elisa_spawns.rdkit_utils import geometry_to_mol

# Read a single file
geom = read_xyz_file("molecule.xyz")
print(f"Atoms: {geom.atoms}")
print(f"Coordinates shape: {geom.coordinates.shape}")

# Analyze connectivity
conn_info = analyze_connectivity(geom)
print(f"Connectivity: {conn_info.connectivity_hash}")

# Convert to RDKit and get symmetries
mol = geometry_to_mol(geom)
automorphisms = get_automorphisms(mol)
print(f"Number of symmetries: {len(automorphisms)}")
```

## How It Works

1. **XYZ Parsing**: Reads atomic symbols and 3D coordinates from files
2. **Bond Inference**: Uses RDKit's `DetermineBonds` to infer chemical bonds from 3D geometry
3. **Connectivity Signature**: Generates canonical SMILES (ignoring stereochemistry) as a unique identifier
4. **Grouping**: Groups molecules with identical connectivity patterns
5. **Automorphism Detection**: Uses RDKit substructure matching to find all symmetry operations
6. **Atom Reordering**: For each molecule, tries all automorphism permutations to find optimal atom correspondence
7. **Kabsch Alignment**: Applies optimal rotation and translation to minimize RMSD between structures

## Contributing

Feel free to extend this code! Some ideas:
- Add molecular alignment algorithms (see `alignment.py`)
- Export results to different formats (JSON, CSV)
- Visualize molecules using RDKit's drawing tools
- Add energy or property calculations
