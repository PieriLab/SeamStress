"""Utilities for converting geometries to RDKit molecules."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from elisa_spawns.geometry import Geometry

# Covalent radii in Angstroms (from Cordero et al. 2008)
COVALENT_RADII = {
    'H': 0.31,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'F': 0.57,
    'S': 1.05,
    'Cl': 1.02,
    'Br': 1.20,
    'I': 1.39,
}


def add_bonds_by_distance(
    mol: Chem.RWMol,
    coords: np.ndarray,
    atoms: list[str],
    cov_factor: float = 1.2
) -> None:
    """
    Add bonds to molecule based strictly on interatomic distances.

    A bond is added if distance < (cov_radius_1 + cov_radius_2) * cov_factor.

    Args:
        mol: RDKit RWMol to add bonds to
        coords: Atomic coordinates (N, 3)
        atoms: List of atom symbols
        cov_factor: Multiplier for covalent radii sum (default: 1.2)
    """
    n_atoms = len(atoms)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Get covalent radii
            r1 = COVALENT_RADII.get(atoms[i], 1.5)
            r2 = COVALENT_RADII.get(atoms[j], 1.5)

            # Calculate distance threshold
            threshold = (r1 + r2) * cov_factor

            # Calculate actual distance
            dist = np.linalg.norm(coords[i] - coords[j])

            # Add bond if within threshold
            if dist < threshold:
                mol.AddBond(i, j, Chem.BondType.SINGLE)


def geometry_to_mol_strict(
    geometry: Geometry, cov_factor: float = 1.2
) -> Chem.Mol:
    """
    Convert Geometry to RDKit molecule using STRICT distance-based bond detection.

    Unlike RDKit's DetermineBonds which uses heuristics, this function
    strictly uses distance cutoffs based on covalent radii.

    Bond threshold = (cov_radius_1 + cov_radius_2) * cov_factor

    For C-H with cov_factor=1.2: (0.76 + 0.31) * 1.2 = 1.28 Å

    Args:
        geometry: Geometry object
        cov_factor: Multiplier for covalent radii (default: 1.2)

    Returns:
        RDKit Mol with distance-based bonds (no bond order assignment)
    """
    mol = Chem.RWMol()

    # Add atoms
    for atom_symbol in geometry.atoms:
        mol.AddAtom(Chem.Atom(atom_symbol))

    # Add conformer
    conformer = Chem.Conformer(len(geometry.atoms))
    for i, coord in enumerate(geometry.coordinates):
        conformer.SetAtomPosition(i, coord.tolist())
    mol.AddConformer(conformer)

    # Add bonds based on distance
    add_bonds_by_distance(mol, geometry.coordinates, geometry.atoms, cov_factor)

    return mol.GetMol()


def geometry_to_mol(
    geometry: Geometry, charge: int = 0, cov_factor: float = 1.3
) -> Chem.Mol | None:
    """
    Convert a Geometry object to an RDKit molecule using DetermineBonds.

    Creates an RDKit molecule from atomic symbols and 3D coordinates.
    The molecule will have a 3D conformer with the coordinates from the geometry.
    Bonds are automatically determined based on 3D geometry using RDKit heuristics.

    Args:
        geometry: Geometry object containing atoms and coordinates
        charge: Molecular charge (default: 0). If bond determination fails,
                will try charges from -2 to +2
        cov_factor: Multiplier for covalent radii (default: 1.3, RDKit default)

    Returns:
        RDKit Mol object with 3D conformer and inferred bonds, or None if failed

    Example:
        >>> geom = read_xyz_file("molecule.xyz")
        >>> mol = geometry_to_mol(geom)
        >>> mol.GetNumAtoms()
        6
    """
    # Create an editable molecule
    mol = Chem.RWMol()

    # Add atoms
    for atom_symbol in geometry.atoms:
        atom = Chem.Atom(atom_symbol)
        mol.AddAtom(atom)

    # Add 3D conformer with coordinates
    conformer = Chem.Conformer(len(geometry.atoms))
    for i, coord in enumerate(geometry.coordinates):
        conformer.SetAtomPosition(i, coord.tolist())

    mol.AddConformer(conformer)

    # Convert to regular molecule
    mol = mol.GetMol()

    # Try to determine bonds with different charges if needed
    charges_to_try = [charge] + [c for c in range(-2, 3) if c != charge]

    for test_charge in charges_to_try:
        try:
            mol_copy = Chem.Mol(mol)
            rdDetermineBonds.DetermineBonds(
                mol_copy, charge=test_charge, covFactor=cov_factor
            )
            return mol_copy
        except ValueError:
            continue

    # If all charges fail, return molecule without bonds
    return mol


def geometry_to_mol_xyz_block(geometry: Geometry, cov_factor: float = 1.2) -> Chem.Mol:
    """
    Convert a Geometry object to an RDKit molecule using XYZ block parsing.

    Alternative approach that uses RDKit's XYZ block parser.

    Args:
        geometry: Geometry object containing atoms and coordinates
        cov_factor: Multiplier for covalent radii (default: 1.2)

    Returns:
        RDKit Mol object with 3D conformer
    """
    # Create XYZ block format
    xyz_lines = [str(len(geometry.atoms)), geometry.metadata]

    for atom, coord in zip(geometry.atoms, geometry.coordinates):
        xyz_lines.append(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

    xyz_block = "\n".join(xyz_lines)

    # Parse XYZ block
    mol = Chem.MolFromXYZBlock(xyz_block)

    if mol is not None:
        rdDetermineBonds.DetermineBonds(mol, charge=0, covFactor=cov_factor)

    return mol
