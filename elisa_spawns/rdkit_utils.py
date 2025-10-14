"""Utilities for converting geometries to RDKit molecules."""

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from elisa_spawns.geometry import Geometry


def geometry_to_mol(geometry: Geometry, charge: int = 0) -> Chem.Mol | None:
    """
    Convert a Geometry object to an RDKit molecule.

    Creates an RDKit molecule from atomic symbols and 3D coordinates.
    The molecule will have a 3D conformer with the coordinates from the geometry.
    Bonds are automatically determined based on 3D geometry.

    Args:
        geometry: Geometry object containing atoms and coordinates
        charge: Molecular charge (default: 0). If bond determination fails,
                will try charges from -2 to +2

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
            rdDetermineBonds.DetermineBonds(mol_copy, charge=test_charge)
            return mol_copy
        except ValueError:
            continue

    # If all charges fail, return molecule without bonds
    return mol


def geometry_to_mol_xyz_block(geometry: Geometry) -> Chem.Mol:
    """
    Convert a Geometry object to an RDKit molecule using XYZ block parsing.

    Alternative approach that uses RDKit's XYZ block parser.

    Args:
        geometry: Geometry object containing atoms and coordinates

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
        rdDetermineBonds.DetermineBonds(mol, charge=0)

    return mol
