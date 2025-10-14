"""Module for analyzing molecular connectivity and grouping molecules."""

from collections import defaultdict
from typing import NamedTuple

import numpy as np
from rdkit import Chem

from elisa_spawns.geometry import Geometry
from elisa_spawns.rdkit_utils import geometry_to_mol


class ConnectivityInfo(NamedTuple):
    """Information about molecular connectivity."""

    distance_matrix: np.ndarray
    adjacency_matrix: np.ndarray
    connectivity_hash: str


def calculate_distance_matrix(geometry: Geometry) -> np.ndarray:
    """
    Calculate the pairwise distance matrix for all atoms in a geometry.

    Args:
        geometry: Geometry object with coordinates

    Returns:
        NxN numpy array where element [i,j] is the distance between atoms i and j
    """
    coords = geometry.coordinates
    n_atoms = len(coords)

    # Calculate all pairwise distances using broadcasting
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))

    return distances


def get_adjacency_matrix(mol: Chem.Mol) -> np.ndarray:
    """
    Get the adjacency (connectivity) matrix from an RDKit molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        NxN numpy array where element [i,j] is 1 if atoms are bonded, 0 otherwise
    """
    n_atoms = mol.GetNumAtoms()
    adj_matrix = np.zeros((n_atoms, n_atoms), dtype=int)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1

    return adj_matrix


def get_connectivity_signature(mol: Chem.Mol) -> str:
    """
    Generate a unique signature for the connectivity pattern of a molecule.

    Uses canonical SMILES without stereochemistry to identify connectivity.

    Args:
        mol: RDKit Mol object

    Returns:
        Canonical connectivity signature string
    """
    # Remove stereochemistry and generate canonical SMILES
    mol_copy = Chem.Mol(mol)
    Chem.RemoveStereochemistry(mol_copy)
    smiles = Chem.MolToSmiles(mol_copy, canonical=True)

    return smiles


def analyze_connectivity(geometry: Geometry) -> ConnectivityInfo:
    """
    Analyze the connectivity of a geometry.

    Args:
        geometry: Geometry object to analyze

    Returns:
        ConnectivityInfo with distance matrix, adjacency matrix, and hash
    """
    # Calculate distance matrix
    dist_matrix = calculate_distance_matrix(geometry)

    # Convert to RDKit molecule to get connectivity
    mol = geometry_to_mol(geometry)

    # Get adjacency matrix
    adj_matrix = get_adjacency_matrix(mol)

    # Get connectivity signature
    conn_hash = get_connectivity_signature(mol)

    return ConnectivityInfo(
        distance_matrix=dist_matrix,
        adjacency_matrix=adj_matrix,
        connectivity_hash=conn_hash,
    )


def group_by_connectivity(geometries: list[Geometry]) -> dict[str, list[Geometry]]:
    """
    Group geometries by their connectivity pattern.

    Args:
        geometries: List of Geometry objects

    Returns:
        Dictionary mapping connectivity signature to list of geometries with that pattern
    """
    groups = defaultdict(list)

    for geom in geometries:
        conn_info = analyze_connectivity(geom)
        groups[conn_info.connectivity_hash].append(geom)

    return dict(groups)


def print_connectivity_summary(groups: dict[str, list[Geometry]]) -> None:
    """
    Print a summary of connectivity groups.

    Args:
        groups: Dictionary from group_by_connectivity
    """
    print(f"\nFound {len(groups)} unique connectivity patterns:\n")
    print("=" * 70)

    for i, (conn_hash, geoms) in enumerate(
        sorted(groups.items(), key=lambda x: -len(x[1])), 1
    ):
        print(f"\nGroup {i}: {len(geoms)} molecules")
        print(f"  Connectivity: {conn_hash}")
        print(f"  Example files: {', '.join(g.filename for g in geoms[:3])}")
        if len(geoms) > 3:
            print(f"  ... and {len(geoms) - 3} more")
