"""Module for computing molecular automorphisms and symmetry mappings."""

from typing import NamedTuple

import numpy as np
from rdkit import Chem

from elisa_spawns.geometry import Geometry
from elisa_spawns.rdkit_utils import geometry_to_mol


class AutomorphismTuple(NamedTuple):
    """Automorphism mapping between reference and target molecule."""

    reference_idx: int
    target_idx: int
    atom_mapping: tuple[int, ...]
    rmsd: float


def get_automorphisms(mol: Chem.Mol) -> list[tuple[int, ...]]:
    """
    Get all automorphisms (symmetry mappings) for a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        List of tuples, where each tuple is a permutation of atom indices
        representing an automorphism of the molecule
    """
    # Get the symmetry classes
    matches = mol.GetSubstructMatches(mol, uniquify=False)

    # Convert to list of tuples
    automorphisms = list(set(matches))

    return automorphisms


def compute_atom_mapping(
    ref_mol: Chem.Mol, target_mol: Chem.Mol, ref_geom: Geometry, target_geom: Geometry
) -> tuple[tuple[int, ...], float]:
    """
    Compute the best atom mapping between reference and target molecules.

    Uses substructure matching to find all possible mappings, then selects
    the one with lowest RMSD after alignment.

    Args:
        ref_mol: Reference RDKit molecule
        target_mol: Target RDKit molecule
        ref_geom: Reference geometry (for coordinates)
        target_geom: Target geometry (for coordinates)

    Returns:
        Tuple of (atom_mapping, rmsd) where atom_mapping[i] gives the index
        in the target molecule that corresponds to atom i in the reference
    """
    # Get all possible matches (automorphisms)
    matches = target_mol.GetSubstructMatches(ref_mol, uniquify=False)

    if not matches:
        # Fallback to identity mapping if no matches found
        n_atoms = ref_mol.GetNumAtoms()
        return tuple(range(n_atoms)), float("inf")

    best_mapping = None
    best_rmsd = float("inf")

    ref_coords = ref_geom.coordinates
    target_coords = target_geom.coordinates

    # Try each possible mapping and find the one with lowest RMSD
    for match in matches:
        # Calculate RMSD for this mapping
        mapped_target_coords = target_coords[list(match), :]
        diff = ref_coords - mapped_target_coords
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_mapping = match

    return best_mapping, best_rmsd


def compute_automorphism_tuples(
    geometries: list[Geometry],
) -> list[AutomorphismTuple]:
    """
    Compute automorphism tuples for a list of geometries.

    Uses the first geometry as reference and maps all others to it.

    Args:
        geometries: List of Geometry objects with same connectivity

    Returns:
        List of AutomorphismTuple objects, one per geometry
    """
    if not geometries:
        return []

    # Use first as reference
    ref_geom = geometries[0]
    ref_mol = geometry_to_mol(ref_geom)

    results = []

    for i, geom in enumerate(geometries):
        target_mol = geometry_to_mol(geom)

        if i == 0:
            # Reference maps to itself with identity
            n_atoms = ref_mol.GetNumAtoms()
            mapping = tuple(range(n_atoms))
            rmsd = 0.0
        else:
            # Compute mapping to reference
            mapping, rmsd = compute_atom_mapping(ref_mol, target_mol, ref_geom, geom)

        results.append(
            AutomorphismTuple(
                reference_idx=0,
                target_idx=i,
                atom_mapping=mapping,
                rmsd=rmsd,
            )
        )

    return results


def print_automorphism_tuples(
    geometries: list[Geometry], tuples: list[AutomorphismTuple], max_show: int = 20
) -> None:
    """
    Print automorphism tuples in a readable format.

    Args:
        geometries: List of geometries
        tuples: List of AutomorphismTuple objects
        max_show: Maximum number to display
    """
    if not tuples:
        print("No automorphism tuples to display")
        return

    ref_geom = geometries[0]

    print(f"\nAutomorphism tuples (reference: {ref_geom.filename})")
    print("=" * 80)
    print(f"Reference atoms: {' '.join(ref_geom.atoms)}")
    print("=" * 80)

    for i, auto_tuple in enumerate(tuples[:max_show]):
        target_geom = geometries[auto_tuple.target_idx]
        mapping_str = " ".join(str(idx) for idx in auto_tuple.atom_mapping)

        print(f"\n{i + 1}. {target_geom.filename}")
        print(f"   Mapping: [{mapping_str}]")
        print(f"   RMSD: {auto_tuple.rmsd:.4f} Å")

        # Show the permuted atom order
        permuted_atoms = [target_geom.atoms[idx] for idx in auto_tuple.atom_mapping]
        print(f"   Permuted atoms: {' '.join(permuted_atoms)}")

    if len(tuples) > max_show:
        print(f"\n... and {len(tuples) - max_show} more")


def print_template_automorphisms(geometry: Geometry) -> None:
    """
    Print all automorphisms for a single template geometry.

    Args:
        geometry: Template geometry to analyze
    """
    mol = geometry_to_mol(geometry)
    automorphisms = get_automorphisms(mol)

    print(f"\nTemplate: {geometry.filename}")
    print("=" * 80)
    print(f"Atoms: {' '.join(geometry.atoms)}")
    print(f"Number of automorphisms: {len(automorphisms)}")
    print("=" * 80)

    # Print tuples in a clean format
    print("\nAutomorphism tuples:")
    tuples_str = ",\n".join(f"  {auto}" for auto in automorphisms)
    print(f"[\n{tuples_str}\n]")

    # Also show detailed mappings
    print("\nDetailed mappings:")
    for i, auto in enumerate(automorphisms, 1):
        mapping_str = ", ".join(str(idx) for idx in auto)
        permuted_atoms = [geometry.atoms[idx] for idx in auto]
        atoms_str = ", ".join(permuted_atoms)
        print(f"  {i}. ({mapping_str}) -> ({atoms_str})")
