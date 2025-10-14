"""Module for Kabsch alignment and RMSD calculation."""

import numpy as np
from scipy.spatial.transform import Rotation

from elisa_spawns.geometry import Geometry


def kabsch_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    atoms1: list[str] | None = None,
    atoms2: list[str] | None = None,
) -> tuple[float, np.ndarray]:
    """
    Calculate RMSD after optimal Kabsch alignment.

    Uses the Kabsch algorithm to find the optimal rotation and translation
    to align coords2 onto coords1, then computes the RMSD.

    When atom labels are provided, the alignment (rotation and translation)
    is computed based ONLY on heavy atoms (non-hydrogen), but the transformation
    is applied to ALL atoms.

    Args:
        coords1: Reference coordinates, shape (N, 3)
        coords2: Target coordinates to align, shape (N, 3)
        atoms1: Optional atom symbols for coords1 (for heavy atom filtering)
        atoms2: Optional atom symbols for coords2 (for heavy atom filtering)

    Returns:
        Tuple of (rmsd, aligned_coords2) where aligned_coords2 has been
        optimally rotated and translated to match coords1
    """
    assert coords1.shape == coords2.shape
    assert coords1.shape[1] == 3

    # Determine which atoms to use for alignment
    if atoms1 is not None and atoms2 is not None:
        # Filter to heavy atoms only (exclude hydrogen)
        heavy_mask1 = np.array([atom != 'H' for atom in atoms1])
        heavy_mask2 = np.array([atom != 'H' for atom in atoms2])

        # Both should have same heavy atom pattern
        assert np.array_equal(heavy_mask1, heavy_mask2), "Heavy atom patterns must match"

        # Use only heavy atoms for computing alignment
        align_coords1 = coords1[heavy_mask1]
        align_coords2 = coords2[heavy_mask2]
    else:
        # Use all atoms for alignment (original behavior)
        align_coords1 = coords1
        align_coords2 = coords2

    # Center both coordinate sets (using heavy atoms)
    centroid1 = align_coords1.mean(axis=0)
    centroid2 = align_coords2.mean(axis=0)

    centered1 = align_coords1 - centroid1
    centered2 = align_coords2 - centroid2

    # Compute covariance matrix (using heavy atoms)
    H = centered2.T @ centered1

    # SVD to find optimal rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation and translation to ALL atoms
    centered_all2 = coords2 - centroid2
    aligned2 = centered_all2 @ R + centroid1

    # Calculate RMSD (using heavy atoms only if specified)
    if atoms1 is not None:
        diff = align_coords1 - aligned2[heavy_mask1]
    else:
        diff = coords1 - aligned2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd, aligned2


def test_automorphisms_kabsch(
    ref_coords: np.ndarray,
    target_coords: np.ndarray,
    automorphisms: list[tuple[int, ...]],
    ref_atoms: list[str] | None = None,
    target_atoms: list[str] | None = None,
) -> tuple[int, float, tuple[int, ...], np.ndarray]:
    """
    Test all automorphism permutations and find the one with lowest Kabsch RMSD.

    When atom labels are provided, alignment is computed using only heavy atoms,
    but the transformation is applied to all atoms.

    Args:
        ref_coords: Reference molecule coordinates, shape (N, 3)
        target_coords: Target molecule coordinates, shape (N, 3)
        automorphisms: List of permutation tuples from get_automorphisms
        ref_atoms: Optional atom symbols for reference (for heavy atom alignment)
        target_atoms: Optional atom symbols for target (for heavy atom alignment)

    Returns:
        Tuple of (best_auto_idx, best_rmsd, best_permutation, best_aligned_coords)
    """
    best_rmsd = float("inf")
    best_auto_idx = 0
    best_permutation = automorphisms[0]
    best_aligned_coords = None

    for i, perm in enumerate(automorphisms):
        # Permute target coordinates according to this automorphism
        permuted_target = target_coords[list(perm), :]

        # Permute atom labels if provided
        if target_atoms is not None:
            permuted_atoms = [target_atoms[j] for j in perm]
        else:
            permuted_atoms = None

        # Compute Kabsch RMSD (using heavy atoms for alignment if atoms provided)
        rmsd, aligned = kabsch_rmsd(ref_coords, permuted_target, ref_atoms, permuted_atoms)

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_auto_idx = i
            best_permutation = perm
            best_aligned_coords = aligned

    return best_auto_idx, best_rmsd, best_permutation, best_aligned_coords


def align_geometries_with_automorphisms(
    reference: Geometry,
    targets: list[Geometry],
    automorphisms: list[tuple[int, ...]],
) -> list[dict]:
    """
    Align all target geometries to reference using automorphisms.

    For each target, tries all automorphism permutations and selects
    the one with lowest Kabsch RMSD.

    Alignment (rotation and translation) is computed using ONLY heavy atoms,
    but the transformation is applied to all atoms (including hydrogens).

    Args:
        reference: Reference geometry (template)
        targets: List of target geometries to align
        automorphisms: Automorphism permutations for this molecule type

    Returns:
        List of dicts with keys:
            - 'geometry': Original Geometry object
            - 'permutation': Best atom permutation tuple
            - 'rmsd': RMSD after optimal alignment (heavy atoms only)
            - 'aligned_coords': Optimally aligned coordinates (N, 3)
            - 'reordered_atoms': Atom symbols in reordered sequence
            - 'reordered_coords': Coordinates in reordered sequence (not aligned)
    """
    ref_coords = reference.coordinates
    ref_atoms = reference.atoms
    results = []

    for target in targets:
        target_coords = target.coordinates
        target_atoms = target.atoms

        # Find best permutation (using heavy atoms for alignment)
        auto_idx, rmsd, perm, aligned = test_automorphisms_kabsch(
            ref_coords, target_coords, automorphisms, ref_atoms, target_atoms
        )

        # Reordered coordinates (permuted but not Kabsch-aligned)
        reordered_coords = target_coords[list(perm), :]
        reordered_atoms = [target.atoms[i] for i in perm]

        results.append(
            {
                "geometry": target,
                "permutation": perm,
                "rmsd": rmsd,
                "aligned_coords": aligned,
                "reordered_atoms": reordered_atoms,
                "reordered_coords": reordered_coords,
            }
        )

    return results
