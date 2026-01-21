"""Module for Kabsch alignment and RMSD calculation."""

import numpy as np

from seamstress.geometry import Geometry

# Atomic masses for weighted alignment
ATOMIC_MASS = {
    'H': 1.008,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'F': 18.998,
    'S': 32.065,
    'Cl': 35.453,
    'Br': 79.904,
    'I': 126.904,
}


def get_atom_weights(atoms: list[str], weight_type: str = "mass") -> np.ndarray:
    """
    Get weights for atoms based on weight type.

    Args:
        atoms: List of atom symbols
        weight_type: "mass" for atomic mass, "uniform" for equal weights,
                     "heavy_only" for 1.0 on heavy atoms and 0.0 on H

    Returns:
        Array of weights, shape (N,)
    """
    if weight_type == "uniform":
        return np.ones(len(atoms))
    elif weight_type == "heavy_only":
        return np.array([0.0 if atom == 'H' else 1.0 for atom in atoms])
    elif weight_type == "mass":
        return np.array([ATOMIC_MASS.get(atom, 12.0) for atom in atoms])
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")


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
    is computed based ONLY on heavy atoms (non-hydrogen), but the RMSD is
    computed using ALL atoms (including hydrogens) to detect optimal
    hydrogen permutations.

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
        heavy_mask1 = np.array([atom != "H" for atom in atoms1])
        heavy_mask2 = np.array([atom != "H" for atom in atoms2])

        # Both should have same heavy atom pattern
        assert np.array_equal(heavy_mask1, heavy_mask2), (
            "Heavy atom patterns must match"
        )

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
    # H = Q.T @ P where Q is reference (centered1), P is target (centered2)
    H = centered1.T @ centered2

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

    # Calculate RMSD using ALL atoms (including hydrogens)
    # This ensures hydrogen permutations affect the RMSD
    diff = coords1 - aligned2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd, aligned2


def kabsch_align_only(
    coords1: np.ndarray,
    coords2: np.ndarray,
    atoms1: list[str] | None = None,
    atoms2: list[str] | None = None,
    use_all_atoms: bool = False,
    weight_type: str = "mass",
) -> np.ndarray:
    """
    Perform Kabsch alignment without computing RMSD.

    Aligns coords2 onto coords1 and returns the aligned coordinates.
    When using all atoms, applies weighted alignment based on atomic mass
    so that heavier atoms (e.g., carbons) contribute more to the alignment
    than lighter atoms (e.g., hydrogens).

    Args:
        coords1: Reference coordinates, shape (N, 3)
        coords2: Target coordinates to align, shape (N, 3)
        atoms1: Optional atom symbols for coords1 (for heavy atom filtering)
        atoms2: Optional atom symbols for coords2 (for heavy atom filtering)
        use_all_atoms: If True, use all atoms for alignment. If False (default),
                       use only heavy atoms when atom labels are provided.
        weight_type: Weight scheme for alignment when use_all_atoms=True:
                     "mass" (default) - use atomic mass (C=12, H=1)
                     "uniform" - equal weights for all atoms
                     "heavy_only" - only heavy atoms contribute (H weight=0)

    Returns:
        Aligned coords2, shape (N, 3)
    """
    assert coords1.shape == coords2.shape
    assert coords1.shape[1] == 3

    # Determine which atoms to use for alignment
    if use_all_atoms or atoms1 is None or atoms2 is None:
        # Use all atoms for alignment with weights
        align_coords1 = coords1
        align_coords2 = coords2

        if atoms1 is not None and use_all_atoms:
            weights = get_atom_weights(atoms1, weight_type)
        else:
            weights = np.ones(len(coords1))
    else:
        # Use only heavy atoms (exclude hydrogen)
        heavy_mask1 = np.array([atom != "H" for atom in atoms1])
        heavy_mask2 = np.array([atom != "H" for atom in atoms2])
        assert np.array_equal(heavy_mask1, heavy_mask2)
        align_coords1 = coords1[heavy_mask1]
        align_coords2 = coords2[heavy_mask2]
        weights = np.ones(len(align_coords1))

    # Normalize weights
    weights = weights / weights.sum()

    # Weighted centroids
    centroid1 = np.sum(weights[:, np.newaxis] * align_coords1, axis=0)
    centroid2 = np.sum(weights[:, np.newaxis] * align_coords2, axis=0)

    centered1 = align_coords1 - centroid1
    centered2 = align_coords2 - centroid2

    # Weighted covariance matrix: H = P^T @ W @ Q
    # where W is diagonal matrix of weights
    # Equivalent to: H = (sqrt(W) @ P)^T @ (sqrt(W) @ Q)
    sqrt_weights = np.sqrt(weights)
    weighted_centered1 = sqrt_weights[:, np.newaxis] * centered1
    weighted_centered2 = sqrt_weights[:, np.newaxis] * centered2

    H = weighted_centered1.T @ weighted_centered2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation and translation to ALL atoms
    centered_all2 = coords2 - centroid2
    aligned2 = centered_all2 @ R + centroid1

    return aligned2


def find_best_permutation_kabsch(
    ref_coords: np.ndarray,
    target_coords: np.ndarray,
    automorphisms: list[tuple[int, ...]],
    ref_atoms: list[str] | None = None,
    target_atoms: list[str] | None = None,
    max_iterations: int = 10,
) -> tuple[int, float, tuple[int, ...], np.ndarray]:
    """
    Find the best atom permutation by brute force over ALL possible permutations.

    ALGORITHM:
    For each permutation of heavy atoms (carbons):
      For each permutation of hydrogens:
        1. Build full permutation
        2. Apply permutation to target
        3. Align to reference (mass-weighted Kabsch)
        4. Calculate RMSD
    Return permutation with lowest RMSD.

    For ethylene (2C, 4H): 2! × 4! = 48 permutations.

    Args:
        ref_coords: Reference molecule coordinates, shape (N, 3)
        target_coords: Target molecule coordinates, shape (N, 3)
        automorphisms: List of permutation tuples (not used, kept for API)
        ref_atoms: Atom symbols for reference
        target_atoms: Atom symbols for target
        max_iterations: Not used (kept for API compatibility)

    Returns:
        Tuple of (best_auto_idx, best_rmsd, best_permutation, best_aligned_coords)
    """
    from itertools import permutations as itertools_perm

    if ref_atoms is None or target_atoms is None:
        # Fallback if no atom labels
        return _find_best_permutation_automorphisms(
            ref_coords, target_coords, automorphisms, ref_atoms, target_atoms
        )

    # Identify heavy atom and hydrogen indices
    ref_heavy_idx = [i for i, a in enumerate(ref_atoms) if a != 'H']
    ref_h_idx = [i for i, a in enumerate(ref_atoms) if a == 'H']
    tgt_heavy_idx = [i for i, a in enumerate(target_atoms) if a != 'H']
    tgt_h_idx = [i for i, a in enumerate(target_atoms) if a == 'H']

    if len(ref_heavy_idx) != len(tgt_heavy_idx) or len(ref_h_idx) != len(tgt_h_idx):
        return _find_best_permutation_automorphisms(
            ref_coords, target_coords, automorphisms, ref_atoms, target_atoms
        )

    best_rmsd = float("inf")
    best_perm = None
    best_aligned = None

    # Brute force: all permutations of heavy atoms × all permutations of hydrogens
    heavy_perms = list(itertools_perm(range(len(tgt_heavy_idx))))
    h_perms = list(itertools_perm(range(len(tgt_h_idx))))

    for heavy_perm in heavy_perms:
        for h_perm in h_perms:
            # Build full permutation
            perm = [0] * len(ref_atoms)
            for i, j in enumerate(heavy_perm):
                perm[ref_heavy_idx[i]] = tgt_heavy_idx[j]
            for i, j in enumerate(h_perm):
                perm[ref_h_idx[i]] = tgt_h_idx[j]

            # Apply permutation
            permuted_coords = target_coords[perm, :]
            permuted_atoms = [target_atoms[p] for p in perm]

            # Align with mass weighting
            aligned = kabsch_align_only(
                ref_coords, permuted_coords, ref_atoms, permuted_atoms,
                use_all_atoms=True, weight_type="mass"
            )

            # Calculate RMSD
            diff = ref_coords - aligned
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_perm = tuple(perm)
                best_aligned = aligned

    # Find index in automorphisms (may not exist)
    try:
        auto_idx = automorphisms.index(best_perm)
    except ValueError:
        auto_idx = -1

    return auto_idx, best_rmsd, best_perm, best_aligned


def _find_best_permutation_automorphisms(
    ref_coords: np.ndarray,
    target_coords: np.ndarray,
    automorphisms: list[tuple[int, ...]],
    ref_atoms: list[str] | None = None,
    target_atoms: list[str] | None = None,
) -> tuple[int, float, tuple[int, ...], np.ndarray]:
    """Fallback: brute force over automorphisms only."""
    best_rmsd = float("inf")
    best_auto_idx = 0
    best_perm = automorphisms[0]
    best_aligned = None

    for i, perm in enumerate(automorphisms):
        permuted = target_coords[list(perm), :]
        if target_atoms is not None:
            permuted_atoms = [target_atoms[j] for j in perm]
        else:
            permuted_atoms = None

        aligned = kabsch_align_only(
            ref_coords, permuted, ref_atoms, permuted_atoms,
            use_all_atoms=True, weight_type="mass"
        )

        diff = ref_coords - aligned
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_auto_idx = i
            best_perm = perm
            best_aligned = aligned

    return best_auto_idx, best_rmsd, best_perm, best_aligned


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
        auto_idx, rmsd, perm, aligned = find_best_permutation_kabsch(
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
