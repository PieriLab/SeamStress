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


def get_atom_weights(
    atoms: list[str],
    weight_type: str = "mass",
    heavy_atom_factor: float = 1.0
) -> np.ndarray:
    """
    Get weights for atoms based on weight type.

    Args:
        atoms: List of atom symbols
        weight_type: "mass" for atomic mass, "uniform" for equal weights,
                     "heavy_only" for 1.0 on heavy atoms and 0.0 on H
        heavy_atom_factor: Multiplier for non-hydrogen atoms (only applies to "mass" weight_type).
                          For example, heavy_atom_factor=10.0 makes heavy atoms 10x more important.
                          Default is 1.0 (no additional weighting beyond atomic mass).

    Returns:
        Array of weights, shape (N,)
    """
    if weight_type == "uniform":
        return np.ones(len(atoms))
    elif weight_type == "heavy_only":
        return np.array([0.0 if atom == 'H' else 1.0 for atom in atoms])
    elif weight_type == "mass":
        weights = np.array([ATOMIC_MASS.get(atom, 12.0) for atom in atoms])
        if heavy_atom_factor != 1.0:
            # Apply heavy atom factor to non-hydrogen atoms
            heavy_mask = np.array([atom != 'H' for atom in atoms])
            weights[heavy_mask] *= heavy_atom_factor
        return weights
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")


def kabsch_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    atoms1: list[str] | None = None,
    atoms2: list[str] | None = None,
    allow_reflection: bool = False,
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
    # Reflection handling
    if not allow_reflection and np.linalg.det(R) < 0:
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
    heavy_atom_factor: float = 1.0,
    allow_reflection: bool = False,
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
        heavy_atom_factor: Multiplier for heavy atom weights (default: 1.0)
        allow_reflection: If True, allow improper rotations (det(R) < 0).
                  If False (default), enforce a proper rotation.

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
            weights = get_atom_weights(atoms1, weight_type, heavy_atom_factor)
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

   

    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation and translation to ALL atoms
    centered_all2 = coords2 - centroid2
    aligned2 = centered_all2 @ R + centroid1

    return aligned2


def _build_fragment_map(atoms: list[str], mol) -> dict[int, list[int]] | None:
    """
    Build fragment map from connectivity graph.

    Each heavy atom and its bonded hydrogens form a fragment.
    Only works if all heavy atoms have exactly 1 hydrogen bonded.

    Args:
        atoms: List of atom symbols
        mol: RDKit molecule with connectivity information

    Returns:
        Dictionary mapping heavy atom index to list [heavy_idx, h_idx]
        Returns None if fragment mode not applicable (inconsistent H counts)
    """
    if mol is None:
        return None

    # Build heavy atom to hydrogens map
    heavy_to_h = {}

    for i, atom_symbol in enumerate(atoms):
        if atom_symbol != 'H':
            # Get bonded hydrogens
            atom = mol.GetAtomWithIdx(i)
            bonded_h = []
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    bonded_h.append(neighbor.GetIdx())
            heavy_to_h[i] = bonded_h

    # Check if all heavy atoms have exactly 1 hydrogen
    h_counts = [len(h_list) for h_list in heavy_to_h.values()]
    if not h_counts or not all(c == 1 for c in h_counts):
        return None  # Fragment mode not applicable

    # Build fragment map: heavy_idx -> [heavy_idx, h_idx]
    fragments = {}
    for heavy_idx, h_list in heavy_to_h.items():
        fragments[heavy_idx] = [heavy_idx] + h_list

    return fragments


def _find_best_permutation_fragments(
    ref_coords: np.ndarray,
    target_coords: np.ndarray,
    automorphisms: list[tuple[int, ...]],
    ref_atoms: list[str],
    target_atoms: list[str],
    ref_fragments: dict[int, list[int]],
    tgt_fragments: dict[int, list[int]],
    allow_reflection: bool = False,
) -> tuple[int, float, tuple[int, ...], np.ndarray]:
    """
    Find best permutation using fragment-based search.

    Permutes heavy atoms and their bonded hydrogens as rigid units.
    Much faster when applicable (e.g., benzene: 720 vs 518,400 permutations).
    """
    from itertools import permutations as itertools_perm

    # Get lists of heavy atom indices
    ref_heavy_indices = sorted(ref_fragments.keys())
    tgt_heavy_indices = sorted(tgt_fragments.keys())

    if len(ref_heavy_indices) != len(tgt_heavy_indices):
        # Fallback to automorphism search
        return _find_best_permutation_automorphisms(
            ref_coords, target_coords, automorphisms, ref_atoms, target_atoms, allow_reflection=allow_reflection
        )

    best_rmsd = float("inf")
    best_perm = None
    best_aligned = None

    # Permute heavy atoms only (hydrogens follow their heavy atoms)
    n_heavy = len(ref_heavy_indices)
    for heavy_perm in itertools_perm(range(n_heavy)):
        # Build full permutation by mapping fragments
        perm = [0] * len(ref_atoms)

        for i, j in enumerate(heavy_perm):
            ref_frag = ref_fragments[ref_heavy_indices[i]]
            tgt_frag = tgt_fragments[tgt_heavy_indices[j]]

            # Map all atoms in fragment together
            for k in range(len(ref_frag)):
                perm[ref_frag[k]] = tgt_frag[k]

        # Apply permutation
        permuted_coords = target_coords[perm, :]
        permuted_atoms = [target_atoms[p] for p in perm]

        # Align with normal mass weighting
        aligned = kabsch_align_only(
            ref_coords, permuted_coords, ref_atoms, permuted_atoms,
            use_all_atoms=True, weight_type="mass", heavy_atom_factor=1.0, allow_reflection = allow_reflection,
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


def find_best_permutation_kabsch(
    ref_coords: np.ndarray,
    target_coords: np.ndarray,
    automorphisms: list[tuple[int, ...]],
    ref_atoms: list[str] | None = None,
    target_atoms: list[str] | None = None,
    max_iterations: int = 10,
    use_permutations: bool = True,
    use_fragment_permutations: bool = False,
    ref_mol=None,
    target_mol=None,
    allow_reflection: bool = False, 
) -> tuple[int, float, tuple[int, ...], np.ndarray]:
    """
    Find the best atom permutation by brute force over ALL possible permutations.

    ALGORITHM (standard mode):
    For each permutation of heavy atoms (carbons):
      For each permutation of hydrogens:
        1. Build full permutation
        2. Apply permutation to target
        3. Align to reference (normal mass-weighted Kabsch)
        4. Calculate RMSD
    Return permutation with lowest RMSD.

    For ethylene (2C, 4H): 2! × 4! = 48 permutations tested.

    ALGORITHM (fragment mode - when use_fragment_permutations=True):
    Treats heavy atoms and their bonded hydrogens as rigid fragments.
    Only applicable when all heavy atoms have exactly 1 hydrogen.
    For benzene (6 C-H fragments): 6! = 720 permutations instead of 6! × 6! = 518,400

    NOTE: This function does NOT apply heavy atom weighting. That should be done
    separately after the permutation is selected using refine_alignment_with_heavy_atoms().

    Args:
        ref_coords: Reference molecule coordinates, shape (N, 3)
        target_coords: Target molecule coordinates, shape (N, 3)
        automorphisms: List of permutation tuples (not used, kept for API)
        ref_atoms: Atom symbols for reference
        target_atoms: Atom symbols for target
        max_iterations: Not used (kept for API compatibility)
        use_permutations: If True, search for optimal permutations; if False, use identity permutation only
        use_fragment_permutations: If True, use fragment-based permutation (heavy atoms + bonded H as units)
        ref_mol: RDKit molecule for reference (needed for fragment mode)
        target_mol: RDKit molecule for target (needed for fragment mode)

    Returns:
        Tuple of (best_auto_idx, best_rmsd, best_permutation, best_aligned_coords)
    """
    from itertools import permutations as itertools_perm

    # If use_permutations is False, just use identity permutation
    if not use_permutations:
        identity_perm = tuple(range(len(ref_coords)))
        aligned = kabsch_align_only(
            ref_coords, target_coords, ref_atoms, target_atoms,
            use_all_atoms=True, weight_type="mass", heavy_atom_factor=1.0,allow_reflection=allow_reflection,
        )
        diff = ref_coords - aligned
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return 0, rmsd, identity_perm, aligned

    if ref_atoms is None or target_atoms is None:
        # Fallback if no atom labels
        return _find_best_permutation_automorphisms(
            ref_coords, target_coords, automorphisms, ref_atoms, target_atoms, allow_reflection=allow_reflection
        )

    # Try fragment-based permutation if requested
    if use_fragment_permutations and ref_mol is not None and target_mol is not None:
        ref_fragments = _build_fragment_map(ref_atoms, ref_mol)
        tgt_fragments = _build_fragment_map(target_atoms, target_mol)

        if ref_fragments is not None and tgt_fragments is not None:
            return _find_best_permutation_fragments(
                ref_coords, target_coords, automorphisms,
                ref_atoms, target_atoms, ref_fragments, tgt_fragments, allow_reflection = allow_reflection
            )
        # If fragment mode not applicable, fall through to standard mode

    # Identify heavy atom and hydrogen indices
    ref_heavy_idx = [i for i, a in enumerate(ref_atoms) if a != 'H']
    ref_h_idx = [i for i, a in enumerate(ref_atoms) if a == 'H']
    tgt_heavy_idx = [i for i, a in enumerate(target_atoms) if a != 'H']
    tgt_h_idx = [i for i, a in enumerate(target_atoms) if a == 'H']

    if len(ref_heavy_idx) != len(tgt_heavy_idx) or len(ref_h_idx) != len(tgt_h_idx):
        return _find_best_permutation_automorphisms(
            ref_coords, target_coords, automorphisms, ref_atoms, target_atoms, allow_reflection=allow_reflection
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

            # Align with normal mass weighting
            aligned = kabsch_align_only(
                ref_coords, permuted_coords, ref_atoms, permuted_atoms,
                use_all_atoms=True, weight_type="mass", heavy_atom_factor=1.0, allow_reflection=allow_reflection,
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
    allow_reflection : bool = False,
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
            use_all_atoms=True, weight_type="mass", heavy_atom_factor=1.0, allow_reflection=allow_reflection,
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
    use_permutations: bool = True,
    heavy_atom_factor: float = 1.0,
    use_fragment_permutations: bool = False,
    allow_reflection: bool = False,
) -> list[dict]:
    """
    Align all target geometries to reference using automorphisms.

    TWO-STAGE ALIGNMENT PROCESS:
    1. Permutation Search: Finds best atom permutation using normal mass weighting (intra-family)
    2. Heavy Atom Refinement: Re-aligns with heavy atom weighting to reference (inter-family)

    This ensures permutation selection is unbiased within each family, while allowing
    heavy atoms to dominate when aligning to the family reference structure.

    Args:
        reference: Reference geometry (template)
        targets: List of target geometries to align
        automorphisms: Automorphism permutations for this molecule type
        use_permutations: If True, search for optimal permutations; if False, use identity permutation only
        heavy_atom_factor: Multiplier for heavy atom weights when aligning to reference (default: 1.0).
                          Values > 1.0 make heavy atoms dominate the final reference alignment.
                          Applied AFTER best permutation is selected.
        use_fragment_permutations: If True, use fragment-based permutation search (faster for molecules
                                  like benzene where each heavy atom has exactly 1 hydrogen)

    Returns:
        List of dicts with keys:
            - 'geometry': Original Geometry object
            - 'permutation': Best atom permutation tuple
            - 'rmsd': RMSD after optimal alignment
            - 'aligned_coords': Optimally aligned coordinates (N, 3)
            - 'reordered_atoms': Atom symbols in reordered sequence
            - 'reordered_coords': Coordinates in reordered sequence (not aligned)
    """
    from seamstress.rdkit_utils import geometry_to_mol

    ref_coords = reference.coordinates
    ref_atoms = reference.atoms
    results = []

    # Get RDKit molecules if fragment mode is enabled
    ref_mol = None
    if use_fragment_permutations:
        ref_mol = geometry_to_mol(reference)

    total = len(targets)
    for idx, target in enumerate(targets, 1):
        target_coords = target.coordinates
        target_atoms = target.atoms

        # Get target molecule for fragment mode
        target_mol = None
        if use_fragment_permutations:
            target_mol = geometry_to_mol(target)

        # STAGE 1: Find best permutation using normal mass weighting
        auto_idx, rmsd, perm, aligned = find_best_permutation_kabsch(
            ref_coords, target_coords, automorphisms, ref_atoms, target_atoms,
            use_permutations=use_permutations,
            use_fragment_permutations=use_fragment_permutations,
            ref_mol=ref_mol,
            target_mol=target_mol,
            allow_reflection=allow_reflection
        )

        # Reordered coordinates (permuted but not Kabsch-aligned)
        reordered_coords = target_coords[list(perm), :]
        reordered_atoms = [target.atoms[i] for i in perm]

        # STAGE 2: Refine alignment with heavy atom weighting if requested
        if heavy_atom_factor > 1.0:
            permuted_coords = target_coords[list(perm), :]
            permuted_atoms = [target_atoms[p] for p in perm]

            aligned = kabsch_align_only(
                ref_coords, permuted_coords, ref_atoms, permuted_atoms,
                use_all_atoms=True, weight_type="mass", heavy_atom_factor=heavy_atom_factor, allow_reflection=allow_reflection,
            )

            # Recalculate RMSD with refined alignment
            diff = ref_coords - aligned
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

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

        # Show progress
        if total >= 50:
            # For large datasets, show progress every 10 geometries
            if idx % 10 == 0 or idx == total:
                print(f"  Progress: {idx}/{total} geometries aligned", end='\r')
        elif total >= 10:
            # For medium datasets, show every 5
            if idx % 5 == 0 or idx == total:
                print(f"  Progress: {idx}/{total} geometries aligned", end='\r')
        # For small datasets (< 10), don't show progress

    # Print final newline if progress was shown
    if total >= 10:
        print()

    return results
