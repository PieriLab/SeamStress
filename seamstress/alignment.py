import numpy as np
from seamstress.geometry import Geometry
from dataclasses import dataclass
from typing import Optional, Tuple, List
from tqdm import tqdm
from enum import Enum
from itertools import permutations, product
from collections import defaultdict
from seamstress.rdkit_utils import geometry_to_mol
from rdkit.Chem import AllChem

# ============================================================
# ENUM
# ============================================================

class AlignmentMethod(str, Enum):
    IDENTITY = "identity"
    BRUTE_FORCE = "bruteforce"
    AUTOMORPHISM = "automorphism"
    FRAGMENT = "fragment"
    ISOMORPHISM = "isomorphism"
    MCS_HUNGARIAN = "mcs-hungarian"


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class AlignmentResult:
    geometry: "Geometry"
    permutation: Tuple[int, ...]
    best_rmsd: float
    identity_rmsd: float
    worst_rmsd: float
    improvement: float
    was_swapped: bool
    aligned_coords: np.ndarray
    reordered_atoms: List[str]
    reordered_coords: np.ndarray


# ============================================================
# ATOMIC WEIGHTS
# ============================================================

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
    if weight_type == "uniform":
        return np.ones(len(atoms))
    elif weight_type == "heavy_only":
        return np.array([0.0 if atom == 'H' else 1.0 for atom in atoms])
    elif weight_type == "mass":
        weights = np.array([ATOMIC_MASS.get(atom, 12.0) for atom in atoms])
        if heavy_atom_factor != 1.0:
            heavy_mask = np.array([atom != 'H' for atom in atoms])
            weights[heavy_mask] *= heavy_atom_factor
        return weights
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")


# ============================================================
# KABSCH ALIGNMENT
# ============================================================

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
    assert coords1.shape == coords2.shape
    assert coords1.shape[1] == 3

    if use_all_atoms or atoms1 is None or atoms2 is None:
        align_coords1 = coords1
        align_coords2 = coords2
        if atoms1 is not None and use_all_atoms:
            weights = get_atom_weights(atoms1, weight_type, heavy_atom_factor)
        else:
            weights = np.ones(len(coords1))
    else:
        heavy_mask1 = np.array([atom != "H" for atom in atoms1])
        heavy_mask2 = np.array([atom != "H" for atom in atoms2])
        assert np.array_equal(heavy_mask1, heavy_mask2)
        align_coords1 = coords1[heavy_mask1]
        align_coords2 = coords2[heavy_mask2]
        weights = np.ones(len(align_coords1))

    weights = weights / weights.sum()

    centroid1 = np.sum(weights[:, np.newaxis] * align_coords1, axis=0)
    centroid2 = np.sum(weights[:, np.newaxis] * align_coords2, axis=0)

    centered1 = align_coords1 - centroid1
    centered2 = align_coords2 - centroid2

    sqrt_weights = np.sqrt(weights)
    weighted_centered1 = sqrt_weights[:, np.newaxis] * centered1
    weighted_centered2 = sqrt_weights[:, np.newaxis] * centered2

    H = weighted_centered1.T @ weighted_centered2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    centered_all2 = coords2 - centroid2
    aligned2 = centered_all2 @ R + centroid1

    return aligned2


# ============================================================
# ALIGNMENT STRATEGIES
# ============================================================

def _search_identity(ref_coords, tgt_coords, ref_atoms, tgt_atoms, allow_reflection):
    identity = tuple(range(len(ref_atoms)))
    aligned = kabsch_align_only(ref_coords, tgt_coords, ref_atoms, tgt_atoms, use_all_atoms=True, allow_reflection=allow_reflection)
    diff = ref_coords - aligned
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return identity, rmsd, rmsd, rmsd, aligned, tgt_coords, tgt_atoms


def _search_bruteforce_elementwise(ref_coords, tgt_coords, ref_atoms, tgt_atoms, allow_reflection):
    ref_groups = defaultdict(list)
    tgt_groups = defaultdict(list)

    for i, a in enumerate(ref_atoms):
        ref_groups[a].append(i)
    for i, a in enumerate(tgt_atoms):
        tgt_groups[a].append(i)

    if set(ref_groups.keys()) != set(tgt_groups.keys()):
        raise ValueError("Element sets differ between molecules.")
    for element in ref_groups:
        if len(ref_groups[element]) != len(tgt_groups[element]):
            raise ValueError(f"Atom count mismatch for element {element}")

    element_perms = [list(permutations(tgt_groups[e])) for e in sorted(ref_groups.keys())]
    element_keys = sorted(ref_groups.keys())

    best_rmsd = float("inf")
    worst_rmsd = 0.0
    identity = tuple(range(len(ref_atoms)))
    identity_rmsd = None
    best_perm = None
    best_aligned = None
    best_reordered = None
    best_atoms = None

    for combo in product(*element_perms):
        perm = [0] * len(ref_atoms)
        for element, tgt_perm in zip(element_keys, combo):
            ref_idx = ref_groups[element]
            for i, tgt_index in enumerate(tgt_perm):
                perm[ref_idx[i]] = tgt_index
        perm = tuple(perm)

        reordered_coords = tgt_coords[list(perm)]
        reordered_atoms = [tgt_atoms[i] for i in perm]
        aligned = kabsch_align_only(ref_coords, reordered_coords, ref_atoms, reordered_atoms, use_all_atoms=True, allow_reflection=allow_reflection)
        diff = ref_coords - aligned
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

        if perm == identity:
            identity_rmsd = rmsd
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_perm = perm
            best_aligned = aligned
            best_reordered = reordered_coords
            best_atoms = reordered_atoms
        worst_rmsd = max(worst_rmsd, rmsd)

    if identity_rmsd is None:
        identity_rmsd = worst_rmsd

    return best_perm, best_rmsd, identity_rmsd, worst_rmsd, best_aligned, best_reordered, best_atoms

def _search_automorphism(ref_coords, tgt_coords, automorphisms, ref_atoms, tgt_atoms, allow_reflection):
    best_rmsd = float("inf")
    worst_rmsd = 0.0
    identity = tuple(range(len(ref_atoms)))
    identity_rmsd = None
    best_perm = None
    best_aligned = None
    best_reordered = None
    best_atoms = None

    for perm in automorphisms:
        
        reordered_coords = tgt_coords[list(perm)]
        #print(tgt_coords)
        #print(reordered_coords)
        reordered_atoms = [tgt_atoms[i] for i in perm]
        aligned = kabsch_align_only(ref_coords, reordered_coords, ref_atoms, reordered_atoms, use_all_atoms=True, allow_reflection=allow_reflection)
        
        diff = ref_coords - aligned
        #print(perm)
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))



        if perm == identity:
            identity_rmsd = rmsd
        if rmsd < best_rmsd:
            #print(rmsd,perm)
            best_rmsd = rmsd
            best_perm = perm
            best_aligned = aligned
            #print(aligned)
            best_reordered = reordered_coords
            #print(reordered_coords)
            best_atoms = reordered_atoms
        worst_rmsd = max(worst_rmsd, rmsd)

    if identity_rmsd is None:
        identity_rmsd = worst_rmsd

    return best_perm, best_rmsd, identity_rmsd, worst_rmsd, best_aligned, best_reordered, best_atoms


def _build_fragment_map(atoms: list[str], mol) -> dict[int, list[int]] | None:
    if mol is None:
        return None

    heavy_to_h = {}
    for i, atom_symbol in enumerate(atoms):
        if atom_symbol != 'H':
            atom = mol.GetAtomWithIdx(i)
            bonded_h = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'H']
            heavy_to_h[i] = bonded_h

    if not heavy_to_h or not all(len(h) == 1 for h in heavy_to_h.values()):
        return None

    fragments = {heavy_idx: [heavy_idx] + h_list for heavy_idx, h_list in heavy_to_h.items()}
    return fragments




def _search_fragments(ref_coords, tgt_coords, ref_atoms, tgt_atoms, ref_mol, tgt_mol, allow_reflection):
    ref_frag = _build_fragment_map(ref_atoms, ref_mol)
    tgt_frag = _build_fragment_map(tgt_atoms, tgt_mol)

    if ref_frag is None or tgt_frag is None:
        return None

    best_rmsd = float("inf")
    worst_rmsd = 0.0
    identity = tuple(range(len(ref_atoms)))
    identity_rmsd = None
    best_perm = None
    best_aligned = None
    best_reordered = None
    best_atoms = None

    ref_keys = sorted(ref_frag.keys())
    tgt_keys = sorted(tgt_frag.keys())

    for heavy_perm in permutations(range(len(ref_keys))):
        perm = [0] * len(ref_atoms)
        for i, j in enumerate(heavy_perm):
            ref_block = ref_frag[ref_keys[i]]
            tgt_block = tgt_frag[tgt_keys[j]]
            for k in range(len(ref_block)):
                perm[ref_block[k]] = tgt_block[k]
        perm = tuple(perm)

        reordered_coords = tgt_coords[list(perm)]
        reordered_atoms = [tgt_atoms[i] for i in perm]
        aligned = kabsch_align_only(ref_coords, reordered_coords, ref_atoms, reordered_atoms, use_all_atoms=True, allow_reflection=allow_reflection)
        diff = ref_coords - aligned
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

        if perm == identity:
            identity_rmsd = rmsd
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_perm = perm
            best_aligned = aligned
            best_reordered = reordered_coords
            best_atoms = reordered_atoms
        worst_rmsd = max(worst_rmsd, rmsd)

    if identity_rmsd is None:
        identity_rmsd = worst_rmsd

    return best_perm, best_rmsd, identity_rmsd, worst_rmsd, best_aligned, best_reordered, best_atoms



def _search_isomorphism(reference, target, allow_reflection=False):
    """
    Align target to reference using RDKit graph isomorphism (substructure matching).
    Checks all possible matches, calculates RMSD for each, and returns the best alignment.

    Returns:
        tuple: (best_perm, best_rmsd, identity_rmsd, worst_rmsd, best_aligned, best_reordered, best_atoms)
        or None if no valid matches are found.
    """
    ref_coords = reference.coordinates
    ref_atoms = reference.atoms
    tgt_coords = target.coordinates
    tgt_atoms = target.atoms

    ref_mol = geometry_to_mol(reference)
    tgt_mol = geometry_to_mol(target)

    matches = ref_mol.GetSubstructMatches(tgt_mol, uniquify=False)
    if not matches:
        return None  # No valid permutations found

    best_rmsd = float("inf")
    worst_rmsd = 0.0
    identity = tuple(range(len(ref_atoms)))
    identity_rmsd = None

    best_perm = None
    best_aligned = None
    best_reordered = None
    best_atoms = None

    for perm in matches:
        
        reordered_coords = tgt_coords[list(perm)]
        reordered_atoms = [tgt_atoms[i] for i in perm]

        aligned = kabsch_align_only(
            ref_coords,
            reordered_coords,
            ref_atoms,
            reordered_atoms,
            use_all_atoms=True,
            allow_reflection=allow_reflection
        )

        diff = ref_coords - aligned
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        #print(perm,rmsd)
        if perm == identity:
            identity_rmsd = rmsd

        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_perm = perm
            best_aligned = aligned
            best_reordered = reordered_coords
            best_atoms = reordered_atoms

        worst_rmsd = max(worst_rmsd, rmsd)

    if identity_rmsd is None:
        identity_rmsd = worst_rmsd

    return best_perm, best_rmsd, identity_rmsd, worst_rmsd, best_aligned, best_reordered, best_atoms

from rdkit.Chem import rdFMCS
from scipy.optimize import linear_sum_assignment
from rdkit import Chem 


def _search_mcs_alignment(reference, target, allow_reflection=False):
    """
    Align target to reference using:
    1) Kabsch on MCS atoms
    2) Propagate rotation to all atoms
    3) Hungarian mapping for all atoms of same type
    4) Final Kabsch on fully mapped atoms
    5) RMSD computation
    6) Final Hungarian reordering preserving atom types
    """
    ref_coords = reference.coordinates
    ref_atoms = reference.atoms
    tgt_coords = target.coordinates
    tgt_atoms = target.atoms

    if len(tgt_atoms) != len(ref_atoms):
        return None  # Skip if number of atoms differ

    # Identity RMSD (no alignment)
    identity_diff = ref_coords - tgt_coords
    identity_rmsd = np.sqrt(np.mean(np.sum(identity_diff**2, axis=1)))

    # Convert to RDKit molecules
    ref_mol = geometry_to_mol(reference)
    tgt_mol = geometry_to_mol(target)

    # Step 1: Find MCS
    mcs_result = rdFMCS.FindMCS([ref_mol, tgt_mol])
    if mcs_result.numAtoms == 0:
        return None

    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    matches_ref = ref_mol.GetSubstructMatches(mcs_mol, uniquify=False)
    matches_tgt = tgt_mol.GetSubstructMatches(mcs_mol, uniquify=False)

    best_rmsd = float("inf")
    worst_rmsd = 0.0
    best_perm = None
    best_aligned = None
    best_reordered = None
    best_atoms = None

    for ref_match in matches_ref:
        for tgt_match in matches_tgt:
            # Step 2: Initial Kabsch on MCS atoms
            P_mcs = tgt_coords[list(tgt_match)]
            Q_mcs = ref_coords[list(ref_match)]
            P_cent = P_mcs - P_mcs.mean(axis=0)
            Q_cent = Q_mcs - Q_mcs.mean(axis=0)
            C = np.dot(P_cent.T, Q_cent)
            V, S, Wt = np.linalg.svd(C)
            U = np.dot(V, Wt)
            P_aligned = (tgt_coords - tgt_coords.mean(axis=0)) @ U + Q_mcs.mean(axis=0)

            # Step 3: Hungarian mapping for all atoms by atom type
            unique_types = set(tgt_atoms)
            full_map_test = []
            full_map_ref = []

            for atom_type in unique_types:
                tgt_idx = [i for i, a in enumerate(tgt_atoms) if a == atom_type]
                ref_idx = [i for i, a in enumerate(ref_atoms) if a == atom_type]

                if len(tgt_idx) == 0 or len(ref_idx) == 0:
                    continue

                cost_matrix = np.zeros((len(tgt_idx), len(ref_idx)))
                for i, ti in enumerate(tgt_idx):
                    for j, rj in enumerate(ref_idx):
                        cost_matrix[i, j] = np.linalg.norm(P_aligned[ti] - ref_coords[rj])

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                full_map_test.extend([tgt_idx[i] for i in row_ind])
                full_map_ref.extend([ref_idx[j] for j in col_ind])

            # Step 4: Final Kabsch on fully mapped atoms
            P_full = P_aligned[full_map_test]
            Q_full = ref_coords[full_map_ref]
            P_cent = P_full - P_full.mean(axis=0)
            Q_cent = Q_full - Q_full.mean(axis=0)
            C = np.dot(P_cent.T, Q_cent)
            V, S, Wt = np.linalg.svd(C)
            U = np.dot(V, Wt)
            full_aligned_final = (P_aligned - P_aligned.mean(axis=0)) @ U + Q_full.mean(axis=0)

            # Step 5: Final Hungarian reordering respecting atom types
            final_reordered = np.zeros_like(full_aligned_final)
            for atom_type in unique_types:
                tgt_idx = [i for i, a in enumerate(tgt_atoms) if a == atom_type]
                ref_idx = [i for i, a in enumerate(ref_atoms) if a == atom_type]

                cost_matrix = np.zeros((len(tgt_idx), len(ref_idx)))
                for i, ti in enumerate(tgt_idx):
                    for j, rj in enumerate(ref_idx):
                        cost_matrix[i, j] = np.linalg.norm(full_aligned_final[ti] - ref_coords[rj])

                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, t in zip(row_ind, col_ind):
                    final_reordered[ref_idx[r]] = full_aligned_final[tgt_idx[t]]

            # Step 6: Compute RMSD
            diff = ref_coords - final_reordered
            rmsd_val = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            worst_rmsd = max(worst_rmsd, rmsd_val)

            if rmsd_val < best_rmsd:
                best_rmsd = rmsd_val
                best_perm = tuple(full_map_test)
                best_aligned = final_reordered
                best_reordered = full_aligned_final[full_map_test]

                # <-- NOW best_atoms comes from final Hungarian ordering -->
                best_atoms = [ref_atoms[i] for i in range(len(ref_atoms))]

    return best_perm, best_rmsd, identity_rmsd, worst_rmsd, best_aligned, best_reordered, best_atoms


# ============================================================
# MASTER ALIGNMENT LOOP
# ============================================================

def align_all(reference, targets, automorphisms=None, method: AlignmentMethod = AlignmentMethod.BRUTE_FORCE, allow_reflection=False):
    ref_coords = reference.coordinates
    ref_atoms = reference.atoms
    results = []

    for target in tqdm(targets, desc="Aligning targets", unit="molecule"):
        tgt_coords = target.coordinates
        tgt_atoms = target.atoms

        search_result = None

        if method == AlignmentMethod.IDENTITY:
            search_result = _search_identity(ref_coords, tgt_coords, ref_atoms, tgt_atoms, allow_reflection)

        elif method == AlignmentMethod.BRUTE_FORCE:
            search_result = _search_bruteforce_elementwise(ref_coords, tgt_coords, ref_atoms, tgt_atoms, allow_reflection)

        elif method == AlignmentMethod.AUTOMORPHISM:
            search_result = _search_automorphism(ref_coords, tgt_coords, automorphisms, ref_atoms, tgt_atoms, allow_reflection)
            if search_result is None:
                search_result = _search_bruteforce_elementwise(ref_coords, tgt_coords, ref_atoms, tgt_atoms, allow_reflection)

        elif method == AlignmentMethod.FRAGMENT:
            #ref_mol = xyz_to_rdkit_mol(reference)
            #tgt_mol = xyz_to_rdkit_mol(target)
            #search_result = _search_fragments(ref_coords, tgt_coords, ref_atoms, tgt_atoms, ref_mol, tgt_mol, allow_reflection)
            #if search_result is None:
            #    search_result = _search_bruteforce_elementwise(ref_coords, tgt_coords, ref_atoms, tgt_atoms, allow_reflection)
            print("no")

        elif method == AlignmentMethod.ISOMORPHISM:
            # Graph isomorphism first
            search_result = _search_isomorphism(reference,target, allow_reflection)
            if search_result is None:
                # fallback to automorphism
                print(f"Falling back to automorphism for {target.filename}")
                search_result = _search_automorphism(ref_coords, tgt_coords, automorphisms, ref_atoms, tgt_atoms, allow_reflection)
                if search_result is None:
                    # final fallback to brute-force
                    search_result = _search_bruteforce_elementwise(ref_coords, tgt_coords, ref_atoms, tgt_atoms, allow_reflection)

        elif method == AlignmentMethod.MCS_HUNGARIAN:
            search_result =_search_mcs_alignment(reference,target,allow_reflection)


        else:
            raise ValueError(f"Unknown alignment method: {method}")

        best_perm, best_rmsd, identity_rmsd, worst_rmsd, aligned, reordered_coords, reordered_atoms = search_result

        results.append(
            AlignmentResult(
                geometry=target,
                permutation=best_perm,
                best_rmsd=best_rmsd,
                identity_rmsd=identity_rmsd,
                worst_rmsd=worst_rmsd,
                improvement=worst_rmsd - best_rmsd,
                was_swapped=best_perm != tuple(range(len(best_perm))),
                aligned_coords=aligned,
                reordered_atoms=reordered_atoms,
                reordered_coords=reordered_coords,
            )
        )

    return results