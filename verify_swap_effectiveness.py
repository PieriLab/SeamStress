"""Verify that atom swapping effectively removes symmetry-induced RMSD variations."""

import sys
from pathlib import Path
import numpy as np
from elisa_spawns.geometry import read_xyz_file
from elisa_spawns.rdkit_utils import geometry_to_mol
from elisa_spawns.automorphism import get_automorphisms
from elisa_spawns.alignment import kabsch_rmsd


def calculate_rmsd_without_swap(geom1, geom2):
    """Calculate RMSD without any atom swapping (identity mapping)."""
    coords1 = geom1.coordinates
    coords2 = geom2.coordinates

    rmsd, _ = kabsch_rmsd(coords1, coords2)
    return rmsd


def calculate_all_possible_rmsds(geom1, geom2, automorphisms):
    """
    Calculate RMSD for all possible automorphism permutations.

    Returns:
        min_rmsd: Best possible RMSD with optimal swapping
        max_rmsd: Worst RMSD with bad swapping
        identity_rmsd: RMSD without any swapping
        num_perms: Number of permutations tested
    """
    coords1 = geom1.coordinates
    coords2 = geom2.coordinates

    rmsds = []
    identity_rmsd = None

    for perm in automorphisms:
        # Apply permutation
        permuted_coords2 = coords2[list(perm), :]
        rmsd, _ = kabsch_rmsd(coords1, permuted_coords2)
        rmsds.append(rmsd)

        # Check if this is identity permutation
        if perm == tuple(range(len(perm))):
            identity_rmsd = rmsd

    return min(rmsds), max(rmsds), identity_rmsd, len(rmsds)


def analyze_family_swap_effectiveness(output_dir: Path, family_num: int):
    """
    Analyze how effective the swapping is for a given family.
    Compares RMSD before and after swapping.
    """
    # Find all molecules in this family
    molecules = []
    reference = None

    for xyz_file in sorted(output_dir.glob("*.xyz")):
        geom = read_xyz_file(xyz_file)
        if f"Family_{family_num}" in geom.metadata:
            if "Reference" in geom.metadata:
                reference = geom
            else:
                molecules.append(geom)

    if not reference:
        print(f"Error: No reference found for Family {family_num}")
        return

    if not molecules:
        print(f"Error: No molecules found for Family {family_num}")
        return

    print(f"\n{'='*90}")
    print(f"SWAP EFFECTIVENESS ANALYSIS - Family {family_num}")
    print(f"{'='*90}")
    print(f"Reference: {reference.filename}")
    print(f"Analyzing {len(molecules)} molecules")

    # Get automorphisms
    ref_mol = geometry_to_mol(reference)
    automorphisms = get_automorphisms(ref_mol)
    print(f"Number of symmetry permutations: {len(automorphisms)}")

    # Read corresponding input files to compare
    input_dir = output_dir.parent / "input"  # Assume input is in ../input
    if not input_dir.exists():
        # Try other common locations
        for possible_input in [Path("./input"), Path("./data/spawns"), output_dir.parent]:
            if possible_input.exists() and possible_input.is_dir():
                input_dir = possible_input
                break

    print(f"\n{'Molecule':<25} {'Output RMSD':<15} {'Best Possible':<15} {'Worst Possible':<15} {'Improvement':<15}")
    print("-" * 90)

    improvements = []
    reduction_factors = []

    for mol_geom in molecules:
        # The output RMSD is already in the metadata
        output_rmsd = None
        if "RMSD:" in mol_geom.metadata:
            rmsd_str = mol_geom.metadata.split("RMSD:")[1].split("|")[0].strip()
            output_rmsd = float(rmsd_str)

        # Calculate what the RMSD range would be without smart swapping
        min_rmsd, max_rmsd, identity_rmsd, num_perms = calculate_all_possible_rmsds(
            reference, mol_geom, automorphisms
        )

        # The output should have the minimum RMSD (best permutation was chosen)
        if output_rmsd is not None:
            improvement = (max_rmsd - output_rmsd) if max_rmsd > 0 else 0
            reduction = output_rmsd / max_rmsd if max_rmsd > 0 else 1.0

            improvements.append(improvement)
            reduction_factors.append(reduction)

            status = "✓" if abs(output_rmsd - min_rmsd) < 0.01 else "⚠️"
            print(f"{mol_geom.filename:<25} {output_rmsd:<15.4f} {min_rmsd:<15.4f} {max_rmsd:<15.4f} {improvement:>10.4f} Å {status}")
        else:
            print(f"{mol_geom.filename:<25} {'N/A':<15} {min_rmsd:<15.4f} {max_rmsd:<15.4f} {'N/A':<15}")

    # Summary statistics
    if improvements:
        print(f"\n{'='*90}")
        print("SUMMARY:")
        print(f"  Mean RMSD improvement: {np.mean(improvements):.4f} Å")
        print(f"  Max RMSD improvement:  {np.max(improvements):.4f} Å")
        print(f"  Mean reduction factor: {np.mean(reduction_factors):.2f}x (closer to 0 is better)")
        print(f"\n  Interpretation:")
        if np.mean(improvements) > 0.1:
            print(f"    ✓ EXCELLENT: Swapping removes {np.mean(improvements):.3f} Å of symmetry-induced RMSD variation")
            print(f"    ✓ Without swapping, clustering would group different conformations incorrectly")
        elif np.mean(improvements) > 0.01:
            print(f"    ✓ GOOD: Swapping improves alignment by {np.mean(improvements):.3f} Å")
        else:
            print(f"    ⚠️  LOW: Swapping only improves by {np.mean(improvements):.3f} Å - molecule may have low symmetry")


def compare_clustering_before_after(output_dir: Path, family_num: int, rmsd_threshold: float = 0.5):
    """
    Simulate how clustering would differ with vs without swapping.
    """
    # Find all molecules in this family
    molecules = []
    reference = None

    for xyz_file in sorted(output_dir.glob("*.xyz")):
        geom = read_xyz_file(xyz_file)
        if f"Family_{family_num}" in geom.metadata:
            if "Reference" in geom.metadata:
                reference = geom
            else:
                molecules.append(geom)

    if not reference or len(molecules) < 2:
        print("Need at least 2 molecules to simulate clustering")
        return

    ref_mol = geometry_to_mol(reference)
    automorphisms = get_automorphisms(ref_mol)

    print(f"\n{'='*90}")
    print(f"CLUSTERING SIMULATION - Family {family_num}")
    print(f"{'='*90}")
    print(f"RMSD threshold for clustering: {rmsd_threshold} Å")
    print(f"Number of molecules: {len(molecules)}")

    # Calculate pairwise RMSDs with and without swapping
    print(f"\nPairwise RMSD Matrix:")
    print(f"\n{'Pair':<30} {'With Swap (✓)':<20} {'Without Swap (✗)':<20} {'Difference':<15}")
    print("-" * 90)

    for i in range(len(molecules)):
        for j in range(i + 1, len(molecules)):
            mol_i = molecules[i]
            mol_j = molecules[j]

            # RMSD with optimal swapping
            min_rmsd, max_rmsd, identity_rmsd, _ = calculate_all_possible_rmsds(
                mol_i, mol_j, automorphisms
            )

            # RMSD without swapping (identity permutation)
            coords_i = mol_i.coordinates
            coords_j = mol_j.coordinates
            rmsd_no_swap, _ = kabsch_rmsd(coords_i, coords_j)

            difference = rmsd_no_swap - min_rmsd

            pair_name = f"{mol_i.filename[:12]} - {mol_j.filename[:12]}"
            print(f"{pair_name:<30} {min_rmsd:<20.4f} {rmsd_no_swap:<20.4f} {difference:>10.4f} Å")

    print(f"\n{'='*90}")
    print("CLUSTERING IMPACT:")
    print(f"  Without swapping: Different orientations of the SAME structure might be")
    print(f"                    clustered as DIFFERENT conformations (false positives)")
    print(f"  With swapping:    Symmetry-equivalent orientations correctly identified")
    print(f"                    as the SAME structure")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python verify_swap_effectiveness.py <output_dir> <family_num>")
        print("\nExample:")
        print("  python verify_swap_effectiveness.py ./output 1")
        print("\nThis will show:")
        print("  - How much RMSD improvement the swapping provides")
        print("  - Whether the optimal permutation was chosen")
        print("  - Impact on clustering results")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    family_num = int(sys.argv[2])

    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)

    # Main analysis
    analyze_family_swap_effectiveness(output_dir, family_num)

    # Clustering simulation
    print("\n")
    compare_clustering_before_after(output_dir, family_num)
