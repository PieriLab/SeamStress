"""Module for processing and displaying molecular geometries."""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from seamstress.alignment import align_all, AlignmentMethod
from seamstress.automorphism import get_automorphisms, print_template_automorphisms
from seamstress.connectivity import (
    analyze_connectivity,
    group_by_connectivity,
    print_connectivity_summary,
)
from seamstress.geometry import Geometry, read_all_geometries, read_xyz_file
from seamstress.io_utils import write_xyz_file
from seamstress.rdkit_utils import geometry_to_mol


def _compose_automorphisms(
    master_automorphisms: list[tuple[int, ...]],
    family_automorphisms: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """
    Compose master (reactant) and family (product) automorphism sets.

    For every permutation a in master_automorphisms and every permutation b in
    family_automorphisms, computes the composition b∘a (i.e. composed[i] = b[a[i]]).
    The result is deduplicated and returned as a list.

    This covers the full valid permutation space across a bond-change event:
    - master_automorphisms captures reactant symmetry (e.g. the 12-fold symmetry of benzene)
    - family_automorphisms captures product symmetry (e.g. the H-swap at an sp3 carbon)
    """
    n = len(master_automorphisms[0])
    combined: set[tuple[int, ...]] = set()
    for a in master_automorphisms:
        for b in family_automorphisms:
            combined.add(tuple(b[a[i]] for i in range(n)))
    return list(combined)


def prealign_centroids(
    centroids_dir: Path,
    align_to: str,
    output_dir: Path,
    allow_reflection: bool = False,
) -> Path:
    """
    Pre-align all centroid structures to a specified reference centroid.

    Two-stage alignment:
        1. Optimal atom permutation search
        2. Kabsch alignment using that permutation

    Uses AUTOMORPHISM strategy with automatic brute-force fallback.

    Returns:
        Path to directory containing pre-aligned centroids
    """

    print("\n" + "=" * 80)
    print("PRE-ALIGNING CENTROIDS")
    print("=" * 80)

    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------
    prealigned_dir = output_dir / "prealigned_centroids"
    prealigned_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Load master reference centroid
    # ------------------------------------------------------------
    master_file = centroids_dir / align_to
    if not master_file.exists():
        available = [f.name for f in centroids_dir.glob("*.xyz")]
        raise ValueError(
            f"Centroid alignment reference '{align_to}' not found.\n"
            f"Available centroids: {', '.join(available)}"
        )

    master_geom = read_xyz_file(master_file)

    print(f"\nAligning all centroids to: {align_to}")
    print(f"Master centroid: {len(master_geom.atoms)} atoms")

    # ------------------------------------------------------------
    # Get automorphisms for master
    # ------------------------------------------------------------
    master_mol = geometry_to_mol(master_geom)
    automorphisms = get_automorphisms(master_mol)

    print(f"Symmetry permutations available: {len(automorphisms)}\n")

    # ------------------------------------------------------------
    # Copy master centroid unchanged
    # ------------------------------------------------------------
    shutil.copy2(master_file, prealigned_dir / align_to)
    print(f"  {align_to:30s} -> reference (unchanged)")

    # ------------------------------------------------------------
    # Align all other centroids
    # ------------------------------------------------------------
    for xyz_file in sorted(centroids_dir.glob("*.xyz")):
        if xyz_file.name == align_to:
            continue

        geom = read_xyz_file(xyz_file)

        # Use new strategy-based aligner
        results = align_all(
            reference=master_geom,
            targets=[geom],
            automorphisms=automorphisms,
            method=AlignmentMethod.MCS_HUNGARIAN,
            allow_reflection=allow_reflection,
        )

        result = results[0]

        # --------------------------------------------------------
        # Write aligned centroid
        # --------------------------------------------------------
        output_file = prealigned_dir / xyz_file.name

        write_xyz_file(
            output_file,
            result.reordered_atoms,
            result.aligned_coords,
            (
                f"Pre-aligned to {align_to} | "
                f"RMSD: {result.best_rmsd:.4f} Å | "
                f"Swapped: {result.was_swapped} | "
                f"{geom.metadata}"
            ),
        )

        print(
            f"  {xyz_file.name:30s} "
            f"-> RMSD: {result.best_rmsd:.4f} Å "
            f"| swapped: {result.was_swapped}"
        )

    print(f"\n✓ Pre-aligned centroids saved to: {prealigned_dir}")

    return prealigned_dir


#def meci_label_assignment(meci)
#    return

def _align_all_to_all(
    geometries: list,
    output_dir: Path,
    permutation_method: str,
    heavy_atom_factor: float = 1.0,
    allow_reflection: bool = False,
    bond_threshold: float = 1.3,
    data_dir: Path | None = None,
) -> None:

    print("\n" + "=" * 80)
    print("ALIGN-ALL-TO-ALL MODE")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    family_dir = output_dir / "family_1"
    family_dir.mkdir(parents=True, exist_ok=True)

    # Copy raw files if requested
    if data_dir and data_dir.exists():
        raw_spawns_dir = output_dir / "raw_spawns"
        raw_spawns_dir.mkdir(exist_ok=True, parents=True)
        for f in data_dir.glob("*.xyz"):
            shutil.copy2(f, raw_spawns_dir / f.name)
        print(f"Copied {len(list(data_dir.glob('*.xyz')))} raw spawn files to {raw_spawns_dir}")

    METHOD_MAP = {
        "identity": AlignmentMethod.IDENTITY,
        "brute-force": AlignmentMethod.BRUTE_FORCE,
        "automorphism": AlignmentMethod.AUTOMORPHISM,
        "fragment": AlignmentMethod.FRAGMENT,
        "isomorphism": AlignmentMethod.ISOMORPHISM,
        "mcs-hungarian": AlignmentMethod.MCS_HUNGARIAN,
        "double-isomorphism": AlignmentMethod.DOUBLE_ISOMORPHISM,
    }

    if permutation_method not in METHOD_MAP:
        raise ValueError(
            f"Unknown permutation_method '{permutation_method}'. "
            f"Choose from: {', '.join(METHOD_MAP.keys())}"
        )

    method = METHOD_MAP[permutation_method]

    n = len(geometries)
    rmsd_matrix = np.zeros((n, n))

    # Extract sample names (adjust if needed)
    names = [g.filename for g in geometries]

    # ---- One-loop symmetric RMSD computation ----
    for i in range(n):
        print(f"{geometries[i].filename}:  {i}/{n}")
        targets = geometries[i + 1:]
        if not targets:
            continue

        if permutation_method == 'automorphism':
            ref_mol = geometry_to_mol(geometries[i])
            automorphisms = get_automorphisms(ref_mol)
           
        else:
            automorphisms = None 

        results = align_all(
            reference=geometries[i],
            targets=targets,
            automorphisms=automorphisms,
            method=method,
            allow_reflection=allow_reflection,
        )

        for result in results:
            target_name = result.geometry.filename  # get the name of the target molecule
            rmsd = result.best_rmsd

        # Fill RMSD matrix using names
            rmsd_matrix[names.index(geometries[i].filename), names.index(target_name)] = rmsd
            rmsd_matrix[names.index(target_name), names.index(geometries[i].filename)] = rmsd

    # ---- Create labeled DataFrame ----
    rmsd_df = pd.DataFrame(
        rmsd_matrix,
        index=names,
        columns=names,
    )

    # ---- Save matrix ----
    output_path = output_dir / "rmsd_matrix.csv"
    rmsd_df.to_csv(output_path, float_format="%.6f")

    print(f"\nSaved labeled RMSD matrix to: {output_path}")

    return

import shutil
import numpy as np
from pathlib import Path

def _gpa_alignment(
    geometries: list,
    centroids_dir: Path,
    master_reference: str,
    output_dir: Path,
    permutation_method: str | None = None,
    heavy_atom_factor: float = 1.0,
    allow_reflection: bool = False,
    bond_threshold: float = 1.3,
    data_dir: Path | None = None,
    max_iterations: int = 20,
    convergence_threshold: float = 1e-4,
) -> None:
    """
    Iterative GPA alignment of geometries to a master reference, 
    producing aligned files, combined family XYZ, RMSD statistics, 
    and reference at each iteration.
    """
    print("\n" + "="*80)
    print("GPA_alignment_mode")
    print("="*80)
    print(f"Treating all {len(geometries)} geometries as a single family.")
    print(f"Reference centroid: {master_reference}\n")

    # Load reference centroid
    centroid_file = centroids_dir / master_reference
    if not centroid_file.exists():
        available = [f.name for f in centroids_dir.glob("*.xyz")]
        raise FileNotFoundError(
            f"Centroid '{master_reference}' not found in {centroids_dir}.\n"
            f"Available centroids: {', '.join(available)}"
        )
    reference = read_xyz_file(centroid_file)
    print(f"Loaded centroid: {master_reference} ({len(reference.atoms)} atoms)")

    # Prepare output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    family_dir = output_dir / "family_1"
    family_dir.mkdir(parents=True, exist_ok=True)

    # Copy raw files if requested
    if data_dir and data_dir.exists():
        raw_spawns_dir = output_dir / "raw_spawns"
        raw_spawns_dir.mkdir(exist_ok=True, parents=True)
        for f in data_dir.glob("*.xyz"):
            shutil.copy2(f, raw_spawns_dir / f.name)
        print(f"Copied {len(list(data_dir.glob('*.xyz')))} raw spawn files to {raw_spawns_dir}")

    if centroids_dir and centroids_dir.exists():
        raw_centroids_dir = output_dir / "raw_centroids"
        raw_centroids_dir.mkdir(exist_ok=True, parents=True)
        for f in centroids_dir.glob("*.xyz"):
            shutil.copy2(f, raw_centroids_dir / f.name)
        print(f"Copied {len(list(centroids_dir.glob('*.xyz')))} raw centroid files to {raw_centroids_dir}")

    # Convert centroid to RDKit molecule and get automorphisms
    centroid_mol = geometry_to_mol(reference)
    automorphisms = get_automorphisms(centroid_mol)
    conn_info = analyze_connectivity(reference, cov_factor=bond_threshold)
    conn_hash = conn_info.connectivity_hash
    print(f"Connectivity hash: {conn_hash}")
    print(f"Symmetry permutations available: {len(automorphisms)}\n")

    # Map string method to AlignmentMethod enum
    METHOD_MAP = {
        "identity": AlignmentMethod.IDENTITY,
        "brute-force": AlignmentMethod.BRUTE_FORCE,
        "automorphism": AlignmentMethod.AUTOMORPHISM,
        "fragment": AlignmentMethod.FRAGMENT,
        "isomorphism": AlignmentMethod.ISOMORPHISM,
        "mcs-hungarian": AlignmentMethod.MCS_HUNGARIAN
    }
    if permutation_method not in METHOD_MAP:
        raise ValueError(
            f"Unknown permutation_method '{permutation_method}'. "
            f"Choose from: {', '.join(METHOD_MAP.keys())}"
        )
    method = METHOD_MAP[permutation_method]

    # Write initial reference
    write_xyz_file(
        family_dir / "reference_iter_0.xyz",
        reference.atoms,
        reference.coordinates,
        f"Family_1 {conn_hash} | Reference | {reference.metadata}"
    )

    # Iterative GPA
    for iteration in range(1, max_iterations + 1):
        print("="*80)
        print(f"GPA Iteration {iteration}")
        print("="*80)

        # Align all geometries to current reference
        aligned_results = align_all(
            reference=reference,
            targets=geometries,
            automorphisms=automorphisms,
            method=method,
            allow_reflection=allow_reflection,
        )

        # Track RMSDs and write per-molecule files
        all_rmsds = []
        high_rmsd_threshold = 5.0
        high_rmsd_files = []
        all_aligned_molecules = []

        for result in aligned_results:
            orig_geom = result.geometry
            best_rmsd = result.best_rmsd
            aligned_coords = result.aligned_coords
            reordered_atoms = result.reordered_atoms

            all_rmsds.append(best_rmsd)
            if best_rmsd > high_rmsd_threshold:
                high_rmsd_files.append((orig_geom.filename, best_rmsd))

            # Write per-molecule aligned file
            write_xyz_file(
                family_dir / orig_geom.filename,
                reordered_atoms,
                aligned_coords,
                f"Family_1 {conn_hash} | RMSD: {best_rmsd:.4f} | {orig_geom.metadata}"
            )

            all_aligned_molecules.append({
                "atoms": reordered_atoms,
                "coords": aligned_coords,
                "rmsd": best_rmsd,
                "metadata": orig_geom.metadata
            })

        # Write combined family XYZ for this iteration
        family_xyz_file = family_dir / f"family_1_iter_{iteration}.xyz"
        with open(family_xyz_file, "w") as f:
            for mol in all_aligned_molecules:
                f.write(f"{len(mol['atoms'])}\n")
                f.write(f"Family_1 {conn_hash} | RMSD: {mol['rmsd']:.4f} | {mol['metadata']}\n")
                for atom, coord in zip(mol['atoms'], mol['coords']):
                    f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
        print(f"✓ Wrote combined family file: {family_xyz_file.name}")

        # Compute mean coordinates
        all_coords = np.array([mol['coords'] for mol in all_aligned_molecules])
        mean_coords = np.mean(all_coords, axis=0)

        # RMSD of mean vs previous reference
        rmsd_mean = np.sqrt(np.mean((mean_coords - reference.coordinates) ** 2))
        print(f"RMSD to previous reference: {rmsd_mean:.6f} Å")

        # Write reference for this iteration
        reference_file = family_dir / f"reference_iter_{iteration}.xyz"
        write_xyz_file(
            reference_file,
            reference.atoms,
            mean_coords,
            f"Family_1 {conn_hash} | Mean Reference Iter {iteration} | {reference.metadata}"
        )
        print(f"✓ Wrote reference for iteration {iteration}: {reference_file.name}")

        # Check convergence
        if rmsd_mean < convergence_threshold:
            print(f"Converged after {iteration} iterations.")
            # Update reference to mean coordinates for final output
            reference = Geometry(
                atoms=reference.atoms.copy(),
                coordinates=mean_coords.copy(),
                metadata=reference.metadata,
                filename=reference.filename
            )
            break

        # Update reference for next iteration
        reference = Geometry(
            atoms=reference.atoms.copy(),
            coordinates=mean_coords.copy(),
            metadata=reference.metadata,
            filename=reference.filename
        )

    # Final combined aligned file in output root
    combined_file = output_dir / "aligned_spawns.xyz"
    with open(combined_file, 'w') as f:
        for mol in all_aligned_molecules:
            f.write(f"{len(mol['atoms'])}\n")
            f.write(f"RMSD: {mol['rmsd']:.4f} | {mol['metadata']}\n")
            for atom, coord in zip(mol['atoms'], mol['coords']):
                f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
    print(f"✓ Wrote final combined aligned spawns: {combined_file.name} ({len(all_aligned_molecules)} molecules)")

    # Summary RMSD statistics
    if all_rmsds:
        mean_rmsd = sum(all_rmsds) / len(all_rmsds)
        max_rmsd = max(all_rmsds)
        print(f"\nMean RMSD: {mean_rmsd:.4f} Å, Max RMSD: {max_rmsd:.4f} Å")
        if mean_rmsd > 1.0:
            print(f"⚠️  WARNING: Mean RMSD > 1.0 Å, consider family-based alignment.")
        if high_rmsd_files:
            print(f"⚠️  {len(high_rmsd_files)} files exceed {high_rmsd_threshold} Å RMSD:")
            for fname, r in sorted(high_rmsd_files, key=lambda x: -x[1])[:10]:
                print(f"   {fname}: {r:.4f} Å")

    print("GPA alignment complete.\n")



def _align_all_to_single_centroid(
    geometries: list,
    centroids_dir: Path,
    master_reference: str,
    output_dir: Path,
    permutation_method: str | None = None,
    heavy_atom_factor: float = 1.0,
    allow_reflection: bool = False,
    bond_threshold: float = 1.3,
    data_dir: Path | None = None,
) -> None:
    """
    Align all geometries to a single reference centroid using a chosen permutation method.

    Args:
        geometries: List of Geometry objects to align.
        centroids_dir: Directory containing centroid XYZ files.
        centroid_filename: Reference centroid filename (e.g., 'benzene.xyz').
        output_dir: Directory where aligned geometries and centroids will be written.
        permutation_method: Which alignment method to use:
            "fragment", "identity", "automorphism", or "brute_force".
        heavy_atom_factor: Weighting factor for heavy atoms in final alignment.
        allow_reflection: Allow improper rotations if True.
        bond_threshold: Covalent bond scaling factor for connectivity analysis.
        data_dir: Optional original data directory to copy raw files from.
    """
    print("\n" + "="*80)
    print("ALIGN-ALL-TO-CENTROID MODE")
    print("="*80)
    print(f"Treating all {len(geometries)} geometries as a single family.")
    print(f"Reference centroid: {master_reference}\n")

    # Load reference centroid
    centroid_file = centroids_dir / master_reference
    if not centroid_file.exists():
        available = [f.name for f in centroids_dir.glob("*.xyz")]
        raise FileNotFoundError(
            f"Centroid '{master_reference}' not found in {centroids_dir}.\n"
            f"Available centroids: {', '.join(available)}"
        )
    centroid = read_xyz_file(centroid_file)
    print(f"Loaded centroid: {master_reference} ({len(centroid.atoms)} atoms)")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    family_dir = output_dir / "family_1"
    family_dir.mkdir(parents=True, exist_ok=True)

    # Copy raw files if requested
    if data_dir and data_dir.exists():
        raw_spawns_dir = output_dir / "raw_spawns"
        raw_spawns_dir.mkdir(exist_ok=True, parents=True)
        for f in data_dir.glob("*.xyz"):
            shutil.copy2(f, raw_spawns_dir / f.name)
        print(f"Copied {len(list(data_dir.glob('*.xyz')))} raw spawn files to {raw_spawns_dir}")

    if centroids_dir and centroids_dir.exists():
        raw_centroids_dir = output_dir / "raw_centroids"
        raw_centroids_dir.mkdir(exist_ok=True, parents=True)
        for f in centroids_dir.glob("*.xyz"):
            shutil.copy2(f, raw_centroids_dir / f.name)
        print(f"Copied {len(list(centroids_dir.glob('*.xyz')))} raw centroid files to {raw_centroids_dir}")

    # Convert centroid to RDKit molecule and get automorphisms
    centroid_mol = geometry_to_mol(centroid)
    automorphisms = get_automorphisms(centroid_mol)
    conn_info = analyze_connectivity(centroid, cov_factor=bond_threshold)
    conn_hash = conn_info.connectivity_hash
    print(f"Connectivity hash: {conn_hash}")
    print(f"Symmetry permutations available: {len(automorphisms)}\n")

    # Map string method to AlignmentMethod enum
    METHOD_MAP = {
        "identity": AlignmentMethod.IDENTITY,
        "brute-force": AlignmentMethod.BRUTE_FORCE,
        "automorphism": AlignmentMethod.AUTOMORPHISM,
        "fragment": AlignmentMethod.FRAGMENT,
        "isomorphism": AlignmentMethod.ISOMORPHISM,
        "mcs-hungarian": AlignmentMethod.MCS_HUNGARIAN,
        "double-isomorphism": AlignmentMethod.DOUBLE_ISOMORPHISM,
    }
    if permutation_method not in METHOD_MAP:
        raise ValueError(
            f"Unknown permutation_method '{permutation_method}'. "
            f"Choose from: {', '.join(METHOD_MAP.keys())}"
        )
    method = METHOD_MAP[permutation_method]

    # Write reference centroid
    write_xyz_file(
        family_dir / "reference.xyz",
        centroid.atoms,
        centroid.coordinates,
        f"Family_1 {conn_hash} | Reference | {centroid.metadata}"
    )

    # Align all geometries
    print("="*80)
    print("ALIGNING GEOMETRIES")
    print("="*80)
    print(f"Using alignment method: {method.value}\n")

    if method == AlignmentMethod.DOUBLE_ISOMORPHISM:
        # Group geometries by connectivity so we compose master automorphisms with
        # the automorphisms of each product topology exactly once per unique connectivity.
        master_automorphisms = automorphisms  # centroid = reactant
        groups = group_by_connectivity(geometries, cov_factor=bond_threshold)
        results_by_filename = {}
        for group_hash, group_geoms in groups.items():
            group_mol = geometry_to_mol(group_geoms[0])
            family_automorphisms = get_automorphisms(group_mol) if group_mol is not None \
                else [tuple(range(len(group_geoms[0].atoms)))]
            combined = _compose_automorphisms(master_automorphisms, family_automorphisms)
            print(f"  Connectivity {group_hash[:40]}...: "
                  f"{len(family_automorphisms)} product perms × "
                  f"{len(master_automorphisms)} master perms → {len(combined)} combined")
            group_results = align_all(
                reference=centroid,
                targets=group_geoms,
                automorphisms=combined,
                method=method,
                allow_reflection=allow_reflection,
            )
            for result in group_results:
                results_by_filename[result.geometry.filename] = result
        aligned_results = [results_by_filename[g.filename] for g in geometries]
    else:
        aligned_results = align_all(
            reference=centroid,
            targets=geometries,
            automorphisms=automorphisms,
            method=method,
            allow_reflection=allow_reflection,
        )

    all_rmsds = []
    high_rmsd_threshold = 5.0
    high_rmsd_files = []
    all_aligned_molecules = []

    # Process results
    for result in aligned_results:
        orig_geom = result.geometry
        best_rmsd = result.best_rmsd
        aligned_coords = result.aligned_coords
        reordered_atoms = result.reordered_atoms

        all_rmsds.append(best_rmsd)

        # Track high RMSD per molecule
        if best_rmsd > high_rmsd_threshold:
            high_rmsd_files.append((orig_geom.filename, best_rmsd))

        # Write aligned geometry
        write_xyz_file(
            family_dir / orig_geom.filename,
            reordered_atoms,
            aligned_coords,
            f"Family_1 {conn_hash} | RMSD: {best_rmsd:.4f} | {orig_geom.metadata}"
        )

        # Collect for combined family file
        all_aligned_molecules.append({
            "atoms": reordered_atoms,
            "coords": aligned_coords,
            "rmsd": best_rmsd,
            "metadata": orig_geom.metadata
        })

    # Write combined family XYZ
    family_xyz_file = family_dir / "family_1.xyz"
    with open(family_xyz_file, "w") as f:
        for mol in all_aligned_molecules:
            f.write(f"{len(mol['atoms'])}\n")
            f.write(f"Family_1 {conn_hash} | RMSD: {mol['rmsd']:.4f} | {mol['metadata']}\n")
            for atom, coord in zip(mol['atoms'], mol['coords']):
                f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
    print(f"✓ Wrote combined family file: {family_xyz_file.name}")



    # Write combined XYZ file in root output folder
    combined_file = output_dir / "aligned_spawns.xyz"
    with open(combined_file, 'w') as f:
        for mol in all_aligned_molecules:
            f.write(f"{len(mol['atoms'])}\n")
            f.write(f"RMSD: {mol['rmsd']:.4f} | {mol['metadata']}\n")
            for atom, coord in zip(mol['atoms'], mol['coords']):
                f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")

    print(f"✓ Wrote combined aligned spawns: {combined_file.name} ({len(all_aligned_molecules)} molecules)")

    # Overall RMSD statistics
    if all_rmsds:
        mean_rmsd = sum(all_rmsds) / len(all_rmsds)
        max_rmsd = max(all_rmsds)
        print(f"\nMean RMSD: {mean_rmsd:.4f} Å, Max RMSD: {max_rmsd:.4f} Å")
        if mean_rmsd > 1.0:
            print(f"⚠️  WARNING: Mean RMSD > 1.0 Å, consider family-based alignment.")
        if high_rmsd_files:
            print(f"⚠️  {len(high_rmsd_files)} files exceed {high_rmsd_threshold} Å RMSD:")
            for fname, r in sorted(high_rmsd_files, key=lambda x: -x[1])[:10]:
                print(f"   {fname}: {r:.4f} Å")


def _multi_ref_align_family(
    geometries: list,
    centroids_dir: Path,
    master_reference: str,
    output_dir: Path,
    references: list,
    groups: dict,
    permutation_method: str | None = None,
    intra_family_heavy_atom_factor: float = 1.0,
    inter_family_heavy_atom_factor: float = 1.0,
    allow_reflection: bool = False,
    bond_threshold: float = 1.3,
    data_dir: Path | None = None,
    filename_to_hash: dict[str, str] = None,
) -> None:

    METHOD_MAP = {
        "identity": AlignmentMethod.IDENTITY,
        "brute-force": AlignmentMethod.BRUTE_FORCE,
        "automorphism": AlignmentMethod.AUTOMORPHISM,
        "fragment": AlignmentMethod.FRAGMENT,
        "isomorphism": AlignmentMethod.ISOMORPHISM,
        "mcs-hungarian": AlignmentMethod.MCS_HUNGARIAN,
        "double-isomorphism": AlignmentMethod.DOUBLE_ISOMORPHISM,
    }

    if permutation_method not in METHOD_MAP:
        raise ValueError(
            f"Unknown permutation_method '{permutation_method}'. "
            f"Choose from: {', '.join(METHOD_MAP.keys())}"
        )

    method = METHOD_MAP[permutation_method]

    if references is None:
        references = {}

    if filename_to_hash is None:
        filename_to_hash = {}

    output_dir.mkdir(parents=True, exist_ok=True)

    # GLOBAL TRACKERS

    global_all_aligned = []
    global_all_rmsds = []
    high_rmsd_threshold = 5.0
    high_rmsd_files = []

    log_file = output_dir / "swap_effectiveness.log"
    log = open(log_file, "w")

    # =========================================================================
    print("STAGE 1: INTER-FAMILY REFERENCE ALIGNMENT")
    # =========================================================================

    family_references = {}
    sorted_families = list(sorted(groups.items(), key=lambda x: -len(x[1])))

    for i, (conn_hash, geoms) in enumerate(sorted_families, 1):
        if conn_hash in references:
            family_references[i] = references[conn_hash]
        else:
            family_references[i] = geoms[0]

    if len(family_references) > 1:

        master_family_id = 1
        if master_reference and filename_to_hash:
            if master_reference in filename_to_hash:
                master_hash = filename_to_hash[master_reference]
                for i, (conn_hash, _) in enumerate(sorted_families, 1):
                    if conn_hash == master_hash:
                        master_family_id = i
                        break

        master_ref = family_references[master_family_id]

        master_mol = geometry_to_mol(master_ref)
        master_automorphisms = get_automorphisms(master_mol)
        automorphisms = master_automorphisms  # used for stage-1 inter-family alignment

        aligned_refs = {master_family_id: master_ref}

        for family_id, ref_geom in family_references.items():
            if family_id == master_family_id:
                continue

            aligned = align_all(
                reference=master_ref,
                targets=[ref_geom],
                automorphisms=automorphisms,
                method=method,
                allow_reflection=allow_reflection,
            )[0]

            aligned_refs[family_id] = Geometry(
                atoms=aligned.reordered_atoms,
                coordinates=aligned.aligned_coords,
                metadata=ref_geom.metadata,
                filename=ref_geom.filename,
            )

        family_references = aligned_refs

    # =========================================================================
    print("STAGE 2: INTRA-FAMILY ALIGNMENT")
    # =========================================================================

    for i, (conn_hash, geoms) in enumerate(sorted_families, 1):
        print(f"Aligning Family {i}, {conn_hash}")
        family_dir = output_dir / f"family_{i}"
        family_dir.mkdir(parents=True, exist_ok=True)

        reference = family_references[i]

        ref_mol = geometry_to_mol(reference)
        family_automorphisms = get_automorphisms(ref_mol)

        if method == AlignmentMethod.DOUBLE_ISOMORPHISM:
            # Compose master (reactant) automorphisms with family (product) automorphisms.
            # master_automorphisms may not exist if there is only one family.
            _master = master_automorphisms if len(family_references) > 1 \
                else family_automorphisms
            automorphisms = _compose_automorphisms(_master, family_automorphisms)
            print(f"  Double-isomorphism: {len(_master)} master × "
                  f"{len(family_automorphisms)} family → {len(automorphisms)} combined perms")
        else:
            automorphisms = family_automorphisms

        # Write reference
        write_xyz_file(
            family_dir / "reference.xyz",
            reference.atoms,
            reference.coordinates,
            f"Family_{i} {conn_hash} | Reference | {reference.metadata}",
        )

        aligned_results = align_all(
            reference=reference,
            targets=geoms,
            automorphisms=automorphisms,
            method=method,
            allow_reflection=allow_reflection,
        )

        family_aligned = []

        for result in aligned_results:

            best_rmsd = result.best_rmsd
            global_all_rmsds.append(best_rmsd)

            if best_rmsd > high_rmsd_threshold:
                high_rmsd_files.append(
                    (result.geometry.filename, i, best_rmsd)
                )

            write_xyz_file(
                family_dir / result.geometry.filename,
                result.reordered_atoms,
                result.aligned_coords,
                f"Family_{i} {conn_hash} | RMSD: {best_rmsd:.4f} | {result.geometry.metadata}",
            )

            entry = {
                "family_id": i,
                "conn_hash": conn_hash,
                "atoms": result.reordered_atoms,
                "coords": result.aligned_coords,
                "rmsd": best_rmsd,
                "metadata": result.geometry.metadata,
            }

            family_aligned.append(entry)
            global_all_aligned.append(entry)

        # Write family combined file
        combined_file = family_dir / f"family_{i}.xyz"

        with open(combined_file, "w") as f:
            for mol in family_aligned:
                f.write(f"{len(mol['atoms'])}\n")
                f.write(
                    f"Family_{mol['family_id']} {mol['conn_hash']} | "
                    f"RMSD: {mol['rmsd']:.4f} | {mol['metadata']}\n"
                )
                for atom, coord in zip(mol["atoms"], mol["coords"]):
                    f.write(
                        f"{atom:2s} "
                        f"{coord[0]:12.6f} "
                        f"{coord[1]:12.6f} "
                        f"{coord[2]:12.6f}\n"
                    )

    log.close()

    # GLOBAL COMBINED FILE
    

    aligned_spawns_file = output_dir / "aligned_spawns.xyz"

    with open(aligned_spawns_file, "w") as f:
        for mol in global_all_aligned:
            f.write(f"{len(mol['atoms'])}\n")
            f.write(
                f"Family_{mol['family_id']} {mol['conn_hash']} | "
                f"RMSD: {mol['rmsd']:.4f} | {mol['metadata']}\n"
            )
            for atom, coord in zip(mol["atoms"], mol["coords"]):
                f.write(
                    f"{atom:2s} "
                    f"{coord[0]:12.6f} "
                    f"{coord[1]:12.6f} "
                    f"{coord[2]:12.6f}\n"
                )

    
    # SUMMARY

    if global_all_rmsds:
        mean_rmsd = sum(global_all_rmsds) / len(global_all_rmsds)
        max_rmsd = max(global_all_rmsds)

        print("\nAlignment Statistics:")
        print(f"  Aligned molecules: {len(global_all_rmsds)}")
        print(f"  Mean RMSD: {mean_rmsd:.4f} Å")
        print(f"  Max RMSD: {max_rmsd:.4f} Å")

        if high_rmsd_files:
            print(
                f"\nWARNING: {len(high_rmsd_files)} molecule(s) "
                f"exceeded {high_rmsd_threshold} Å RMSD:"
            )
            for filename, family_id, rmsd in sorted(
                high_rmsd_files, key=lambda x: -x[2]
            )[:10]:
                print(f"  {filename} (Family {family_id}): {rmsd:.4f} Å")
    

def _multi_ref_align_rmsd(
    geometries: list,
    centroids_dir: Path,
    output_dir: Path,
    permutation_method: str | None = None,
    intra_family_heavy_atom_factor: float = 1.0,
    inter_family_heavy_atom_factor: float = 1.0,
    allow_reflection: bool = False,
    bond_threshold: float = 1.3,
    master_reference: str | None = None,
) -> None:
    """
    Multi-reference alignment mode (per-geometry outer loop).

    Each geometry is aligned against ALL centroid references.
    The alignment corresponding to the LOWEST RMSD is selected.
    """

    print("\n" + "=" * 80)
    print("MULTI-REFERENCE ALIGNMENT MODE (PER-GEOMETRY OUTER LOOP)")
    print("=" * 80)
    print(f"Total geometries: {len(geometries)}")

    # ------------------------------------------------------------------
    # Load all centroids
    # ------------------------------------------------------------------
    centroid_files = sorted(centroids_dir.glob("*.xyz"))
    if not centroid_files:
        raise FileNotFoundError(f"No centroid XYZ files found in {centroids_dir}")

    print(f"Found {len(centroid_files)} centroid references.\n")

    centroids = {}
    centroid_automorphisms = {}
    centroid_hashes = {}

    for cf in centroid_files:
        centroid = read_xyz_file(cf)
        mol = geometry_to_mol(centroid)
        automorphisms = get_automorphisms(mol)
        conn_info = analyze_connectivity(centroid, cov_factor=bond_threshold)

        centroids[cf.name] = centroid
        centroid_automorphisms[cf.name] = automorphisms
        centroid_hashes[cf.name] = conn_info.connectivity_hash

        print(f"Loaded centroid: {cf.name}")
        print(f"  Atoms: {len(centroid.atoms)}")
        print(f"  Connectivity hash: {centroid_hashes[cf.name]}")
        print(f"  Symmetry permutations: {len(automorphisms)}\n")

    # ------------------------------------------------------------------
    # Map alignment method
    # ------------------------------------------------------------------
    METHOD_MAP = {
        "identity": AlignmentMethod.IDENTITY,
        "brute-force": AlignmentMethod.BRUTE_FORCE,
        "automorphism": AlignmentMethod.AUTOMORPHISM,
        "fragment": AlignmentMethod.FRAGMENT,
        "isomorphism": AlignmentMethod.ISOMORPHISM,
        "mcs-hungarian": AlignmentMethod.MCS_HUNGARIAN,
        "double-isomorphism": AlignmentMethod.DOUBLE_ISOMORPHISM,
    }

    if permutation_method not in METHOD_MAP:
        raise ValueError(
            f"Unknown permutation_method '{permutation_method}'. "
            f"Choose from: {', '.join(METHOD_MAP.keys())}"
        )

    method = METHOD_MAP[permutation_method]
    print(f"Using alignment method: {method.value}\n")

    # For double-isomorphism, compose master (reactant) automorphisms with each
    # centroid's (product) automorphisms once upfront.
    if method == AlignmentMethod.DOUBLE_ISOMORPHISM:
        if master_reference is None or master_reference not in centroid_automorphisms:
            print("  Warning: --master not specified or not found for double-isomorphism. "
                  "Falling back to plain automorphism search.")
        else:
            master_autos = centroid_automorphisms[master_reference]
            for cf_name in list(centroid_automorphisms.keys()):
                combined = _compose_automorphisms(master_autos, centroid_automorphisms[cf_name])
                print(f"  {cf_name}: {len(master_autos)} master × "
                      f"{len(centroid_automorphisms[cf_name])} centroid "
                      f"→ {len(combined)} combined perms")
                centroid_automorphisms[cf_name] = combined

    # ------------------------------------------------------------------
    # Prepare output — flat directory, consistent with align_dataset.py
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # GLOBAL TRACKERS
    # ------------------------------------------------------------------
    global_all_rmsds = []
    high_rmsd_threshold = 5.0
    high_rmsd_files = []

    print("=" * 80)
    print("ALIGNING GEOMETRIES TO ALL REFERENCES (PER-GEOMETRY LOOP)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # GEOMETRY OUTER LOOP
    # ------------------------------------------------------------------
    all_aligned_molecules = []

    for geom in tqdm(geometries, desc="  multi_ref", unit="geom"):
        best_rmsd = float("inf")
        best_result = None
        best_centroid_name = None

        for centroid_name, centroid in centroids.items():
            result = align_all(
                reference=centroid,
                targets=[geom],
                automorphisms=centroid_automorphisms[centroid_name],
                method=method,
                allow_reflection=allow_reflection,
            )[0]

            if result.best_rmsd < best_rmsd:
                best_rmsd = result.best_rmsd
                best_result = result
                best_centroid_name = centroid_name

        global_all_rmsds.append(best_rmsd)

        if best_rmsd > high_rmsd_threshold:
            high_rmsd_files.append((geom.filename, best_rmsd))

        # Write to flat output directory (all geometries share the same
        # reference frame because all centroids are pre-aligned to master)
        write_xyz_file(
            output_dir / geom.filename,
            best_result.reordered_atoms,
            best_result.aligned_coords,
            f"{geom.metadata} | closest_meci={best_centroid_name} rmsd={best_rmsd:.4f}",
        )

        all_aligned_molecules.append({
            "atoms": best_result.reordered_atoms,
            "coords": best_result.aligned_coords,
            "rmsd": best_rmsd,
            "metadata": geom.metadata,
            "reference": best_centroid_name,
        })

    # ------------------------------------------------------------------
    # Combined trajectory file
    # ------------------------------------------------------------------
    trajectory_file = output_dir.parent / f"{output_dir.name}_trajectory.xyz"
    with open(trajectory_file, "w") as f:
        for mol in all_aligned_molecules:
            f.write(f"{len(mol['atoms'])}\n")
            f.write(
                f"{mol['metadata']} | closest_meci={mol['reference']} "
                f"rmsd={mol['rmsd']:.4f}\n"
            )
            for atom, coord in zip(mol["atoms"], mol["coords"]):
                f.write(f" {atom}  {coord[0]:18.9f}  {coord[1]:18.9f}  {coord[2]:18.9f}\n")

    print(f"\n✓ Trajectory → {trajectory_file}  ({len(all_aligned_molecules)} frames)")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    if global_all_rmsds:
        mean_rmsd = sum(global_all_rmsds) / len(global_all_rmsds)
        max_rmsd = max(global_all_rmsds)

        print(f"Mean RMSD: {mean_rmsd:.4f} Å")
        print(f"Max RMSD:  {max_rmsd:.4f} Å")

        if high_rmsd_files:
            print(f"\n⚠️  {len(high_rmsd_files)} geometries exceed "
                  f"{high_rmsd_threshold} Å:")
            for fname, r in sorted(high_rmsd_files, key=lambda x: -x[1])[:10]:
                print(f"   {fname}: {r:.4f} Å")

    print("\n✓ Multi-reference alignment complete.\n")
    

def load_references(
    centroids_dir: Path, bond_threshold: float = 1.3
) -> tuple[dict[str, Geometry], dict[str, str]]:
    """
    Load reference structures and map them by connectivity.

    Args:
        centroids_dir: Path to directory containing reference XYZ files
        bond_threshold: Covalent factor multiplier for bond detection (default: 1.3)

    Returns:
        Tuple of (references dict, filename_to_hash dict)
        - references: Dictionary mapping connectivity hash to reference Geometry
        - filename_to_hash: Dictionary mapping filename to connectivity hash
    """
    references = {}
    filename_to_hash = {}

    if not centroids_dir.exists():
        return references, filename_to_hash

    for xyz_file in centroids_dir.glob("*.xyz"):
        geom = read_xyz_file(xyz_file)
        conn_info = analyze_connectivity(geom, cov_factor=bond_threshold)
        references[conn_info.connectivity_hash] = geom
        filename_to_hash[xyz_file.name] = conn_info.connectivity_hash
        print(f"  Loaded reference: {xyz_file.name} -> {conn_info.connectivity_hash}")

    return references, filename_to_hash


def process_geometries(
    data_dir: Path | str,
    analyze_connectivity: bool = True,
    compute_automorphisms: bool = False,
    output_dir: Path | str = None,
    centroids_dir: Path | str = None,
    permutation_method: str | None = None, 
    inter_family_heavy_atom_factor: float = 1.0,
    intra_family_heavy_atom_factor: float = 1.0,
    master_reference: str | None = None,
    prealign_centroids_to: str | None = None,
    alignment_type: str | None = None, 
    allow_reflection: bool = False,
    bond_threshold: float = 1.3,
) -> None:
    """
    Load and analyze all geometries from a directory.

    Performs optional pre-alignment and two-stage alignment:
    0. (Optional) Pre-align all centroids to a user-specified centroid
    1. Inter-family: Align all family references to a master reference
    2. Intra-family: Align molecules to their (aligned) family reference

    Args:
        data_dir: Path to directory containing XYZ files
        analyze_connectivity: If True, group by connectivity and show summary
        compute_automorphisms: If True, show automorphisms for template of each family
        output_dir: Directory to write aligned/swapped geometry files (required)
        centroids_dir: Path to directory containing reference structures (optional)
        use_permutations: If True, search for optimal permutations; if False, use identity permutation only
        inter_family_heavy_atom_factor: Multiplier for heavy atoms when aligning family references to each other (default: 1.0).
                                        Use values > 1.0 (e.g., 10.0, 100.0) to prioritize heavy atoms in centroid alignment.
        intra_family_heavy_atom_factor: Multiplier for heavy atoms when aligning molecules to their family reference (default: 1.0).
                                        Applied AFTER best permutation is found.
                                        Use values > 1.0 to prioritize heavy atoms in final molecule alignment.
        master_reference: Filename of centroid to use as master reference (e.g., 'ethylene.xyz').
                         If None, the largest family (Family 1) is used as master.
        prealign_centroids_to: Filename of centroid to pre-align all centroids to before main workflow (e.g., 'benzene.xyz').
                               If specified, all centroids are aligned to this reference first.
        use_fragment_permutations: If True, use fragment-based permutation search (treats heavy atoms + bonded H as rigid units).
                                  Only applicable when all heavy atoms have exactly 1 hydrogen (e.g., benzene).
                                  Provides ~720x speedup for benzene-like molecules. Falls back to standard mode if not applicable.
        align_all_to_centroid: Filename of centroid to align ALL spawning points to, bypassing family detection (e.g., 'benzene.xyz').
                              Treats all geometries as one family. Useful when all points have same connectivity.
                              Warns if mean RMSD > 1.0 Å. Requires centroids_dir to be specified.
        bond_threshold: Covalent factor multiplier for bond detection (default: 1.3, RDKit default).
                       Bond threshold = (cov_radius_1 + cov_radius_2) × bond_threshold.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return

    output_dir = Path(output_dir)

    # Pre-align centroids if requested
    if prealign_centroids_to and centroids_dir is not None:
        centroids_dir = Path(centroids_dir)
        centroids_dir = prealign_centroids(centroids_dir, prealign_centroids_to, output_dir,allow_reflection=allow_reflection)

    # Load reference structures if provided
    references = {}
    filename_to_hash = {}
    if centroids_dir is not None:
        centroids_dir = Path(centroids_dir)
        print(f"\nLoading reference structures from: {centroids_dir}")
        references, filename_to_hash = load_references(centroids_dir, bond_threshold=bond_threshold)
        print(f"  Found {len(references)} reference structures\n")

    print(f"Reading geometries from: {data_dir}")
    print("=" * 60)

    geometries = read_all_geometries(data_dir)

    print(f"\nFound {len(geometries)} geometry files")


    if analyze_connectivity:
        print("\nAnalyzing connectivity patterns...")
        groups = group_by_connectivity(geometries, cov_factor=bond_threshold)
        print_connectivity_summary(groups)

        if compute_automorphisms:
            print("\n\n" + "=" * 80)
            print("AUTOMORPHISMS FOR TEMPLATE MEMBERS OF EACH FAMILY")
            print("=" * 80)

            # Iterate through groups sorted by size (largest first)
            for i, (conn_hash, geoms) in enumerate(
                sorted(groups.items(), key=lambda x: -len(x[1])), 1
            ):
                print(f"\n\nFamily {i}: {conn_hash} ({len(geoms)} molecules)")
                print_template_automorphisms(geoms[0])

        # Write aligned/swapped geometries to output folder
        print("\n\n" + "=" * 80)
        print("WRITING ALIGNED/SWAPPED GEOMETRIES")
        print("=" * 80)


    

    if alignment_type == 'single-reference':
        ### add permutation type 
        _align_all_to_single_centroid(
            geometries=geometries,
            centroids_dir=Path(centroids_dir),
            master_reference=master_reference,
            output_dir=output_dir,
            permutation_method=permutation_method,            # "fragment_permutations", "identity", etc.
            heavy_atom_factor=intra_family_heavy_atom_factor,
            allow_reflection=allow_reflection,
            bond_threshold=bond_threshold,
            data_dir=data_dir
        )
        return 
    
    


    if alignment_type =='multireference-family':
        _multi_ref_align_family(
            geometries=geometries,
            centroids_dir=Path(centroids_dir),
            master_reference=master_reference,
            references=references,
            output_dir=output_dir,
            groups=groups,
            permutation_method=permutation_method,
            inter_family_heavy_atom_factor=inter_family_heavy_atom_factor,
            intra_family_heavy_atom_factor=intra_family_heavy_atom_factor,
            allow_reflection=allow_reflection,
            filename_to_hash=filename_to_hash
        )
        return


    if alignment_type == 'multireference-rmsd':
        _multi_ref_align_rmsd(
            geometries=geometries,
            centroids_dir=Path(centroids_dir),
            output_dir=output_dir,
            permutation_method=permutation_method,
            inter_family_heavy_atom_factor=inter_family_heavy_atom_factor,
            intra_family_heavy_atom_factor=intra_family_heavy_atom_factor,
            allow_reflection=allow_reflection,
            master_reference=master_reference,
        )
        return 
    


    if alignment_type == 'all-to-all-rmsd':
        _align_all_to_all(
            geometries=geometries,
            output_dir=output_dir,
            permutation_method=permutation_method,
            allow_reflection=allow_reflection,
            data_dir=data_dir
        )
        return 
    

    if alignment_type == 'gpa':
        _gpa_alignment(
            geometries=geometries,
            centroids_dir=Path(centroids_dir),
            master_reference=master_reference,
            output_dir=output_dir,
            permutation_method=permutation_method,            
            heavy_atom_factor=intra_family_heavy_atom_factor,
            allow_reflection=allow_reflection,
            bond_threshold=bond_threshold,
            data_dir=data_dir
        )
        return 



def _display_geometries(geometries: list[Geometry]) -> None:
    """
    Display all geometries with formatting.

    Args:
        geometries: List of Geometry objects to display
    """
    for i, geom in enumerate(geometries, 1):
        print(f"\n{'=' * 60}")
        print(f"Geometry {i}/{len(geometries)}")
        print("=" * 60)
        print(geom)


