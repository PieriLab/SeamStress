"""Module for processing and displaying molecular geometries."""

from pathlib import Path
import shutil

from seamstress.alignment import align_geometries_with_automorphisms, kabsch_rmsd, kabsch_align_only
from seamstress.automorphism import get_automorphisms, print_template_automorphisms
from seamstress.connectivity import (
    analyze_connectivity,
    group_by_connectivity,
    print_connectivity_summary,
)
from seamstress.geometry import Geometry, read_all_geometries, read_xyz_file
from seamstress.io_utils import write_xyz_file
from seamstress.rdkit_utils import geometry_to_mol


def prealign_centroids(centroids_dir: Path, align_to: str, output_dir: Path) -> Path:
    """
    Pre-align all centroid structures to a specified reference centroid.

    This creates aligned versions of all centroids in a temporary directory,
    which can then be used for the main alignment workflow.

    Uses two-stage alignment:
    1. Find optimal atom permutation for each centroid relative to master
    2. Apply permutation and align with Kabsch algorithm

    Args:
        centroids_dir: Path to directory containing original centroid XYZ files
        align_to: Filename of the centroid to use as alignment reference (e.g., 'benzene.xyz')
        output_dir: Base output directory (aligned centroids will be in output_dir/prealigned_centroids)

    Returns:
        Path to directory containing pre-aligned centroids
    """
    print("\n" + "=" * 80)
    print("PRE-ALIGNING CENTROIDS")
    print("=" * 80)

    # Create output directory for pre-aligned centroids
    prealigned_dir = output_dir / "prealigned_centroids"
    prealigned_dir.mkdir(parents=True, exist_ok=True)

    # Load the master reference centroid
    master_file = centroids_dir / align_to
    if not master_file.exists():
        available = [f.name for f in centroids_dir.glob("*.xyz")]
        raise ValueError(
            f"Centroid alignment reference '{align_to}' not found in {centroids_dir}.\n"
            f"Available centroids: {', '.join(available)}"
        )

    master_geom = read_xyz_file(master_file)
    print(f"\nAligning all centroids to: {align_to}")
    print(f"Master centroid: {len(master_geom.atoms)} atoms")

    # Get automorphisms for the master centroid for permutation search
    master_mol = geometry_to_mol(master_geom)
    automorphisms = get_automorphisms(master_mol)
    print(f"Symmetry permutations available: {len(automorphisms)}\n")

    # Copy master centroid unchanged
    shutil.copy2(master_file, prealigned_dir / align_to)
    print(f"  {align_to:30s} -> reference (unchanged)")

    # Align all other centroids to the master
    from seamstress.alignment import find_best_permutation_kabsch

    for xyz_file in sorted(centroids_dir.glob("*.xyz")):
        if xyz_file.name == align_to:
            continue

        geom = read_xyz_file(xyz_file)

        # Find best permutation and align (two-stage process)
        auto_idx, rmsd, best_perm, aligned_coords = find_best_permutation_kabsch(
            master_geom.coordinates,
            geom.coordinates,
            automorphisms,
            master_geom.atoms,
            geom.atoms,
            use_permutations=True
        )

        # Get reordered atoms based on best permutation
        reordered_atoms = [geom.atoms[i] for i in best_perm]

        # Write aligned centroid with permutation info
        output_file = prealigned_dir / xyz_file.name
        write_xyz_file(
            output_file,
            reordered_atoms,
            aligned_coords,
            f"Pre-aligned to {align_to} | RMSD: {rmsd:.4f} | {geom.metadata}"
        )

        print(f"  {xyz_file.name:30s} -> RMSD: {rmsd:.4f} Å")

    print(f"\n✓ Pre-aligned centroids saved to: {prealigned_dir}")
    return prealigned_dir


def load_references(centroids_dir: Path) -> tuple[dict[str, Geometry], dict[str, str]]:
    """
    Load reference structures and map them by connectivity.

    Args:
        centroids_dir: Path to directory containing reference XYZ files

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
        conn_info = analyze_connectivity(geom)
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
    use_permutations: bool = True,
    inter_family_heavy_atom_factor: float = 1.0,
    intra_family_heavy_atom_factor: float = 1.0,
    master_reference: str | None = None,
    prealign_centroids_to: str | None = None,
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
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return

    output_dir = Path(output_dir)

    # Pre-align centroids if requested
    if prealign_centroids_to and centroids_dir is not None:
        centroids_dir = Path(centroids_dir)
        centroids_dir = prealign_centroids(centroids_dir, prealign_centroids_to, output_dir)

    # Load reference structures if provided
    references = {}
    filename_to_hash = {}
    if centroids_dir is not None:
        centroids_dir = Path(centroids_dir)
        print(f"\nLoading reference structures from: {centroids_dir}")
        references, filename_to_hash = load_references(centroids_dir)
        print(f"  Found {len(references)} reference structures\n")

    print(f"Reading geometries from: {data_dir}")
    print("=" * 60)

    geometries = read_all_geometries(data_dir)

    print(f"\nFound {len(geometries)} geometry files")

    if analyze_connectivity:
        print("\nAnalyzing connectivity patterns...")
        groups = group_by_connectivity(geometries)
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
        _write_aligned_geometries(
            groups, output_dir, references, use_permutations,
            inter_family_heavy_atom_factor, intra_family_heavy_atom_factor,
            master_reference, filename_to_hash, data_dir, centroids_dir
        )

        return groups
    else:
        _display_geometries(geometries)
        return None


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


def _write_aligned_geometries(
    groups: dict[str, list[Geometry]],
    output_dir: Path,
    references: dict[str, Geometry] = None,
    use_permutations: bool = True,
    inter_family_heavy_atom_factor: float = 1.0,
    intra_family_heavy_atom_factor: float = 1.0,
    master_reference: str | None = None,
    filename_to_hash: dict[str, str] = None,
    data_dir: Path = None,
    centroids_dir: Path = None,
) -> None:
    """
    Write aligned and swapped geometries to output directory.

    Creates one aligned XYZ file per input file with original filename.
    Family information (index and SMILES) is included in the comment line.

    This function performs alignment in two stages:
    1. Inter-family alignment: Align all family references to a master reference
    2. Intra-family alignment: Align individual molecules to their family reference

    Args:
        groups: Dictionary mapping connectivity hash to list of geometries
        output_dir: Base output directory
        references: Dictionary mapping connectivity hash to reference Geometry
        use_permutations: If True, search for optimal permutations; if False, use identity permutation only
        inter_family_heavy_atom_factor: Multiplier for heavy atoms in inter-family (centroid-to-centroid) alignment (default: 1.0)
        intra_family_heavy_atom_factor: Multiplier for heavy atoms in intra-family (molecule-to-centroid) alignment (default: 1.0)
        master_reference: Filename of centroid to use as master reference (e.g., 'ethylene.xyz')
        filename_to_hash: Dictionary mapping reference filenames to connectivity hashes
        data_dir: Original input data directory (for copying raw spawns)
        centroids_dir: Original centroids directory (for copying raw centroids)
    """
    if references is None:
        references = {}
    if filename_to_hash is None:
        filename_to_hash = {}
    print(f"\nWriting aligned geometries to: {output_dir}/ (one folder per family)")

    # Copy original files to raw_spawns and raw_centroids
    if data_dir and data_dir.exists():
        raw_spawns_dir = output_dir / "raw_spawns"
        raw_spawns_dir.mkdir(parents=True, exist_ok=True)
        for xyz_file in data_dir.glob("*.xyz"):
            shutil.copy2(xyz_file, raw_spawns_dir / xyz_file.name)
        print(f"✓ Copied {len(list(data_dir.glob('*.xyz')))} original spawn files to: {raw_spawns_dir}")

    if centroids_dir and centroids_dir.exists():
        raw_centroids_dir = output_dir / "raw_centroids"
        raw_centroids_dir.mkdir(parents=True, exist_ok=True)
        for xyz_file in centroids_dir.glob("*.xyz"):
            shutil.copy2(xyz_file, raw_centroids_dir / xyz_file.name)
        print(f"✓ Copied {len(list(centroids_dir.glob('*.xyz')))} original centroid files to: {raw_centroids_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track high RMSDs for warnings
    high_rmsd_threshold = 0.5  # Ångströms
    high_rmsd_files = []
    all_rmsds = []

    # Open log file for swap effectiveness
    log_file = output_dir / "swap_effectiveness.log"
    log = open(log_file, "w")
    log.write("=" * 120 + "\n")
    log.write("ATOM SWAPPING EFFECTIVENESS LOG\n")
    log.write("=" * 120 + "\n")
    log.write(
        "\nShows which atom permutation was chosen and RMSD improvement vs worst/identity permutation.\n"
    )
    log.write(
        "Improvement = How much RMSD was reduced by choosing optimal swap (removes symmetry artifacts).\n\n"
    )

    # =========================================================================
    # STAGE 1: COLLECT AND ALIGN FAMILY REFERENCES
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: INTER-FAMILY REFERENCE ALIGNMENT")
    print("=" * 80)

    # First pass: select references for all families
    family_references = {}
    sorted_families = list(sorted(groups.items(), key=lambda x: -len(x[1])))

    for i, (conn_hash, geoms) in enumerate(sorted_families, 1):
        # Select reference: use predefined if available, else fall back to geoms[0]
        if conn_hash in references:
            family_references[i] = references[conn_hash]
            ref_source = f"predefined ({references[conn_hash].filename})"
        else:
            family_references[i] = geoms[0]
            ref_source = f"first geometry ({geoms[0].filename})"

        print(f"Family {i}: Reference from {ref_source}")

    # Determine which family to use as master
    if len(family_references) > 1:
        # If user specified a master reference file, find its family
        if master_reference and filename_to_hash:
            if master_reference not in filename_to_hash:
                print(f"\nWarning: Master reference '{master_reference}' not found in centroids.")
                print(f"Available references: {', '.join(filename_to_hash.keys())}")
                print("Falling back to Family 1 (largest family) as master.\n")
                master_family_id = 1
            else:
                # Find which family corresponds to this reference
                master_conn_hash = filename_to_hash[master_reference]
                master_family_id = None
                for i, (conn_hash, geoms) in enumerate(sorted_families, 1):
                    if conn_hash == master_conn_hash:
                        master_family_id = i
                        break

                if master_family_id is None:
                    print(f"\nWarning: No family found matching master reference '{master_reference}'")
                    print("Falling back to Family 1 (largest family) as master.\n")
                    master_family_id = 1
                else:
                    print(f"\nUsing specified master reference: {master_reference} (Family {master_family_id})")
        else:
            # Default to Family 1 (largest family)
            master_family_id = 1
            print(f"\nUsing Family {master_family_id} (largest family) as master reference")

        master_ref = family_references[master_family_id]
        print(f"Aligning all family references to Family {master_family_id}")
        print(f"Using inter_family_heavy_atom_factor = {inter_family_heavy_atom_factor} for inter-family alignment\n")

        # Store aligned reference coordinates (master stays unchanged)
        aligned_family_refs = {master_family_id: master_ref}

        for family_id, ref_geom in family_references.items():
            if family_id == master_family_id:
                continue

            # Align this family's reference to the master reference
            aligned_coords = kabsch_align_only(
                master_ref.coordinates,
                ref_geom.coordinates,
                master_ref.atoms,
                ref_geom.atoms,
                use_all_atoms=True,
                weight_type="mass",
                heavy_atom_factor=inter_family_heavy_atom_factor
            )

            # Calculate RMSD to master
            from seamstress.alignment import kabsch_rmsd
            rmsd, _ = kabsch_rmsd(
                master_ref.coordinates,
                aligned_coords,
                master_ref.atoms,
                ref_geom.atoms
            )

            # Create new Geometry with aligned coordinates
            aligned_ref = Geometry(
                atoms=ref_geom.atoms,
                coordinates=aligned_coords,
                metadata=ref_geom.metadata,
                filename=ref_geom.filename
            )
            aligned_family_refs[family_id] = aligned_ref

            print(f"  Family {family_id} -> Master: RMSD = {rmsd:.4f} Å")

        # Use aligned references
        family_references = aligned_family_refs
    else:
        print("Only one family found - no inter-family alignment needed")

    # Write combined aligned_centroids.xyz file
    aligned_centroids_file = output_dir / "aligned_centroids.xyz"
    with open(aligned_centroids_file, 'w') as f:
        for family_id in sorted(family_references.keys()):
            ref = family_references[family_id]
            conn_hash = sorted_families[family_id - 1][0]  # Get the connectivity hash for this family

            # Write atom count
            f.write(f"{len(ref.atoms)}\n")
            # Write comment line
            f.write(f"Family_{family_id} {conn_hash} | Aligned Reference | {ref.metadata}\n")
            # Write atoms
            for atom, coord in zip(ref.atoms, ref.coordinates):
                f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")

    print(f"✓ Wrote combined aligned centroids: {aligned_centroids_file.name} ({len(family_references)} centroids)")

    # =========================================================================
    # STAGE 2: ALIGN MOLECULES WITHIN EACH FAMILY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: INTRA-FAMILY MOLECULE ALIGNMENT")
    print("=" * 80)

    # Track all aligned molecules for combined file
    all_aligned_molecules = []

    for i, (conn_hash, geoms) in enumerate(sorted_families, 1):
        print(f"\nFamily {i}: {conn_hash} ({len(geoms)} molecules)")

        # Create family-specific output directory
        family_dir = output_dir / f"family_{i}"
        family_dir.mkdir(parents=True, exist_ok=True)

        # Get the aligned reference from Stage 1
        reference = family_references[i]

        # Write family header to log
        log.write("\n" + "=" * 120 + "\n")
        log.write(f"FAMILY {i}: {conn_hash}\n")
        log.write("=" * 120 + "\n")
        log.write(f"Reference: {reference.filename} (aligned to master)\n")
        log.write(f"Molecules: {len(geoms)}\n")
        log.write(f"Output directory: {family_dir}\n")

        ref_mol = geometry_to_mol(reference)
        automorphisms = get_automorphisms(ref_mol)

        log.write(f"Symmetry permutations available: {len(automorphisms)}\n")
        log.write(
            "\nMolecules where atoms were swapped (non-identity permutation used):\n"
        )
        log.write(
            f"\n{'Molecule':<30} {'Best RMSD':<12} {'Worst RMSD':<12} {'Identity RMSD':<15} {'Improvement':<15} {'Swap Used':<20}\n"
        )
        log.write("-" * 120 + "\n")

        swaps_in_family = 0

        # Write reference structure
        write_xyz_file(
            family_dir / "reference.xyz",
            reference.atoms,
            reference.coordinates,
            f"Family_{i} {conn_hash} | Reference | {reference.metadata}",
        )

        # Align ALL geometries to the reference (including geoms[0] if using predefined ref)
        aligned_results = align_geometries_with_automorphisms(
            reference, geoms, automorphisms, use_permutations=use_permutations,
            heavy_atom_factor=intra_family_heavy_atom_factor
        )

        ref_coords = reference.coordinates
        ref_atoms = reference.atoms

        if aligned_results:

            for result in aligned_results:
                orig_geom = result["geometry"]
                best_rmsd = result["rmsd"]
                best_perm = result["permutation"]
                all_rmsds.append(best_rmsd)

                # Calculate RMSD for all permutations to find worst and identity
                target_coords = orig_geom.coordinates
                target_atoms = orig_geom.atoms
                all_perm_rmsds = []
                identity_rmsd = None

                for perm in automorphisms:
                    permuted_coords = target_coords[list(perm), :]
                    permuted_atoms = [target_atoms[i] for i in perm]
                    perm_rmsd, _ = kabsch_rmsd(
                        ref_coords, permuted_coords, ref_atoms, permuted_atoms
                    )
                    all_perm_rmsds.append(perm_rmsd)

                    # Check if this is identity permutation
                    if perm == tuple(range(len(perm))):
                        identity_rmsd = perm_rmsd

                worst_rmsd = max(all_perm_rmsds)
                if identity_rmsd is None:
                    identity_rmsd = worst_rmsd  # Fallback

                # Calculate improvement
                improvement = worst_rmsd - best_rmsd

                # Check if identity permutation was used (no swap)
                identity_perm = tuple(range(len(best_perm)))
                is_swapped = best_perm != identity_perm

                # Only log if a swap actually happened
                if is_swapped:
                    swaps_in_family += 1
                    # Format permutation for display
                    perm_str = str(best_perm)
                    if len(perm_str) > 18:
                        perm_str = perm_str[:15] + "..."

                    # Log this molecule with swap effectiveness
                    log.write(
                        f"{orig_geom.filename:<30} {best_rmsd:<12.4f} {worst_rmsd:<12.4f} {identity_rmsd:<15.4f} {improvement:<15.4f} {perm_str:<20}\n"
                    )

                # Track high RMSD for warnings
                if best_rmsd > high_rmsd_threshold:
                    high_rmsd_files.append((orig_geom.filename, i, best_rmsd))

                # Write aligned (swapped + Kabsch aligned) with original filename
                write_xyz_file(
                    family_dir / orig_geom.filename,
                    result["reordered_atoms"],
                    result["aligned_coords"],
                    f"Family_{i} {conn_hash} | RMSD: {best_rmsd:.4f} | {orig_geom.metadata}",
                )

                # Track this molecule for combined aligned_spawns.xyz file
                all_aligned_molecules.append({
                    "family_id": i,
                    "conn_hash": conn_hash,
                    "atoms": result["reordered_atoms"],
                    "coords": result["aligned_coords"],
                    "rmsd": best_rmsd,
                    "metadata": orig_geom.metadata,
                })

            # Write combined XYZ file with all aligned molecules in this family
            combined_file = family_dir / f"family_{i}.xyz"
            with open(combined_file, 'w') as f:
                for result in aligned_results:
                    atoms = result["reordered_atoms"]
                    coords = result["aligned_coords"]
                    orig_geom = result["geometry"]
                    best_rmsd = result["rmsd"]

                    # Write atom count
                    f.write(f"{len(atoms)}\n")
                    # Write comment line
                    f.write(f"Family_{i} {conn_hash} | RMSD: {best_rmsd:.4f} | {orig_geom.metadata}\n")
                    # Write atoms
                    for atom, coord in zip(atoms, coords):
                        f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")

            print(f"  ✓ Wrote combined file: {combined_file.name} ({len(aligned_results)} molecules)")

        # Summary for this family
        if swaps_in_family == 0:
            log.write(
                "  (No swaps performed - all molecules used identity permutation)\n"
            )
        else:
            log.write(f"\n  Total swaps in this family: {swaps_in_family}\n")

    # Close log file
    log.write("\n" + "=" * 120 + "\n")
    log.write("END OF LOG\n")
    log.write("=" * 120 + "\n")
    log.close()

    # Write combined aligned_spawns.xyz file with all aligned molecules
    aligned_spawns_file = output_dir / "aligned_spawns.xyz"
    with open(aligned_spawns_file, 'w') as f:
        for mol in all_aligned_molecules:
            # Write atom count
            f.write(f"{len(mol['atoms'])}\n")
            # Write comment line
            f.write(f"Family_{mol['family_id']} {mol['conn_hash']} | RMSD: {mol['rmsd']:.4f} | {mol['metadata']}\n")
            # Write atoms
            for atom, coord in zip(mol['atoms'], mol['coords']):
                f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")

    print(f"✓ Wrote combined aligned spawns: {aligned_spawns_file.name} ({len(all_aligned_molecules)} molecules)")

    total_aligned = len(all_rmsds)

    # Print summary to terminal
    num_families = len(groups)
    print(
        f"\n✓ Wrote {sum(len(geoms) for geoms in groups.values())} aligned geometries into {num_families} family folder(s)"
    )
    print(f"✓ Swap effectiveness log written to: {log_file}")

    if all_rmsds:
        mean_rmsd = sum(all_rmsds) / len(all_rmsds)
        max_rmsd = max(all_rmsds)

        print("\nAlignment Statistics:")
        print(f"  Aligned molecules: {total_aligned}")
        print(f"  Mean RMSD: {mean_rmsd:.4f} Å")
        print(f"  Max RMSD: {max_rmsd:.4f} Å")

        # Warn about high RMSDs
        if high_rmsd_files:
            print(
                f"\n⚠️  WARNING: {len(high_rmsd_files)} file(s) have unusually high RMSD (> {high_rmsd_threshold} Å):"
            )
            for filename, family, rmsd in sorted(high_rmsd_files, key=lambda x: -x[2])[
                :10
            ]:
                print(f"     {filename} (Family {family}): {rmsd:.4f} Å")
            if len(high_rmsd_files) > 10:
                print(f"     ... and {len(high_rmsd_files) - 10} more")
            print(f"\n  Check {log_file} to see swap effectiveness for each molecule")
