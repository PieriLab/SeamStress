"""Module for processing and displaying molecular geometries."""

from pathlib import Path

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
    heavy_atom_factor: float = 1.0,
    master_reference: str | None = None,
) -> None:
    """
    Load and analyze all geometries from a directory.

    Performs two-stage alignment:
    1. Inter-family: Align all family references to a master reference
    2. Intra-family: Align molecules to their (aligned) family reference

    Args:
        data_dir: Path to directory containing XYZ files
        analyze_connectivity: If True, group by connectivity and show summary
        compute_automorphisms: If True, show automorphisms for template of each family
        output_dir: Directory to write aligned/swapped geometry files (required)
        centroids_dir: Path to directory containing reference structures (optional)
        use_permutations: If True, search for optimal permutations; if False, use identity permutation only
        heavy_atom_factor: Multiplier for heavy atom weights in alignment (default: 1.0).
                          Used in two contexts:
                          (1) Inter-family reference alignment (all families to master)
                          (2) Intra-family final refinement (AFTER best permutation found)
                          Use values > 1.0 (e.g., 10.0, 100.0) to prioritize heavy atoms.
        master_reference: Filename of centroid to use as master reference (e.g., 'ethylene.xyz').
                         If None, the largest family (Family 1) is used as master.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return

    output_dir = Path(output_dir)

    # Load reference structures if provided
    references = {}
    filename_to_hash = {}
    if centroids_dir is not None:
        centroids_dir = Path(centroids_dir)
        print(f"Loading reference structures from: {centroids_dir}")
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
            groups, output_dir, references, use_permutations, heavy_atom_factor,
            master_reference, filename_to_hash
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
    heavy_atom_factor: float = 1.0,
    master_reference: str | None = None,
    filename_to_hash: dict[str, str] = None,
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
        heavy_atom_factor: Multiplier for heavy atom weights in inter-family reference alignment (default: 1.0)
        master_reference: Filename of centroid to use as master reference (e.g., 'ethylene.xyz')
        filename_to_hash: Dictionary mapping reference filenames to connectivity hashes
    """
    if references is None:
        references = {}
    if filename_to_hash is None:
        filename_to_hash = {}
    print(f"\nWriting aligned geometries to: {output_dir}/ (one folder per family)")

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
        print(f"Using heavy_atom_factor = {heavy_atom_factor} for inter-family alignment\n")

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
                heavy_atom_factor=heavy_atom_factor
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

    # =========================================================================
    # STAGE 2: ALIGN MOLECULES WITHIN EACH FAMILY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: INTRA-FAMILY MOLECULE ALIGNMENT")
    print("=" * 80)

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
            heavy_atom_factor=heavy_atom_factor
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
