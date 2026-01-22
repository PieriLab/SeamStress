"""Module for processing and displaying molecular geometries."""

from pathlib import Path

from seamstress.alignment import align_geometries_with_automorphisms, kabsch_rmsd
from seamstress.automorphism import get_automorphisms, print_template_automorphisms
from seamstress.connectivity import (
    analyze_connectivity,
    group_by_connectivity,
    print_connectivity_summary,
)
from seamstress.geometry import Geometry, read_all_geometries, read_xyz_file
from seamstress.io_utils import write_xyz_file
from seamstress.rdkit_utils import geometry_to_mol


def load_references(centroids_dir: Path) -> dict[str, Geometry]:
    """
    Load reference structures and map them by connectivity.

    Args:
        centroids_dir: Path to directory containing reference XYZ files

    Returns:
        Dictionary mapping connectivity hash to reference Geometry
    """
    references = {}

    if not centroids_dir.exists():
        return references

    for xyz_file in centroids_dir.glob("*.xyz"):
        geom = read_xyz_file(xyz_file)
        conn_info = analyze_connectivity(geom)
        references[conn_info.connectivity_hash] = geom
        print(f"  Loaded reference: {xyz_file.name} -> {conn_info.connectivity_hash}")

    return references


def process_geometries(
    data_dir: Path | str,
    analyze_connectivity: bool = True,
    compute_automorphisms: bool = False,
    output_dir: Path | str = None,
    centroids_dir: Path | str = None,
    use_permutations: bool = True,
) -> None:
    """
    Load and analyze all geometries from a directory.

    Args:
        data_dir: Path to directory containing XYZ files
        analyze_connectivity: If True, group by connectivity and show summary
        compute_automorphisms: If True, show automorphisms for template of each family
        output_dir: Directory to write aligned/swapped geometry files (required)
        centroids_dir: Path to directory containing reference structures (optional)
        use_permutations: If True, search for optimal permutations; if False, use identity permutation only
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return

    output_dir = Path(output_dir)

    # Load reference structures if provided
    references = {}
    if centroids_dir is not None:
        centroids_dir = Path(centroids_dir)
        print(f"Loading reference structures from: {centroids_dir}")
        references = load_references(centroids_dir)
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
        _write_aligned_geometries(groups, output_dir, references, use_permutations)

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
) -> None:
    """
    Write aligned and swapped geometries to output directory.

    Creates one aligned XYZ file per input file with original filename.
    Family information (index and SMILES) is included in the comment line.

    Args:
        groups: Dictionary mapping connectivity hash to list of geometries
        output_dir: Base output directory
        references: Dictionary mapping connectivity hash to reference Geometry
        use_permutations: If True, search for optimal permutations; if False, use identity permutation only
    """
    if references is None:
        references = {}
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

    for i, (conn_hash, geoms) in enumerate(
        sorted(groups.items(), key=lambda x: -len(x[1])), 1
    ):
        print(f"\nFamily {i}: {conn_hash} ({len(geoms)} molecules)")

        # Create family-specific output directory
        family_dir = output_dir / f"family_{i}"
        family_dir.mkdir(parents=True, exist_ok=True)

        # Select reference: use predefined if available, else fall back to geoms[0]
        if conn_hash in references:
            reference = references[conn_hash]
            ref_source = f"predefined ({reference.filename})"
        else:
            reference = geoms[0]
            ref_source = f"first geometry ({reference.filename})"

        # Write family header to log
        log.write("\n" + "=" * 120 + "\n")
        log.write(f"FAMILY {i}: {conn_hash}\n")
        log.write("=" * 120 + "\n")
        log.write(f"Reference: {ref_source}\n")
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
            reference, geoms, automorphisms, use_permutations=use_permutations
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
