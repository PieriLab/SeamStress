"""Check atom swapping/reordering is correct."""

import sys
from pathlib import Path

from elisa_spawns.geometry import read_xyz_file


def compare_atom_order(input_dir: Path, output_dir: Path, filename: str):
    """
    Compare atom order between input and output file.

    Args:
        input_dir: Directory with original XYZ files
        output_dir: Directory with aligned/swapped XYZ files
        filename: Name of file to compare
    """
    input_file = input_dir / filename
    output_file = output_dir / filename

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    if not output_file.exists():
        print(f"Error: Output file not found: {output_file}")
        return

    input_geom = read_xyz_file(input_file)
    output_geom = read_xyz_file(output_file)

    print(f"\n{'=' * 80}")
    print(f"Comparing: {filename}")
    print(f"{'=' * 80}")
    print(f"\nOutput metadata: {output_geom.metadata}")

    # Check if it's a reference
    if "Reference" in output_geom.metadata:
        print("\n✓ This is a REFERENCE molecule (not swapped)")
        print("  Atom order should be IDENTICAL to input")

    print(f"\n{'Index':<8} {'Input Atom':<15} {'Output Atom':<15} {'Swapped?':<10}")
    print("-" * 80)

    swapped_count = 0
    for i, (in_atom, out_atom) in enumerate(zip(input_geom.atoms, output_geom.atoms)):
        in_str = f"{in_atom} ({input_geom.coordinates[i][0]:.2f}, {input_geom.coordinates[i][1]:.2f}, {input_geom.coordinates[i][2]:.2f})"
        out_str = f"{out_atom} ({output_geom.coordinates[i][0]:.2f}, {output_geom.coordinates[i][1]:.2f}, {output_geom.coordinates[i][2]:.2f})"

        # Check if atom type changed (definite swap)
        if in_atom != out_atom:
            print(f"{i:<8} {in_str:<15} {out_str:<15} {'YES ⚠️':<10}")
            swapped_count += 1
        else:
            # Same atom type - check if position changed significantly
            coord_diff = abs(
                input_geom.coordinates[i] - output_geom.coordinates[i]
            ).max()
            if coord_diff > 0.1:  # More than 0.1 Å movement
                print(f"{i:<8} {in_str:<15} {out_str:<15} {'MOVED':<10}")
            else:
                print(f"{i:<8} {in_str:<15} {out_str:<15} {'-':<10}")

    print(f"\n{'=' * 80}")
    if swapped_count > 0:
        print(
            f"⚠️  {swapped_count} atoms changed type - this indicates INCORRECT swapping!"
        )
        print("   Atoms should only be REORDERED, not change element type!")
    else:
        print("✓ All atoms kept their element type (good!)")
        print("  Atoms were reordered/realigned but C stays C, H stays H, etc.")


def find_reference_for_family(output_dir: Path, family_num: int) -> str:
    """Find the reference molecule filename for a given family."""
    for xyz_file in output_dir.glob("*.xyz"):
        geom = read_xyz_file(xyz_file)
        if f"Family_{family_num}" in geom.metadata and "Reference" in geom.metadata:
            return xyz_file.name
    return None


def compare_to_reference(output_dir: Path, filename: str):
    """
    Compare a swapped molecule to its family reference.
    Shows which atoms map to which positions in reference.
    """
    output_file = output_dir / filename

    if not output_file.exists():
        print(f"Error: File not found: {output_file}")
        return

    target_geom = read_xyz_file(output_file)

    # Extract family number
    if "Family_" not in target_geom.metadata:
        print("Error: No family information found in file")
        return

    family_num = int(target_geom.metadata.split("Family_")[1].split()[0])

    # Find reference
    ref_filename = find_reference_for_family(output_dir, family_num)
    if not ref_filename:
        print(f"Error: Could not find reference for Family {family_num}")
        return

    ref_geom = read_xyz_file(output_dir / ref_filename)

    print(f"\n{'=' * 80}")
    print("Comparing to Reference")
    print(f"{'=' * 80}")
    print(f"Reference: {ref_filename}")
    print(f"Target:    {filename}")
    print(f"Family:    {family_num}")
    print(f"\nTarget metadata: {target_geom.metadata}")

    print(f"\n{'Target Idx':<12} {'Target Atom':<12} {'Ref Atom':<12} {'Match?':<10}")
    print("-" * 80)

    match_count = 0
    for i, (target_atom, ref_atom) in enumerate(zip(target_geom.atoms, ref_geom.atoms)):
        match = "✓" if target_atom == ref_atom else "✗"
        if target_atom == ref_atom:
            match_count += 1
        print(f"{i:<12} {target_atom:<12} {ref_atom:<12} {match:<10}")

    print(f"\n{'=' * 80}")
    print(f"Atom order match: {match_count}/{len(target_geom.atoms)} atoms")
    if match_count == len(target_geom.atoms):
        print("✓ PERFECT: All atoms in same order as reference!")
    else:
        print(
            f"⚠️  Only {match_count}/{len(target_geom.atoms)} atoms match reference order"
        )


def list_families(output_dir: Path):
    """List all families and their molecules."""
    families = {}

    for xyz_file in output_dir.glob("*.xyz"):
        geom = read_xyz_file(xyz_file)
        if "Family_" in geom.metadata:
            family_num = int(geom.metadata.split("Family_")[1].split()[0])
            is_ref = "Reference" in geom.metadata

            if family_num not in families:
                families[family_num] = {"ref": None, "molecules": []}

            if is_ref:
                families[family_num]["ref"] = xyz_file.name
            else:
                families[family_num]["molecules"].append(xyz_file.name)

    print("\nFamilies found:")
    print("=" * 80)
    for family_num in sorted(families.keys()):
        info = families[family_num]
        print(f"\nFamily {family_num}:")
        print(f"  Reference: {info['ref']}")
        print(f"  Molecules: {len(info['molecules'])}")
        for mol in info["molecules"][:5]:
            print(f"    - {mol}")
        if len(info["molecules"]) > 5:
            print(f"    ... and {len(info['molecules']) - 5} more")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print(
            "  python check_swapping.py <output_dir>                    # List all families"
        )
        print(
            "  python check_swapping.py <output_dir> <filename>         # Compare to reference"
        )
        print(
            "  python check_swapping.py <input_dir> <output_dir> <file> # Compare input vs output"
        )
        print("\nExamples:")
        print("  python check_swapping.py ./output")
        print("  python check_swapping.py ./output molecule2.xyz")
        print("  python check_swapping.py ./input ./output molecule2.xyz")
        sys.exit(1)

    if len(sys.argv) == 2:
        # List families
        output_dir = Path(sys.argv[1])
        list_families(output_dir)

    elif len(sys.argv) == 3:
        # Compare to reference
        output_dir = Path(sys.argv[1])
        filename = sys.argv[2]
        compare_to_reference(output_dir, filename)

    elif len(sys.argv) == 4:
        # Compare input vs output
        input_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
        filename = sys.argv[3]
        compare_atom_order(input_dir, output_dir, filename)
