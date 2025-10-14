"""Quick visualization script to check alignment quality."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from elisa_spawns.geometry import read_xyz_file


def visualize_family(output_dir: Path, family_num: int):
    """
    Visualize all molecules from a specific family overlaid.

    Args:
        output_dir: Directory containing output XYZ files
        family_num: Family number to visualize
    """
    # Read all files from this family
    geometries = []
    for xyz_file in sorted(output_dir.glob("*.xyz")):
        geom = read_xyz_file(xyz_file)
        if geom.metadata.startswith(f"Family {family_num}"):
            geometries.append(geom)

    if not geometries:
        print(f"No molecules found for Family {family_num}")
        return

    print(f"Found {len(geometries)} molecules in Family {family_num}")

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Color map for different molecules
    colors = plt.cm.tab10(range(len(geometries)))

    # Plot each geometry
    for i, geom in enumerate(geometries):
        coords = geom.coordinates
        label = f"{geom.filename}"
        if "Reference" in geom.metadata:
            # Plot reference larger and black
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c="black",
                s=100,
                marker="o",
                label=label,
                alpha=0.8,
            )
        else:
            # Plot aligned molecules
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=[colors[i]],
                s=50,
                marker="o",
                label=label,
                alpha=0.6,
            )

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(f"Family {family_num} - Aligned Molecules Overlay")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_rmsd_distribution(output_dir: Path):
    """Plot RMSD distribution across all families."""
    rmsds = []
    families = []

    for xyz_file in output_dir.glob("*.xyz"):
        geom = read_xyz_file(xyz_file)
        if "RMSD:" in geom.metadata:
            # Extract RMSD value
            rmsd_str = geom.metadata.split("RMSD:")[1].split("|")[0].strip()
            rmsd = float(rmsd_str)

            # Extract family number
            family = int(geom.metadata.split("Family")[1].split("|")[0].strip())

            rmsds.append(rmsd)
            families.append(family)

    if not rmsds:
        print("No RMSD values found (only reference molecules?)")
        return

    # Create histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Overall RMSD distribution
    ax1.hist(rmsds, bins=20, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("RMSD (Å)")
    ax1.set_ylabel("Count")
    ax1.set_title("RMSD Distribution (All Families)")
    ax1.axvline(
        sum(rmsds) / len(rmsds),
        color="red",
        linestyle="--",
        label=f"Mean: {sum(rmsds) / len(rmsds):.4f} Å",
    )
    ax1.legend()

    # RMSD by family
    unique_families = sorted(set(families))
    family_rmsds = [
        [r for r, f in zip(rmsds, families) if f == fam] for fam in unique_families
    ]

    positions = range(1, len(unique_families) + 1)
    ax2.boxplot(family_rmsds, positions=positions, labels=unique_families)
    ax2.set_xlabel("Family Number")
    ax2.set_ylabel("RMSD (Å)")
    ax2.set_title("RMSD Distribution by Family")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nAlignment Statistics:")
    print(f"  Total aligned molecules: {len(rmsds)}")
    print(f"  Mean RMSD: {sum(rmsds) / len(rmsds):.4f} Å")
    print(f"  Min RMSD: {min(rmsds):.4f} Å")
    print(f"  Max RMSD: {max(rmsds):.4f} Å")
    print("\nRMSD by family:")
    for fam, rmsd_list in zip(unique_families, family_rmsds):
        print(
            f"  Family {fam}: {sum(rmsd_list) / len(rmsd_list):.4f} Å (n={len(rmsd_list)})"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_alignment.py <output_dir> [family_num]")
        print("\nExamples:")
        print(
            "  python visualize_alignment.py ./output          # Show RMSD statistics"
        )
        print("  python visualize_alignment.py ./output 1        # Visualize Family 1")
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        # Visualize specific family
        family_num = int(sys.argv[2])
        visualize_family(output_dir, family_num)
    else:
        # Show RMSD statistics
        plot_rmsd_distribution(output_dir)
