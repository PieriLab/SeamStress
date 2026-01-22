"""Main entry point for reading and displaying molecular geometries."""

from pathlib import Path

from seamstress.processor import process_geometries


def main():
    """Read all XYZ files from data/spawns and analyze them."""
    data_dir = Path(__file__).parent / "data" / "spawns"
    output_dir = Path(__file__).parent / "aligned_output"
    centroids_dir = Path(__file__).parent / "centroids"
    process_geometries(
        data_dir,
        analyze_connectivity=True,
        compute_automorphisms=True,
        output_dir=output_dir,
        centroids_dir=centroids_dir,
        use_permutations=True,
        heavy_atom_factor=1.0,  # Use default weighting
        master_reference=None,  # Use default (largest family)
    )


if __name__ == "__main__":
    main()
