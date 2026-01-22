"""Main entry point for reading and displaying molecular geometries."""

from pathlib import Path

from seamstress.processor import process_geometries


def main():
    """Read all XYZ files from input_data/spawns and analyze them."""
    input_base = Path(__file__).parent / "input_data"
    data_dir = input_base / "spawns"
    centroids_dir = input_base / "centroids"
    output_dir = Path(__file__).parent / "aligned_output"

    process_geometries(
        data_dir,
        analyze_connectivity=True,
        compute_automorphisms=True,
        output_dir=output_dir,
        centroids_dir=centroids_dir,
        use_permutations=True,
        inter_family_heavy_atom_factor=1.0,  # Use default weighting for centroid alignment
        intra_family_heavy_atom_factor=1.0,  # Use default weighting for molecule alignment
        master_reference=None,  # Use default (largest family)
    )


if __name__ == "__main__":
    main()
