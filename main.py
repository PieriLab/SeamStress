"""Main entry point for reading and displaying molecular geometries."""

from pathlib import Path

from elisa_spawns.processor import process_geometries


def main():
    """Read all XYZ files from data/spawns and analyze them."""
    data_dir = Path(__file__).parent / "data" / "spawns"
    output_dir = Path(__file__).parent / "aligned_output"
    process_geometries(data_dir, analyze_connectivity=True, compute_automorphisms=True, output_dir=output_dir)


if __name__ == "__main__":
    main()
