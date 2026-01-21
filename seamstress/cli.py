"""Command-line interface for SeamStress."""

import argparse
import sys
from pathlib import Path

from seamstress.processor import process_geometries


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="seamstress",
        description="Analyze molecular geometries from XYZ files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  seamstress -f ./data/spawns
  seamstress -f ./molecules -o ./results
  seamstress -f ./data --no-automorphisms
        """,
    )

    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="Folder containing XYZ files to analyze",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output folder for aligned/swapped geometry files",
    )

    parser.add_argument(
        "--no-connectivity",
        action="store_true",
        help="Disable connectivity analysis (just display geometries)",
    )

    parser.add_argument(
        "--no-automorphisms",
        action="store_true",
        help="Disable automorphism computation (only show connectivity groups)",
    )

    args = parser.parse_args()

    # Validate input folder
    input_folder = Path(args.folder)
    if not input_folder.exists():
        print(f"Error: Input folder not found: {input_folder}", file=sys.stderr)
        sys.exit(1)

    if not input_folder.is_dir():
        print(f"Error: Input path is not a directory: {input_folder}", file=sys.stderr)
        sys.exit(1)

    # Create output folder
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Run analysis
    analyze_connectivity = not args.no_connectivity
    compute_automorphisms = not args.no_automorphisms and analyze_connectivity

    try:
        process_geometries(
            input_folder,
            analyze_connectivity=analyze_connectivity,
            compute_automorphisms=compute_automorphisms,
            output_dir=output_folder,
        )
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
