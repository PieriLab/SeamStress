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
  # Align only
  seamstress -f ./data/spawns -o ./results

  # Align + dimensionality reduction analysis
  seamstress -f ./data/spawns -o ./results --analyze

  # With custom centroids and analysis
  seamstress -f ./molecules -o ./output -c ./centroids --analyze

  # Fast mode for large molecules (no permutation search)
  seamstress -f ./data -o ./output --no-permutations --analyze

  # Final refinement with heavy atom weighting (reduces H alignment artifacts)
  seamstress -f ./data -o ./output --heavy-atom-factor 10.0
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
        "-c",
        "--centroids",
        type=str,
        default=None,
        help="Folder containing reference/centroid structures (XYZ files)",
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

    parser.add_argument(
        "--no-permutations",
        action="store_true",
        help="Disable permutation search (use identity permutation only, faster for large molecules)",
    )

    parser.add_argument(
        "--heavy-atom-factor",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Weight multiplier for heavy (non-H) atoms in alignment (default: 1.0). "
             "Applied in two contexts: "
             "(1) Inter-family: aligning family references to each other "
             "(2) Intra-family: final refinement AFTER best permutation is found. "
             "Use larger values (e.g., 10.0, 100.0) to prioritize heavy atom alignment. "
             "Helpful when hydrogens cause alignment orientation issues.",
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run dimensionality reduction analysis after alignment (generates interactive HTML dashboard)",
    )

    parser.add_argument(
        "--analysis-output",
        type=str,
        default=None,
        help="Output directory for analysis results (default: <output>/analysis)",
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

    # Run alignment
    analyze_connectivity = not args.no_connectivity
    compute_automorphisms = not args.no_automorphisms and analyze_connectivity
    use_permutations = not args.no_permutations
    centroids_folder = Path(args.centroids) if args.centroids else None
    heavy_atom_factor = args.heavy_atom_factor

    try:
        process_geometries(
            input_folder,
            analyze_connectivity=analyze_connectivity,
            compute_automorphisms=compute_automorphisms,
            output_dir=output_folder,
            centroids_dir=centroids_folder,
            use_permutations=use_permutations,
            heavy_atom_factor=heavy_atom_factor,
        )

        # Run dimensionality reduction analysis if requested
        if args.analyze:
            from seamstress.analysis import run_analysis

            analysis_output = (
                Path(args.analysis_output)
                if args.analysis_output
                else output_folder / "analysis"
            )

            print("\n" + "=" * 80)
            print("RUNNING DIMENSIONALITY REDUCTION ANALYSIS")
            print("=" * 80)

            run_analysis(
                aligned_dir=output_folder,
                output_dir=analysis_output,
            )

    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
