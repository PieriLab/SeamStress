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

  # Fragment permutations for benzene-like molecules (massive speedup)
  seamstress -f ./benzene_spawns -o ./output -c ./centroids --fragment-permutations

  # Heavy atom weighting for inter-family centroid alignment only
  seamstress -f ./data -o ./output -c ./centroids --inter-family-heavy-atom-factor 10.0

  # Heavy atom weighting for both inter-family and intra-family alignment
  seamstress -f ./data -o ./output --inter-family-heavy-atom-factor 10.0 --intra-family-heavy-atom-factor 5.0
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
        "--master-reference",
        type=str,
        default=None,
        metavar="FILENAME",
        help="Filename of centroid to use as master reference for inter-family alignment (e.g., 'ethylene.xyz'). "
             "If not specified, the largest family (Family 1) is used as master.",
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
        "--inter-family-heavy-atom-factor",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Weight multiplier for heavy atoms when aligning family references to each other (default: 1.0). "
             "Use larger values (e.g., 10.0, 100.0) to prioritize heavy atoms in inter-family alignment. "
             "Helpful when hydrogens cause centroid alignment orientation issues.",
    )

    parser.add_argument(
        "--intra-family-heavy-atom-factor",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Weight multiplier for heavy atoms when aligning molecules to their family reference (default: 1.0). "
             "Applied AFTER best permutation is found. "
             "Use larger values (e.g., 10.0, 100.0) to prioritize heavy atoms in final molecule alignment.",
    )

    parser.add_argument(
        "--prealign-centroids-to",
        type=str,
        default=None,
        metavar="FILENAME",
        help="Pre-align all centroids to this centroid before main workflow (e.g., 'benzene.xyz'). "
             "This is an optional first step that aligns all centroid structures to a user-specified reference. "
             "The aligned centroids are saved to output/prealigned_centroids/ and used for subsequent alignment.",
    )

    parser.add_argument(
        "--fragment-permutations",
        action="store_true",
        help="Use fragment-based permutation search (treats heavy atoms + bonded H as rigid units). "
             "Only applicable when all heavy atoms have exactly 1 hydrogen (e.g., benzene). "
             "Provides ~720x speedup for benzene-like molecules. Automatically falls back to standard mode if not applicable.",
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
    inter_family_heavy_atom_factor = args.inter_family_heavy_atom_factor
    intra_family_heavy_atom_factor = args.intra_family_heavy_atom_factor
    master_reference = args.master_reference
    prealign_centroids_to = args.prealign_centroids_to
    use_fragment_permutations = args.fragment_permutations

    try:
        process_geometries(
            input_folder,
            analyze_connectivity=analyze_connectivity,
            compute_automorphisms=compute_automorphisms,
            output_dir=output_folder,
            centroids_dir=centroids_folder,
            use_permutations=use_permutations,
            inter_family_heavy_atom_factor=inter_family_heavy_atom_factor,
            intra_family_heavy_atom_factor=intra_family_heavy_atom_factor,
            master_reference=master_reference,
            prealign_centroids_to=prealign_centroids_to,
            use_fragment_permutations=use_fragment_permutations,
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
