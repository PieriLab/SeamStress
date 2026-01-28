#!/usr/bin/env python3
"""
Dimensionality reduction analysis - generates single interactive HTML dashboard.

This script provides a standalone analysis tool that can be run independently
from the alignment step, with options to exclude specific families.

Usage:
    # Analyze all families
    python analyze_all.py

    # Exclude specific families
    python analyze_all.py --exclude family_3 family_7

    # Custom input/output directories
    python analyze_all.py -i ./aligned_output -o ./my_analysis

    # Or use the integrated CLI:
    seamstress -f ./data -o ./output --analyze
"""

import argparse
import sys
from pathlib import Path
from seamstress.analysis import run_analysis


def main():
    """Run analysis with optional family exclusions."""
    parser = argparse.ArgumentParser(
        prog="analyze_all",
        description="Run dimensionality reduction analysis on aligned geometries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all families (default paths)
  python analyze_all.py

  # Exclude specific families from analysis
  python analyze_all.py --exclude family_3 family_7

  # Custom input/output directories
  python analyze_all.py -i ./results -o ./analysis

  # List available families first
  ls aligned_output/
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="aligned_output",
        help="Directory containing aligned family folders (default: aligned_output)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="analysis_output",
        help="Output directory for analysis results (default: analysis_output)",
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Family names to exclude from analysis (e.g., family_1 family_3)",
    )

    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Only include these families (overrides --exclude if specified)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available families and exit (useful for deciding what to exclude)",
    )

    parser.add_argument(
        "--use-smiles",
        action="store_true",
        help="Use SMILES strings in plot legends instead of display names",
    )

    args = parser.parse_args()

    aligned_dir = Path(args.input)
    output_dir = Path(args.output)

    # Validate input directory
    if not aligned_dir.exists():
        print(f"Error: Input directory not found: {aligned_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover available families
    all_families = sorted([f.name for f in aligned_dir.glob("family_*")])
    if not all_families:
        print(f"Error: No family directories found in {aligned_dir}", file=sys.stderr)
        sys.exit(1)

    # List mode: show families and exit
    if args.list:
        print(f"\nAvailable families in {aligned_dir}:")
        print("=" * 70)
        for family_name in all_families:
            family_dir = aligned_dir / family_name
            n_files = len(list(family_dir.glob("*.xyz"))) - 1  # Subtract reference.xyz
            ref_file = family_dir / "reference.xyz"
            if ref_file.exists():
                with open(ref_file) as f:
                    lines = f.readlines()
                    header = lines[1].strip() if len(lines) > 1 else ""
                    # Extract SMILES
                    import re
                    smiles_match = re.search(r"Family_\d+\s+(\S+)", header)
                    smiles = smiles_match.group(1) if smiles_match else "Unknown"
            else:
                smiles = "Unknown"
            print(f"  {family_name}: {n_files} molecules, SMILES: {smiles}")
        print("\nTo exclude families, use:")
        print(f"  python analyze_all.py --exclude {all_families[0]} {all_families[1] if len(all_families) > 1 else ''}")
        sys.exit(0)

    print(f"Found {len(all_families)} families: {', '.join(all_families)}")

    # Determine which families to include
    if args.include:
        # Include mode: only analyze specified families
        families_to_analyze = [f for f in args.include if f in all_families]
        excluded = set(all_families) - set(families_to_analyze)
        if excluded:
            print(f"Including only: {', '.join(families_to_analyze)}")
            print(f"Excluding: {', '.join(sorted(excluded))}")
    elif args.exclude:
        # Exclude mode: analyze all except specified
        families_to_analyze = [f for f in all_families if f not in args.exclude]
        excluded = set(args.exclude) & set(all_families)
        if excluded:
            print(f"Excluding: {', '.join(sorted(excluded))}")
            print(f"Analyzing: {', '.join(families_to_analyze)}")
    else:
        # Default: analyze all families
        families_to_analyze = all_families

    if not families_to_analyze:
        print("Error: No families to analyze after filtering", file=sys.stderr)
        sys.exit(1)

    # Run analysis with filtered families
    run_analysis(
        aligned_dir=aligned_dir,
        output_dir=output_dir,
        families_to_include=families_to_analyze,
        use_smiles_in_legend=args.use_smiles,
    )


if __name__ == "__main__":
    main()
