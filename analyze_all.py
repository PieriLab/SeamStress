#!/usr/bin/env python3
"""
Dimensionality reduction analysis - generates single interactive HTML dashboard.

This script is now a thin wrapper around seamstress.analysis for backward compatibility.
For new code, import and use seamstress.analysis.run_analysis() directly, or use the CLI:

    seamstress -f ./data -o ./output --analyze
"""

from pathlib import Path
from seamstress.analysis import run_analysis


def main():
    """Run analysis with default paths."""
    aligned_dir = Path("aligned_output")
    output_dir = Path("analysis_output")

    run_analysis(aligned_dir, output_dir)


if __name__ == "__main__":
    main()
