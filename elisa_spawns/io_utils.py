"""Module for input/output utilities."""

from pathlib import Path

import numpy as np

from elisa_spawns.geometry import Geometry


def write_xyz_file(
    filepath: Path | str,
    atoms: list[str],
    coordinates: np.ndarray,
    comment: str = "",
) -> None:
    """
    Write an XYZ file.

    Args:
        filepath: Path to write the XYZ file
        atoms: List of atom symbols
        coordinates: Coordinates array (N, 3)
        comment: Comment line (metadata)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for atom, coord in zip(atoms, coordinates):
            f.write(f"{atom:2s} {coord[0]:15.8f} {coord[1]:15.8f} {coord[2]:15.8f}\n")


def save_aligned_geometries(
    alignment_results: list[dict],
    output_dir: Path | str,
    save_aligned: bool = True,
    save_reordered: bool = True,
) -> None:
    """
    Save aligned and/or reordered geometries to XYZ files.

    Args:
        alignment_results: Results from align_geometries_with_automorphisms
        output_dir: Base output directory
        save_aligned: If True, save Kabsch-aligned coordinates
        save_reordered: If True, save reordered (permuted) coordinates
    """
    output_dir = Path(output_dir)

    if save_aligned:
        aligned_dir = output_dir / "aligned"
        aligned_dir.mkdir(parents=True, exist_ok=True)

    if save_reordered:
        reordered_dir = output_dir / "reordered"
        reordered_dir.mkdir(parents=True, exist_ok=True)

    for result in alignment_results:
        geom = result["geometry"]
        filename = geom.filename

        # Save Kabsch-aligned coordinates
        if save_aligned:
            comment = f"{geom.metadata} | RMSD: {result['rmsd']:.4f} | Perm: {result['permutation']}"
            write_xyz_file(
                aligned_dir / filename,
                result["reordered_atoms"],
                result["aligned_coords"],
                comment=comment,
            )

        # Save reordered (permuted) coordinates
        if save_reordered:
            comment = f"{geom.metadata} | Permutation: {result['permutation']}"
            write_xyz_file(
                reordered_dir / filename,
                result["reordered_atoms"],
                result["reordered_coords"],
                comment=comment,
            )
