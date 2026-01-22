"""Module for reading and handling molecular geometries from XYZ files."""

from pathlib import Path
from typing import NamedTuple

import numpy as np


class Geometry(NamedTuple):
    """Represents a molecular geometry."""

    atoms: list[str]
    coordinates: np.ndarray
    metadata: str
    filename: str

    def __str__(self) -> str:
        """String representation of the geometry."""
        lines = [f"File: {self.filename}"]
        lines.append(f"Metadata: {self.metadata}")
        lines.append(f"Number of atoms: {len(self.atoms)}")
        lines.append("\nAtom  X           Y           Z")
        lines.append("-" * 40)
        for atom, coord in zip(self.atoms, self.coordinates):
            lines.append(
                f"{atom:4s}  {coord[0]:10.6f}  {coord[1]:10.6f}  {coord[2]:10.6f}"
            )
        return "\n".join(lines)


def read_xyz_file(filepath: Path | str) -> Geometry:
    """
    Read a single XYZ file and return a Geometry object.

    XYZ format:
    - Line 1: Number of atoms
    - Line 2: Comment/metadata
    - Lines 3+: Element X Y Z

    Args:
        filepath: Path to the XYZ file

    Returns:
        Geometry object containing atoms, coordinates, and metadata
    """
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    metadata = lines[1].strip() if len(lines) > 1 else ""

    atoms = []
    coordinates = []

    for line in lines[2 : 2 + n_atoms]:
        line = line.strip()
        if not line:  # Skip empty lines within atom section
            continue
        parts = line.split()
        atoms.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return Geometry(
        atoms=atoms,
        coordinates=np.array(coordinates),
        metadata=metadata,
        filename=filepath.name,
    )


def read_all_geometries(data_dir: Path | str) -> list[Geometry]:
    """
    Read all XYZ files from a directory.

    Args:
        data_dir: Path to directory containing XYZ files

    Returns:
        List of Geometry objects sorted by filename
    """
    data_dir = Path(data_dir)
    xyz_files = sorted(data_dir.glob("*.xyz"))

    geometries = []
    for xyz_file in xyz_files:
        geometries.append(read_xyz_file(xyz_file))

    return geometries
