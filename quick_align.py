#!/usr/bin/env python3
"""Quick and dirty script to align geometries and save to new folder."""

import numpy as np
from pathlib import Path
from rdkit import Chem
from collections import defaultdict
import click


# === READ XYZ ===
def read_xyz(filepath):
    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]
    n_atoms = int(lines[0])
    metadata = lines[1]
    atoms, coords = [], []
    for line in lines[2:2+n_atoms]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return atoms, np.array(coords), metadata


# === WRITE XYZ ===
def write_xyz(filepath, atoms, coords, metadata):
    with open(filepath, 'w') as f:
        f.write(f"{len(atoms)}\n{metadata}\n")
        for atom, coord in zip(atoms, coords):
            f.write(f"{atom}  {coord[0]:12.8f}  {coord[1]:12.8f}  {coord[2]:12.8f}\n")


# === CONNECTIVITY HASH ===
def get_connectivity_hash(atoms, coords):
    mol = Chem.RWMol()
    atom_map = {}
    for i, atom_symbol in enumerate(atoms):
        atom_map[i] = mol.AddAtom(Chem.Atom(atom_symbol))
    for i in range(len(atoms)):
        mol.GetAtomWithIdx(i).SetNoImplicit(True)
    conf = Chem.Conformer(len(atoms))
    for i, coord in enumerate(coords):
        conf.SetAtomPosition(i, coord.tolist())
    mol.AddConformer(conf)
    Chem.rdDetermineBonds.DetermineBonds(mol, charge=0)
    smiles = Chem.MolToSmiles(mol, canonical=True)
    return smiles, mol


# === AUTOMORPHISMS ===
def get_automorphisms(mol):
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=False)
    return list(matches) if matches else [(tuple(range(mol.GetNumAtoms())),)]


# === KABSCH ALIGNMENT ===
def kabsch_align(coords1, coords2, atoms1, atoms2, heavy_only=True):
    """
    Kabsch alignment with optional heavy-atom-only mode.

    Args:
        coords1: Reference coordinates
        coords2: Target coordinates to align
        atoms1: Reference atom symbols
        atoms2: Target atom symbols
        heavy_only: If True, compute alignment using only heavy atoms (default: True)

    Returns:
        rmsd: Root mean square deviation
        aligned_all: Aligned coordinates for all atoms
    """
    if heavy_only:
        heavy_mask = np.array([a != 'H' for a in atoms1])
        c1_align, c2_align = coords1[heavy_mask], coords2[heavy_mask]
    else:
        heavy_mask = None
        c1_align, c2_align = coords1, coords2

    centroid1, centroid2 = c1_align.mean(0), c2_align.mean(0)
    H = (c2_align - centroid2).T @ (c1_align - centroid1)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    aligned_all = (coords2 - centroid2) @ R + centroid1

    if heavy_only:
        diff = c1_align - aligned_all[heavy_mask]
    else:
        diff = c1_align - aligned_all

    rmsd = np.sqrt((diff**2).sum(1).mean())
    return rmsd, aligned_all


# === BEST PERMUTATION ===
def find_best_alignment(ref_coords, tgt_coords, automorphisms, ref_atoms, tgt_atoms, heavy_only=True):
    best_rmsd, best_perm, best_aligned = float('inf'), None, None
    for perm in automorphisms:
        perm_coords = tgt_coords[list(perm)]
        perm_atoms = [tgt_atoms[i] for i in perm]
        rmsd, aligned = kabsch_align(ref_coords, perm_coords, ref_atoms, perm_atoms, heavy_only=heavy_only)
        if rmsd < best_rmsd:
            best_rmsd, best_perm, best_aligned = rmsd, perm, aligned
    return best_rmsd, best_perm, best_aligned


@click.command()
@click.option('-i', '--input', 'input_folder', required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), help='Input folder containing XYZ files')
@click.option('-o', '--output', 'output_folder', required=True, type=click.Path(path_type=Path), help='Output folder for aligned geometries')
@click.option('--heavy-only/--all-atoms', default=True, help='Align using only heavy atoms (default) or all atoms including hydrogens')
def main(input_folder, output_folder, heavy_only):
    """
    Quick and dirty script to align molecular geometries.

    Reads XYZ files, groups by connectivity, finds best atom permutations
    using automorphisms, performs Kabsch alignment, and saves aligned
    geometries to output folder.
    """
    # Create output folder
    output_folder.mkdir(exist_ok=True, parents=True)

    # Find all XYZ files
    xyz_files = sorted(input_folder.glob("*.xyz"))

    if not xyz_files:
        click.echo(f"No XYZ files found in {input_folder}", err=True)
        return

    click.echo(f"Found {len(xyz_files)} XYZ files")
    click.echo(f"Output folder: {output_folder}")
    click.echo(f"Alignment mode: {'Heavy atoms only' if heavy_only else 'All atoms'}")
    click.echo()

    families = defaultdict(list)

    # Group by connectivity
    click.echo("Grouping by connectivity...")
    for f in xyz_files:
        atoms, coords, meta = read_xyz(f)
        conn_hash, mol = get_connectivity_hash(atoms, coords)
        families[conn_hash].append((f, atoms, coords, meta, mol))

    click.echo(f"Found {len(families)} unique connectivity patterns")
    click.echo()

    # Process each family
    total_processed = 0
    for family_idx, (conn_hash, members) in enumerate(families.items(), 1):
        click.echo(f"Family {family_idx}/{len(families)}: {len(members)} molecules ({conn_hash})")

        ref_file, ref_atoms, ref_coords, ref_meta, ref_mol = members[0]
        automorphisms = get_automorphisms(ref_mol)

        click.echo(f"  Reference: {ref_file.name}")
        click.echo(f"  Automorphisms: {len(automorphisms)}")

        # Write reference as-is
        write_xyz(output_folder / ref_file.name, ref_atoms, ref_coords,
                  f"Family {family_idx} | Reference | {ref_meta}")
        total_processed += 1

        # Align and write others
        for tgt_file, tgt_atoms, tgt_coords, tgt_meta, _ in members[1:]:
            rmsd, perm, aligned = find_best_alignment(
                ref_coords, tgt_coords, automorphisms, ref_atoms, tgt_atoms, heavy_only=heavy_only
            )
            reordered_atoms = [tgt_atoms[i] for i in perm]
            write_xyz(output_folder / tgt_file.name, reordered_atoms, aligned,
                      f"Family {family_idx} | RMSD: {rmsd:.6f} | {tgt_meta}")
            click.echo(f"    {tgt_file.name}: RMSD = {rmsd:.6f}")
            total_processed += 1

        click.echo()

    click.echo(f"Done! Processed {total_processed} files into {output_folder}/")


if __name__ == '__main__':
    main()
