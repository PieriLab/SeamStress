"""
Align spawns and write them to data/<dataset>/aligned_spawns/.

Two sub-folders are produced:

  single_ref/
      Every spawn is aligned to one master reference (see DATASETS below).
      The atom permutation is read from reports/<dataset>.csv (bf_perm column).

  multi_ref/
      Every MECI centroid in data/<dataset>/mecis/ is first aligned to the
      master reference via brute-force permutation search + Kabsch.
      Each spawn is then aligned to whichever centroid yields the lowest RMSD.

Reflection is allowed in all Kabsch calls.

Usage:
    uv run python align_dataset.py
    uv run python align_dataset.py -d benzene_s0
    uv run python align_dataset.py --spawns-subdir new_spawns
"""

import argparse
import ast
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from seamstress.alignment import (
    _search_bruteforce_elementwise,
    kabsch_align_rmsd,
)
from seamstress.geometry import read_all_geometries, read_xyz_file


# ---------------------------------------------------------------------------
# Dataset configuration  (mirrors benchmark_full.py)
# ---------------------------------------------------------------------------

DATASETS = {
    "benzene_s0": {
        "spawns":    "data/benzene_s0/spawns",
        "reference": "data/benzene_s0/mecis/Type_2.xyz",
        "mecis":     "data/benzene_s0/mecis",
    },
    "benzene_s1": {
        "spawns":    "data/benzene_s1/spawns",
        "reference": "data/benzene_s1/mecis/Type_3.xyz",
        "mecis":     "data/benzene_s1/mecis",
    },
    "ethylene": {
        "spawns":    "data/ethylene/spawns",
        "reference": "data/ethylene/mecis/Twist.xyz",
        "mecis":     "data/ethylene/mecis",
    },
    "butadiene_s0": {
        "spawns":    "data/butadiene_s0/spawns",
        "reference": "data/butadiene_s0/mecis/type12.xyz",
        "mecis":     "data/butadiene_s0/mecis",
    },
    "butadiene_s1": {
        "spawns":    "data/butadiene_s1/spawns",
        "reference": "data/butadiene_s1/mecis/type7.xyz",
        "mecis":     "data/butadiene_s1/mecis",
    },
}

# Medoid references determined by evaluate_reference.py
MEDOID_DATASETS = {
    "benzene_s0": {
        "spawns":    "data/benzene_s0/spawns",
        "reference": "data/benzene_s0/mecis/Type_4.xyz",
        "mecis":     "data/benzene_s0/mecis",
    },
    "benzene_s1": {
        "spawns":    "data/benzene_s1/spawns",
        "reference": "data/benzene_s1/mecis/Type_1.xyz",
        "mecis":     "data/benzene_s1/mecis",
    },
    "ethylene": {
        "spawns":    "data/ethylene/spawns",
        "reference": "data/ethylene/mecis/Ethylidene_Bent.xyz",
        "mecis":     "data/ethylene/mecis",
    },
    "butadiene_s0": {
        "spawns":    "data/butadiene_s0/spawns",
        "reference": "data/butadiene_s0/mecis/type12.xyz",
        "mecis":     "data/butadiene_s0/mecis",
    },
    "butadiene_s1": {
        "spawns":    "data/butadiene_s1/spawns",
        "reference": "data/butadiene_s1/mecis/type5.xyz",
        "mecis":     "data/butadiene_s1/mecis",
    },
}

ALLOW_REFLECTION = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_xyz(path: Path, atoms: list[str], coords: np.ndarray, comment: str = "") -> None:
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for atom, coord in zip(atoms, coords):
            f.write(f" {atom}  {coord[0]:18.9f}  {coord[1]:18.9f}  {coord[2]:18.9f}\n")


def _write_trajectory(xyz_dir: Path, traj_path: Path) -> int:
    """Concatenate all XYZ files in xyz_dir into a single multi-frame trajectory file."""
    xyz_files = sorted(xyz_dir.glob("*.xyz"))
    n = 0
    with open(traj_path, "w") as out:
        for xyz_file in xyz_files:
            out.write(xyz_file.read_text())
            n += 1
    return n


def _load_permutations(csv_path: Path) -> dict[str, tuple[int, ...]]:
    """Return {filename: bf_perm} from a benchmark CSV report."""
    perms = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            perms[row["filename"]] = ast.literal_eval(row["bf_perm"])
    return perms


def _align_meci_to_reference(meci, reference) -> tuple[np.ndarray, tuple[int, ...]]:
    """
    Brute-force align a MECI centroid to the master reference.
    Returns (aligned_coords, best_perm).
    """
    best_perm, _, _, _, aligned_coords, _, _ = _search_bruteforce_elementwise(
        reference.coordinates,
        meci.coordinates,
        reference.atoms,
        meci.atoms,
        ALLOW_REFLECTION,
    )
    return aligned_coords, best_perm


# ---------------------------------------------------------------------------
# Core alignment routines
# ---------------------------------------------------------------------------

def align_single_ref(spawns, reference, permutations, out_dir: Path) -> None:
    """Align each spawn to the master reference using the pre-computed bf_perm."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for tgt in tqdm(spawns, desc="  single_ref", unit="geom", leave=False):
        perm = permutations.get(tgt.filename)
        if perm is None:
            print(f"  WARNING: no permutation for {tgt.filename}, skipping.")
            continue

        reordered_coords = tgt.coordinates[list(perm)]
        reordered_atoms  = [tgt.atoms[i] for i in perm]

        aligned_coords, _ = kabsch_align_rmsd(
            reference.coordinates,
            reordered_coords,
            reference.atoms,
            reordered_atoms,
            use_all_atoms=True,
            allow_reflection=ALLOW_REFLECTION,
        )

        _write_xyz(out_dir / tgt.filename, reference.atoms, aligned_coords, tgt.metadata)


def align_multi_ref(spawns, reference, mecis_dir: Path, permutations, out_dir: Path) -> None:
    """
    Align each spawn to the closest MECI centroid.

    Each MECI is first brute-force aligned to the master reference so all
    centroids live in a common reference frame.  For each spawn the bf_perm
    is applied first, then the spawn is Kabsch-aligned to each aligned MECI;
    the centroid giving the lowest RMSD is chosen.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- align every MECI to the master reference ---
    meci_files = sorted(mecis_dir.glob("*.xyz"))
    if not meci_files:
        print(f"  WARNING: no XYZ files found in {mecis_dir}")
        return

    aligned_mecis = []  # list of (filename, aligned_coords)
    print(f"  Aligning {len(meci_files)} MECI centroids to master reference...")
    for mf in tqdm(meci_files, desc="  MECIs", unit="meci", leave=False):
        meci = read_xyz_file(mf)
        # Skip if this is the master reference itself (already aligned)
        if meci.filename == reference.filename:
            aligned_mecis.append((meci.filename, reference.coordinates))
        else:
            aligned_coords, _ = _align_meci_to_reference(meci, reference)
            aligned_mecis.append((meci.filename, aligned_coords))

    # --- align each spawn to its closest MECI ---
    for tgt in tqdm(spawns, desc="  multi_ref", unit="geom", leave=False):
        perm = permutations.get(tgt.filename)
        if perm is None:
            print(f"  WARNING: no permutation for {tgt.filename}, skipping.")
            continue

        reordered_coords = tgt.coordinates[list(perm)]
        reordered_atoms  = [tgt.atoms[i] for i in perm]

        best_rmsd     = float("inf")
        best_aligned  = None
        best_meci_name = None

        for meci_name, meci_coords in aligned_mecis:
            aligned_coords, rmsd = kabsch_align_rmsd(
                meci_coords,
                reordered_coords,
                reference.atoms,   # atom types are the same for all
                reordered_atoms,
                use_all_atoms=True,
                allow_reflection=ALLOW_REFLECTION,
            )
            if rmsd < best_rmsd:
                best_rmsd     = rmsd
                best_aligned  = aligned_coords
                best_meci_name = meci_name

        _write_xyz(out_dir / tgt.filename, reference.atoms, best_aligned,
                   f"{tgt.metadata} | closest_meci={best_meci_name} rmsd={best_rmsd:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-d", "--dataset",
        choices=list(DATASETS.keys()),
        default=None,
        help="Process only this dataset (default: all)",
    )
    parser.add_argument(
        "-r", "--reference",
        default=None,
        metavar="XYZ",
        help="Override the reference XYZ file (requires -d)",
    )
    parser.add_argument(
        "--spawns-subdir",
        default=None,
        metavar="SUBDIR",
        help="Override the spawns sub-directory name (e.g. 'new_spawns')",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory containing per-dataset CSV reports (default: reports/)",
    )
    parser.add_argument(
        "--use-medoid", action="store_true",
        help="Use medoid references instead of the default ones",
    )
    parser.add_argument(
        "--aligned-subdir",
        default="aligned_spawns",
        metavar="SUBDIR",
        help="Output sub-directory name under data/<dataset>/ (default: aligned_spawns)",
    )
    args = parser.parse_args()

    if args.reference and not args.dataset:
        parser.error("--reference requires --dataset/-d")

    active_datasets = MEDOID_DATASETS if args.use_medoid else DATASETS
    datasets_to_run = (
        {args.dataset: active_datasets[args.dataset]} if args.dataset else active_datasets
    )

    for dataset_name, cfg in datasets_to_run.items():
        spawns_dir = Path(cfg["spawns"])
        if args.spawns_subdir:
            spawns_dir = spawns_dir.parent / args.spawns_subdir
        ref_path   = Path(args.reference) if args.reference else Path(cfg["reference"])
        mecis_dir  = Path(cfg["mecis"])
        csv_path   = Path(args.reports_dir) / f"{dataset_name}.csv"
        out_base   = Path("data") / dataset_name / args.aligned_subdir

        print(f"\n{'='*60}")
        print(f"Dataset : {dataset_name}")
        print(f"Spawns  : {spawns_dir}")
        print(f"Ref     : {ref_path}")
        print(f"MECIs   : {mecis_dir}")
        print(f"CSV     : {csv_path}")

        # --- checks ---
        missing = [p for p in (spawns_dir, ref_path, mecis_dir, csv_path) if not p.exists()]
        if missing:
            for m in missing:
                print(f"  MISSING: {m}")
            print("  Skipping dataset.")
            continue

        reference   = read_xyz_file(ref_path)
        spawns      = read_all_geometries(spawns_dir)
        permutations = _load_permutations(csv_path)

        print(f"Spawns loaded : {len(spawns)}")
        print(f"Perms loaded  : {len(permutations)}")

        # --- single_ref ---
        print("Running single_ref alignment...")
        single_ref_dir = out_base / "single_ref"
        align_single_ref(spawns, reference, permutations, out_dir=single_ref_dir)
        print(f"  Written to {single_ref_dir}")

        traj_path = out_base / "single_ref_trajectory.xyz"
        n_frames = _write_trajectory(single_ref_dir, traj_path)
        print(f"  Trajectory  → {traj_path}  ({n_frames} frames)")

        # --- multi_ref ---
        print("Running multi_ref alignment...")
        multi_ref_dir = out_base / "multi_ref"
        align_multi_ref(spawns, reference, mecis_dir, permutations, out_dir=multi_ref_dir)
        print(f"  Written to {multi_ref_dir}")

        traj_path = out_base / "multi_ref_trajectory.xyz"
        n_frames = _write_trajectory(multi_ref_dir, traj_path)
        print(f"  Trajectory  → {traj_path}  ({n_frames} frames)")

    print("\nDone.")


if __name__ == "__main__":
    main()
