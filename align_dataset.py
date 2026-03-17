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
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        "spawns":        "data/benzene_s0/spawns",
        "reference":     "data/benzene_s0/mecis/Type_2.xyz",
        "mecis":         "data/benzene_s0/mecis",
        "mecis_aligned": "data/benzene_s0/mecis_aligned",
    },
    "benzene_s1": {
        "spawns":        "data/benzene_s1/spawns",
        "reference":     "data/benzene_s1/mecis/Type_3.xyz",
        "mecis":         "data/benzene_s1/mecis",
        "mecis_aligned": "data/benzene_s1/mecis_aligned",
    },
    "ethylene": {
        "spawns":        "data/ethylene/spawns",
        "reference":     "data/ethylene/mecis/Twist.xyz",
        "mecis":         "data/ethylene/mecis",
        "mecis_aligned": "data/ethylene/mecis_aligned",
    },
    "butadiene_s0": {
        "spawns":        "data/butadiene_s0/spawns",
        "reference":     "data/butadiene_s0/mecis/type12.xyz",
        "mecis":         "data/butadiene_s0/mecis",
        "mecis_aligned": "data/butadiene_s0/mecis_aligned",
    },
    "butadiene_s1": {
        "spawns":        "data/butadiene_s1/spawns",
        "reference":     "data/butadiene_s1/mecis/type7.xyz",
        "mecis":         "data/butadiene_s1/mecis",
        "mecis_aligned": "data/butadiene_s1/mecis_aligned",
    },
}

# Medoid references determined by evaluate_reference.py
MEDOID_DATASETS = {
    "benzene_s0": {
        "spawns":        "data/benzene_s0/spawns",
        "reference":     "data/benzene_s0/mecis/Type_4.xyz",
        "mecis":         "data/benzene_s0/mecis",
        "mecis_aligned": "data/benzene_s0/mecis_aligned",
    },
    "benzene_s1": {
        "spawns":        "data/benzene_s1/spawns",
        "reference":     "data/benzene_s1/mecis/Type_1.xyz",
        "mecis":         "data/benzene_s1/mecis",
        "mecis_aligned": "data/benzene_s1/mecis_aligned",
    },
    "ethylene": {
        "spawns":        "data/ethylene/spawns",
        "reference":     "data/ethylene/mecis/Ethylidene_Bent.xyz",
        "mecis":         "data/ethylene/mecis",
        "mecis_aligned": "data/ethylene/mecis_aligned",
    },
    "butadiene_s0": {
        "spawns":        "data/butadiene_s0/spawns",
        "reference":     "data/butadiene_s0/mecis/type12.xyz",
        "mecis":         "data/butadiene_s0/mecis",
        "mecis_aligned": "data/butadiene_s0/mecis_aligned",
    },
    "butadiene_s1": {
        "spawns":        "data/butadiene_s1/spawns",
        "reference":     "data/butadiene_s1/mecis/type5.xyz",
        "mecis":         "data/butadiene_s1/mecis",
        "mecis_aligned": "data/butadiene_s1/mecis_aligned",
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


def _align_spawn_worker(args):
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    filename, metadata, tgt_coords, tgt_atoms, ref_atoms, aligned_mecis, allow_reflection = args
    best_rmsd      = float("inf")
    best_aligned   = None
    best_atoms     = None
    best_meci_name = None

    for meci_name, meci_coords in aligned_mecis:
        _, rmsd, _, _, aligned_coords, _, reordered_atoms = \
            _search_bruteforce_elementwise(
                meci_coords, tgt_coords, ref_atoms, tgt_atoms, allow_reflection,
            )
        if rmsd < best_rmsd:
            best_rmsd      = rmsd
            best_aligned   = aligned_coords
            best_atoms     = reordered_atoms
            best_meci_name = meci_name

    return filename, metadata, best_atoms, best_aligned, best_meci_name, best_rmsd


def align_multi_ref(spawns, reference, mecis_aligned_dir: Path, out_dir: Path,
                    n_workers: int = 1) -> None:
    """
    Align each spawn to the closest MECI centroid using a fresh brute-force
    permutation search per spawn-MECI pair.

    All MECIs are loaded from mecis_aligned_dir, where they have already been
    Kabsch-aligned (with optimal atom permutation) to the master reference.
    This means every centroid lives in the master reference frame.

    For each spawn, BF+Kabsch is run independently against every centroid.
    The centroid yielding the lowest RMSD is chosen, and the corresponding
    aligned coordinates (in the master reference frame) are written out.
    Spawns are processed in parallel using n_workers processes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    meci_files = sorted(mecis_aligned_dir.glob("*.xyz"))
    if not meci_files:
        print(f"  WARNING: no XYZ files found in {mecis_aligned_dir}")
        return

    aligned_mecis = [(read_xyz_file(mf).filename, read_xyz_file(mf).coordinates)
                     for mf in meci_files]
    print(f"  Loaded {len(aligned_mecis)} pre-aligned MECI centroids from {mecis_aligned_dir}")
    print(f"  Workers: {n_workers}")

    work_items = [
        (tgt.filename, tgt.metadata, tgt.coordinates, tgt.atoms,
         reference.atoms, aligned_mecis, ALLOW_REFLECTION)
        for tgt in spawns
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_align_spawn_worker, item): item[0] for item in work_items}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  multi_ref", unit="geom", leave=False):
            filename, metadata, best_atoms, best_aligned, best_meci_name, best_rmsd = \
                future.result()
            _write_xyz(out_dir / filename, best_atoms, best_aligned,
                       f"{metadata} | closest_meci={best_meci_name} rmsd={best_rmsd:.4f}")


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
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=os.cpu_count(),
        metavar="N",
        help=f"Number of parallel worker processes for multi_ref (default: {os.cpu_count()})",
    )
    args = parser.parse_args()

    if args.reference and not args.dataset:
        parser.error("--reference requires --dataset/-d")

    active_datasets = MEDOID_DATASETS if args.use_medoid else DATASETS
    datasets_to_run = (
        {args.dataset: active_datasets[args.dataset]} if args.dataset else active_datasets
    )

    for dataset_name, cfg in datasets_to_run.items():
        spawns_dir        = Path(cfg["spawns"])
        if args.spawns_subdir:
            spawns_dir = spawns_dir.parent / args.spawns_subdir
        ref_path          = Path(args.reference) if args.reference else Path(cfg["reference"])
        mecis_aligned_dir = Path(cfg["mecis_aligned"])
        csv_path          = Path(args.reports_dir) / f"{dataset_name}.csv"
        out_base          = Path("data") / dataset_name / args.aligned_subdir

        print(f"\n{'='*60}")
        print(f"Dataset       : {dataset_name}")
        print(f"Spawns        : {spawns_dir}")
        print(f"Ref           : {ref_path}")
        print(f"MECIs aligned : {mecis_aligned_dir}")
        print(f"CSV           : {csv_path}")

        # --- checks ---
        missing = [p for p in (spawns_dir, ref_path, mecis_aligned_dir, csv_path) if not p.exists()]
        if missing:
            for m in missing:
                print(f"  MISSING: {m}")
            print("  Skipping dataset.")
            continue

        reference    = read_xyz_file(ref_path)
        spawns       = read_all_geometries(spawns_dir)
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
        align_multi_ref(spawns, reference, mecis_aligned_dir, out_dir=multi_ref_dir,
                        n_workers=args.workers)
        print(f"  Written to {multi_ref_dir}")

        traj_path = out_base / "multi_ref_trajectory.xyz"
        n_frames = _write_trajectory(multi_ref_dir, traj_path)
        print(f"  Trajectory  → {traj_path}  ({n_frames} frames)")

    print("\nDone.")


if __name__ == "__main__":
    main()
