"""
Benchmark brute-force vs. double-isomorphism permutation methods.

Aligns a subset of geometries to a master reference centroid and compares:
  - Number of permutations tested
  - Wall-clock time
  - RMSD achieved
  - Swaps caught (permutation != identity)

Usage:
    python benchmark_permutations.py \\
        -f data/benzene/spawns -c data/benzene/centroids \\
        --master meci2.xyz --allow-reflection

    # Limit to first 100 geometries, use 8 workers for brute-force:
    python benchmark_permutations.py \\
        -f data/benzene/spawns -c data/benzene/centroids \\
        --master meci2.xyz -n 100 -j 8 --allow-reflection
"""

import argparse
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from seamstress.alignment import _search_automorphism, _search_bruteforce_elementwise
from seamstress.automorphism import get_automorphisms
from seamstress.connectivity import group_by_connectivity
from seamstress.geometry import read_all_geometries, read_xyz_file
from seamstress.processor import _compose_automorphisms
from seamstress.rdkit_utils import geometry_to_mol


def _bruteforce_perm_count(atoms: list[str]) -> int:
    """Analytically compute the number of element-constrained permutations."""
    counts = Counter(atoms)
    result = 1
    for c in counts.values():
        result *= math.factorial(c)
    return result


def _bruteforce_one(args):
    """Worker: run brute-force on a single geometry. Must be module-level for pickling."""
    ref_coords, tgt_coords, ref_atoms, tgt_atoms, allow_reflection = args
    result = _search_bruteforce_elementwise(
        ref_coords, tgt_coords, ref_atoms, tgt_atoms, allow_reflection
    )
    best_perm, best_rmsd, *_ = result
    return best_perm, best_rmsd


def _run_bruteforce(reference, targets, allow_reflection, n_workers):
    ref_coords = reference.coordinates
    ref_atoms = reference.atoms
    identity = tuple(range(len(ref_atoms)))
    bf_count = _bruteforce_perm_count(ref_atoms)

    worker_args = [
        (ref_coords, tgt.coordinates, ref_atoms, tgt.atoms, allow_reflection)
        for tgt in targets
    ]

    results_map = {}  # index -> (perm, rmsd)
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_bruteforce_one, a): i for i, a in enumerate(worker_args)}
        with tqdm(total=len(targets), unit="geom") as bar:
            for fut in as_completed(futures):
                idx = futures[fut]
                results_map[idx] = fut.result()
                bar.update(1)

    elapsed = time.perf_counter() - t0

    per_geom_perm = []
    per_geom_rmsd = []
    per_geom_swapped = []
    for i in range(len(targets)):
        perm, rmsd = results_map[i]
        per_geom_perm.append(perm)
        per_geom_rmsd.append(rmsd)
        per_geom_swapped.append(perm != identity)

    return {
        "method": "brute-force",
        "n_geoms": len(targets),
        "time_s": elapsed,
        "mean_rmsd": sum(per_geom_rmsd) / len(per_geom_rmsd),
        "max_rmsd": max(per_geom_rmsd),
        "mean_nperms": float(bf_count),
        "swaps_caught": sum(per_geom_swapped),
        "per_geom_perm": per_geom_perm,
        "per_geom_rmsd": per_geom_rmsd,
        "per_geom_swapped": per_geom_swapped,
    }


def _run_double_isomorphism(reference, targets, master_automorphisms, bond_threshold, allow_reflection):
    ref_coords = reference.coordinates
    ref_atoms = reference.atoms
    identity = tuple(range(len(ref_atoms)))

    groups = group_by_connectivity(targets, cov_factor=bond_threshold)
    results_by_filename = {}

    t0 = time.perf_counter()
    for _, group_geoms in groups.items():
        group_mol = geometry_to_mol(group_geoms[0])
        family_automorphisms = (
            get_automorphisms(group_mol)
            if group_mol is not None
            else [tuple(range(len(group_geoms[0].atoms)))]
        )
        combined = _compose_automorphisms(master_automorphisms, family_automorphisms)
        n_combined = len(combined)

        for target in group_geoms:
            result = _search_automorphism(
                ref_coords, target.coordinates, combined, ref_atoms, target.atoms, allow_reflection
            )
            best_perm, best_rmsd, *_ = result
            results_by_filename[target.filename] = (best_perm, best_rmsd, n_combined)

    elapsed = time.perf_counter() - t0

    per_geom_perm = []
    per_geom_rmsd = []
    per_geom_nperms = []
    per_geom_swapped = []
    for target in targets:
        best_perm, best_rmsd, n_combined = results_by_filename[target.filename]
        per_geom_perm.append(best_perm)
        per_geom_rmsd.append(best_rmsd)
        per_geom_nperms.append(n_combined)
        per_geom_swapped.append(best_perm != identity)

    return {
        "method": "double-isomorphism",
        "n_geoms": len(targets),
        "time_s": elapsed,
        "mean_rmsd": sum(per_geom_rmsd) / len(per_geom_rmsd),
        "max_rmsd": max(per_geom_rmsd),
        "mean_nperms": sum(per_geom_nperms) / len(per_geom_nperms),
        "swaps_caught": sum(per_geom_swapped),
        "per_geom_perm": per_geom_perm,
        "per_geom_rmsd": per_geom_rmsd,
        "per_geom_swapped": per_geom_swapped,
    }


def _print_table(results: list[dict], n: int) -> None:
    col_w = [22, 18, 10, 16, 14, 14]
    header = [
        "Method", "N perms (mean)", "Time (s)",
        "Mean RMSD (Å)", "Max RMSD (Å)", "Swaps caught",
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    row_fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print()
    print(sep)
    print(row_fmt.format(*header))
    print(sep)
    for r in results:
        print(row_fmt.format(
            r["method"],
            f"{r['mean_nperms']:.0f}",
            f"{r['time_s']:.2f}",
            f"{r['mean_rmsd']:.4f}",
            f"{r['max_rmsd']:.4f}",
            f"{r['swaps_caught']} / {n}",
        ))
    print(sep)

    if len(results) == 2:
        bf, di = results[0], results[1]
        diffs = [abs(a - b) for a, b in zip(bf["per_geom_rmsd"], di["per_geom_rmsd"])]
        print(f"\nRMSD agreement (|brute-force - double-isomorphism|):")
        print(f"  Mean difference : {sum(diffs)/len(diffs):.6f} Å")
        print(f"  Max  difference : {max(diffs):.6f} Å")

        disagreements = [
            i for i, (a, b) in enumerate(zip(bf["per_geom_swapped"], di["per_geom_swapped"]))
            if a != b
        ]
        print(f"  Swap disagreements: {len(disagreements)} / {len(diffs)}")
    print()


def _print_disagreements(bf_stats: dict, di_stats: dict, targets: list) -> None:
    identity = tuple(range(len(targets[0].atoms)))

    disagreements = [
        i for i, (a, b) in enumerate(
            zip(bf_stats["per_geom_swapped"], di_stats["per_geom_swapped"])
        )
        if a != b
    ]

    if not disagreements:
        print("No swap disagreements.\n")
        return

    print(f"{'='*80}")
    print(f"Swap disagreements ({len(disagreements)} geometries):")
    print(f"{'='*80}")

    for i in disagreements:
        tgt = targets[i]
        bf_perm = bf_stats["per_geom_perm"][i]
        bf_rmsd = bf_stats["per_geom_rmsd"][i]
        di_perm = di_stats["per_geom_perm"][i]
        di_rmsd = di_stats["per_geom_rmsd"][i]

        bf_swap = bf_perm != identity
        di_swap = di_perm != identity

        print(f"\n  {tgt.filename}")
        print(f"    brute-force       : swap={bf_swap}  RMSD={bf_rmsd:.4f} Å  perm={bf_perm}")
        print(f"    double-isomorphism: swap={di_swap}  RMSD={di_rmsd:.4f} Å  perm={di_perm}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--folder", required=True,
                        help="Folder containing spawning geometry XYZ files")
    parser.add_argument("-c", "--centroids", required=True,
                        help="Folder containing centroid XYZ files")
    parser.add_argument("--master", required=True,
                        help="Path to master reactant geometry XYZ file "
                             "(e.g. data/benzene/benzene_opt.xyz)")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Number of geometries to benchmark (default: all)")
    parser.add_argument("-j", "--workers", type=int, default=os.cpu_count(),
                        help=f"Parallel workers for brute-force (default: {os.cpu_count()} = all CPUs)")
    parser.add_argument("--bond-threshold", type=float, default=1.3,
                        help="Covalent factor for bond detection (default: 1.3)")
    parser.add_argument("--allow-reflection", action="store_true",
                        help="Allow improper rotations during alignment")
    args = parser.parse_args()

    master_file = Path(args.master)
    if not master_file.exists():
        raise FileNotFoundError(f"Master geometry not found: {master_file}")

    reference = read_xyz_file(master_file)
    master_mol = geometry_to_mol(reference)
    master_automorphisms = get_automorphisms(master_mol)

    all_geoms = read_all_geometries(Path(args.folder))
    targets = all_geoms if args.nsamples is None else all_geoms[: args.nsamples]

    bf_count = _bruteforce_perm_count(reference.atoms)

    print(f"Master reference      : {args.master} ({len(reference.atoms)} atoms)")
    print(f"Master automorphisms  : {len(master_automorphisms)}")
    print(f"Geometries            : {len(targets)} (of {len(all_geoms)} total)")
    print(f"Allow reflection      : {args.allow_reflection}")
    print(f"Bond threshold        : {args.bond_threshold}")
    print(f"Brute-force perms     : {bf_count:,}")
    print(f"Workers (brute-force) : {args.workers}\n")

    print(f"Running double-isomorphism on {len(targets)} geometries...")
    di_stats = _run_double_isomorphism(
        reference, targets, master_automorphisms, args.bond_threshold, args.allow_reflection
    )
    print(f"  Done in {di_stats['time_s']:.2f}s\n")

    print(f"Running brute-force on {len(targets)} geometries ({args.workers} workers)...")
    bf_stats = _run_bruteforce(reference, targets, args.allow_reflection, args.workers)
    print(f"  Done in {bf_stats['time_s']:.2f}s")

    _print_table([bf_stats, di_stats], len(targets))
    _print_disagreements(bf_stats, di_stats, targets)


if __name__ == "__main__":
    main()
