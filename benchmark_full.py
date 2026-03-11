"""
Full benchmark: brute-force vs. double-isomorphism vs. MCS+Hungarian
across all five datasets.

For each dataset:
  - Brute-force is the reference ("ground truth")
  - Double-isomorphism and MCS+Hungarian are evaluated against it
  - Results are written to reports/<dataset>.csv (per-geometry)
  - A summary table is printed to stdout

Usage:
    uv run python benchmark_full.py
    uv run python benchmark_full.py -n 50          # limit geometries per dataset
    uv run python benchmark_full.py -j 6           # worker count for brute-force
    uv run python benchmark_full.py --no-mcs       # skip MCS+Hungarian (slow)
"""

import argparse
import datetime
import io
import math
import os
import time
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from seamstress.alignment import (
    _search_automorphism,
    _search_bruteforce_elementwise,
    _search_mcs_alignment,
)
from seamstress.automorphism import get_automorphisms
from seamstress.connectivity import group_by_connectivity
from seamstress.geometry import read_all_geometries, read_xyz_file
from seamstress.processor import _compose_automorphisms
from seamstress.rdkit_utils import geometry_to_mol


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

DATASETS = {
    "benzene_s0": {
        "spawns":    "data/benzene_s0/spawns",
        "reference": "data/benzene_s0/mecis/Type_2.xyz",
    },
    "benzene_s1": {
        "spawns":    "data/benzene_s1/spawns",
        "reference": "data/benzene_s1/mecis/Type_3.xyz",
    },
    "ethylene": {
        "spawns":    "data/ethylene/spawns",
        "reference": "data/ethylene/mecis/Twist.xyz",
    },
    "butadiene_s0": {
        "spawns":    "data/butadiene_s0/spawns",
        "reference": "data/butadiene_s0/mecis/type12.xyz",
    },
    "butadiene_s1": {
        "spawns":    "data/butadiene_s1/spawns",
        "reference": "data/butadiene_s1/mecis/type7.xyz",
    },
}

# Medoid references determined by evaluate_reference.py
MEDOID_DATASETS = {
    "benzene_s0": {
        "spawns":    "data/benzene_s0/spawns",
        "reference": "data/benzene_s0/mecis/Type_4.xyz",
    },
    "benzene_s1": {
        "spawns":    "data/benzene_s1/spawns",
        "reference": "data/benzene_s1/mecis/Type_1.xyz",
    },
    "ethylene": {
        "spawns":    "data/ethylene/spawns",
        "reference": "data/ethylene/mecis/Ethylidene_Bent.xyz",
    },
    "butadiene_s0": {
        "spawns":    "data/butadiene_s0/spawns",
        "reference": "data/butadiene_s0/mecis/type12.xyz",
    },
    "butadiene_s1": {
        "spawns":    "data/butadiene_s1/spawns",
        "reference": "data/butadiene_s1/mecis/type5.xyz",
    },
}

ALLOW_REFLECTION = True
BOND_THRESHOLD   = 1.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perm_count(atoms):
    counts = Counter(atoms)
    result = 1
    for c in counts.values():
        result *= math.factorial(c)
    return result


# ---------------------------------------------------------------------------
# Worker functions (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _bf_worker(args):
    ref_coords, tgt_coords, ref_atoms, tgt_atoms, allow_reflection = args
    result = _search_bruteforce_elementwise(
        ref_coords, tgt_coords, ref_atoms, tgt_atoms, allow_reflection
    )
    best_perm, best_rmsd, *_ = result
    return best_perm, best_rmsd


def _mcs_worker(args):
    reference, target, allow_reflection = args
    try:
        result = _search_mcs_alignment(reference, target, allow_reflection)
        if result is None:
            return None, float("inf")
        best_perm, best_rmsd, *_ = result
        return best_perm, best_rmsd
    except Exception:
        return None, float("inf")


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------

def run_bruteforce(reference, targets, n_workers):
    ref_coords, ref_atoms = reference.coordinates, reference.atoms
    identity = tuple(range(len(ref_atoms)))

    worker_args = [
        (ref_coords, tgt.coordinates, ref_atoms, tgt.atoms, ALLOW_REFLECTION)
        for tgt in targets
    ]

    results_map = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_bf_worker, a): i for i, a in enumerate(worker_args)}
        with tqdm(total=len(targets), desc="  brute-force", unit="geom", leave=False) as bar:
            for fut in as_completed(futures):
                results_map[futures[fut]] = fut.result()
                bar.update(1)
    elapsed = time.perf_counter() - t0

    perms, rmsds, swapped = [], [], []
    for i in range(len(targets)):
        perm, rmsd = results_map[i]
        perms.append(perm)
        rmsds.append(rmsd)
        swapped.append(perm != identity)

    return {"perms": perms, "rmsds": rmsds, "swapped": swapped, "time_s": elapsed}


def run_double_isomorphism(reference, targets):
    ref_coords, ref_atoms = reference.coordinates, reference.atoms
    identity = tuple(range(len(ref_atoms)))

    ref_mol = geometry_to_mol(reference)
    master_autos = get_automorphisms(ref_mol)

    groups = group_by_connectivity(targets, cov_factor=BOND_THRESHOLD)
    by_filename = {}

    t0 = time.perf_counter()
    for _, group_geoms in groups.items():
        gmol = geometry_to_mol(group_geoms[0])
        family_autos = (
            get_automorphisms(gmol)
            if gmol is not None
            else [tuple(range(len(ref_atoms)))]
        )
        combined = _compose_automorphisms(master_autos, family_autos)
        for tgt in group_geoms:
            result = _search_automorphism(
                ref_coords, tgt.coordinates, combined, ref_atoms, tgt.atoms,
                ALLOW_REFLECTION,
            )
            best_perm, best_rmsd, *_ = result
            by_filename[tgt.filename] = (best_perm, best_rmsd, len(combined))
    elapsed = time.perf_counter() - t0

    perms, rmsds, swapped, nperms = [], [], [], []
    for tgt in targets:
        perm, rmsd, n = by_filename[tgt.filename]
        perms.append(perm)
        rmsds.append(rmsd)
        swapped.append(perm != identity)
        nperms.append(n)

    return {
        "perms": perms, "rmsds": rmsds, "swapped": swapped,
        "nperms_mean": sum(nperms) / len(nperms), "time_s": elapsed,
    }


def run_mcs_hungarian(reference, targets, n_workers):
    identity = tuple(range(len(reference.atoms)))
    worker_args = [(reference, tgt, ALLOW_REFLECTION) for tgt in targets]

    results_map = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_mcs_worker, a): i for i, a in enumerate(worker_args)}
        with tqdm(total=len(targets), desc="  MCS+Hungarian", unit="geom", leave=False) as bar:
            for fut in as_completed(futures):
                results_map[futures[fut]] = fut.result()
                bar.update(1)
    elapsed = time.perf_counter() - t0

    perms, rmsds, swapped = [], [], []
    for i in range(len(targets)):
        perm, rmsd = results_map[i]
        perms.append(perm)
        rmsds.append(rmsd)
        swapped.append(perm != identity if perm is not None else False)

    return {"perms": perms, "rmsds": rmsds, "swapped": swapped, "time_s": elapsed}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_csv_report(dataset_name, targets, bf, di, mcs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset_name}.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "bf_perm", "bf_rmsd", "bf_swap",
            "di_perm", "di_rmsd", "di_swap", "bf_eq_di",
            "mcs_perm", "mcs_rmsd", "mcs_swap", "bf_eq_mcs",
        ])
        for i, tgt in enumerate(targets):
            bp = bf["perms"][i]
            dp = di["perms"][i]
            mp = mcs["perms"][i] if mcs else None
            writer.writerow([
                tgt.filename,
                str(bp), f"{bf['rmsds'][i]:.4f}", bf["swapped"][i],
                str(dp), f"{di['rmsds'][i]:.4f}", di["swapped"][i],
                bp == dp,
                str(mp) if mp else "N/A",
                f"{mcs['rmsds'][i]:.4f}" if mcs else "N/A",
                mcs["swapped"][i] if mcs else "N/A",
                (bp == mp) if (mcs and mp is not None) else "N/A",
            ])

    return path


def write_txt_report(path, dataset_name, ref_path, summary, bf_count, n_workers):
    """Write a human-readable text summary alongside the CSV."""
    buf = io.StringIO()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    buf.write(f"Benchmark Report — {dataset_name}\n")
    buf.write("=" * 60 + "\n")
    buf.write(f"Run at      : {ts}\n")
    buf.write(f"Reference   : {ref_path}\n")
    buf.write(f"Geometries  : {summary['n']}\n")
    buf.write(f"Atoms       : {summary['n_atoms']}\n")
    buf.write(f"BF perms    : {bf_count:,}\n")
    buf.write(f"Workers     : {n_workers}\n\n")

    # --- Brute-force ---
    bf = summary["bf"]
    buf.write("Brute-Force\n")
    buf.write("-" * 40 + "\n")
    buf.write(f"  Time      : {bf['time_s']:.1f} s\n")
    buf.write(f"  Swaps     : {sum(bf['swapped'])}/{summary['n']} ({100*sum(bf['swapped'])/summary['n']:.1f}%)\n")
    buf.write(f"  RMSD mean : {sum(bf['rmsds'])/len(bf['rmsds']):.4f} Å\n")
    buf.write(f"  RMSD max  : {max(bf['rmsds']):.4f} Å\n\n")

    # --- Double-isomorphism ---
    di = summary["di"]
    buf.write("Double-Isomorphism\n")
    buf.write("-" * 40 + "\n")
    buf.write(f"  Time      : {di['time_s']:.2f} s  (speedup: {bf['time_s']/di['time_s']:.0f}x)\n")
    buf.write(f"  Perms (mean): {di['nperms_mean']:.0f}  (vs {bf_count:,} BF)\n")
    buf.write(f"  Agree w/ BF : {summary['di_agree']}/{summary['n']} ({100*summary['di_agree']/summary['n']:.1f}%)\n")
    buf.write(f"  RMSD mean : {sum(di['rmsds'])/len(di['rmsds']):.4f} Å\n")
    buf.write(f"  RMSD max  : {max(di['rmsds']):.4f} Å\n\n")

    # --- MCS+Hungarian ---
    mcs = summary.get("mcs")
    if mcs is not None:
        buf.write("MCS + Hungarian\n")
        buf.write("-" * 40 + "\n")
        buf.write(f"  Time      : {mcs['time_s']:.1f} s  (speedup: {bf['time_s']/mcs['time_s']:.1f}x)\n")
        buf.write(f"  Agree w/ BF : {summary['mcs_agree']}/{summary['n']} ({100*summary['mcs_agree']/summary['n']:.1f}%)\n")
        valid_rmsds = [r for r, p in zip(mcs['rmsds'], mcs['perms']) if p is not None]
        if valid_rmsds:
            buf.write(f"  RMSD mean : {sum(valid_rmsds)/len(valid_rmsds):.4f} Å\n")
            buf.write(f"  RMSD max  : {max(valid_rmsds):.4f} Å\n")
    else:
        buf.write("MCS + Hungarian : skipped\n")

    path.write_text(buf.getvalue())


def print_summary_table(all_results):
    """Print a summary table across all datasets."""

    col_w = [14, 8, 10, 14, 10, 14, 10, 12, 12]
    header = [
        "Dataset", "N", "BF perms",
        "DI agree", "DI time",
        "MCS agree", "MCS time",
        "BF time", "DI nperms",
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print()
    print(sep)
    print(fmt.format(*header))
    print(sep)

    for row in all_results:
        n = row["n"]
        bf_count = row["bf_perm_count"]
        di_agree = row["di_agree"]
        mcs_agree = row["mcs_agree"]

        di_pct  = f"{di_agree}/{n} ({100*di_agree/n:.0f}%)"  if n else "N/A"
        mcs_pct = f"{mcs_agree}/{n} ({100*mcs_agree/n:.0f}%)" if (n and mcs_agree is not None) else "N/A"

        print(fmt.format(
            row["dataset"],
            str(n),
            f"{bf_count:,}",
            di_pct,
            f"{row['di_time']:.1f}s",
            mcs_pct,
            f"{row['mcs_time']:.1f}s" if row["mcs_time"] is not None else "skip",
            f"{row['bf_time']:.1f}s",
            f"{row['di_nperms']:.0f}",
        ))

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset", choices=list(DATASETS.keys()), default=None,
                        help="Run only this dataset (default: all)")
    parser.add_argument("-r", "--reference", default=None, metavar="XYZ",
                        help="Override the reference XYZ file (requires -d)")
    parser.add_argument("-n", "--nsamples", type=int, default=None,
                        help="Max geometries per dataset (default: all)")
    parser.add_argument("-j", "--workers", type=int, default=os.cpu_count(),
                        help=f"Parallel workers (default: {os.cpu_count()})")
    parser.add_argument("--no-mcs", action="store_true",
                        help="Skip MCS+Hungarian (useful when it is slow)")
    parser.add_argument("--reports-dir", default="reports",
                        help="Directory for per-geometry CSV reports (default: reports/)")
    parser.add_argument("--use-medoid", action="store_true",
                        help="Use medoid references instead of the default ones")
    args = parser.parse_args()

    if args.reference and not args.dataset:
        parser.error("--reference requires --dataset/-d")

    out_dir = Path(args.reports_dir)
    all_summary = []

    active_datasets = MEDOID_DATASETS if args.use_medoid else DATASETS
    datasets_to_run = (
        {args.dataset: active_datasets[args.dataset]} if args.dataset else active_datasets
    )

    for dataset_name, cfg in datasets_to_run.items():
        spawns_dir = Path(cfg["spawns"])
        ref_path   = Path(args.reference) if args.reference else Path(cfg["reference"])

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"  Reference : {ref_path}")

        if not spawns_dir.exists() or not any(spawns_dir.iterdir()):
            print(f"  Spawns dir empty or missing — skipping.")
            continue
        if not ref_path.exists():
            print(f"  Reference file not found — skipping.")
            continue

        reference = read_xyz_file(ref_path)
        all_geoms = read_all_geometries(spawns_dir)
        targets   = all_geoms if args.nsamples is None else all_geoms[:args.nsamples]

        bf_count = _perm_count(reference.atoms)
        print(f"  Spawns    : {len(targets)} (of {len(all_geoms)} total)")
        print(f"  Atoms     : {len(reference.atoms)}")
        print(f"  BF perms  : {bf_count:,}")
        print(f"  Workers   : {args.workers}")

        # --- brute-force ---
        print(f"  Running brute-force...")
        bf = run_bruteforce(reference, targets, args.workers)
        print(f"    Done in {bf['time_s']:.1f}s  |  swaps: {sum(bf['swapped'])}/{len(targets)}")

        # --- double-isomorphism ---
        print(f"  Running double-isomorphism...")
        di = run_double_isomorphism(reference, targets)
        di_agree = sum(bp == dp for bp, dp in zip(bf["perms"], di["perms"]))
        print(f"    Done in {di['time_s']:.2f}s  |  agree: {di_agree}/{len(targets)}  |  nperms: {di['nperms_mean']:.0f}")

        # --- MCS + Hungarian ---
        mcs = None
        mcs_agree = None
        mcs_time  = None
        if not args.no_mcs:
            print(f"  Running MCS+Hungarian...")
            mcs = run_mcs_hungarian(reference, targets, args.workers)
            mcs_agree = sum(
                bp == mp for bp, mp in zip(bf["perms"], mcs["perms"])
                if mp is not None
            )
            mcs_time = mcs["time_s"]
            print(f"    Done in {mcs['time_s']:.1f}s  |  agree: {mcs_agree}/{len(targets)}")

        # --- write CSV ---
        csv_path = write_csv_report(dataset_name, targets, bf, di, mcs, out_dir)
        print(f"  Report    : {csv_path}")

        # --- write TXT summary ---
        summary_data = {
            "n": len(targets), "n_atoms": len(reference.atoms),
            "bf": bf, "di": di,
            "di_agree": di_agree, "mcs_agree": mcs_agree,
        }
        if mcs is not None:
            summary_data["mcs"] = mcs
        txt_path = out_dir / f"{dataset_name}.txt"
        write_txt_report(txt_path, dataset_name, ref_path, summary_data, bf_count, args.workers)
        print(f"  Summary   : {txt_path}")

        all_summary.append({
            "dataset":      dataset_name,
            "n":            len(targets),
            "bf_perm_count": bf_count,
            "bf_time":      bf["time_s"],
            "di_agree":     di_agree,
            "di_time":      di["time_s"],
            "di_nperms":    di["nperms_mean"],
            "mcs_agree":    mcs_agree,
            "mcs_time":     mcs_time,
        })

    print_summary_table(all_summary)


if __name__ == "__main__":
    main()
