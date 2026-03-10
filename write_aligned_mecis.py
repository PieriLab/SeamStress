"""
For each dataset, find the MECI medoid, align all MECIs to it, and write:

  data/<dataset>/mecis_aligned/
      All MECI XYZ files, permuted and Kabsch-aligned to the medoid.
      The medoid itself is written as-is (identity alignment).

  data/<dataset>/mecis_medoid_report.txt
      - Which MECI is the medoid
      - Per-MECI RMSD to the medoid
      - Full pairwise RMSD table
      - Mean RMSD of each MECI to all others

Reflection is allowed in all Kabsch calls.

Usage:
    uv run python write_aligned_mecis.py
    uv run python write_aligned_mecis.py -d benzene_s0
"""

import argparse
import io
from pathlib import Path

import numpy as np

from seamstress.alignment import (
    _search_bruteforce_elementwise,
    kabsch_align_rmsd,
)
from seamstress.geometry import read_all_geometries, read_xyz_file

# ---------------------------------------------------------------------------
# Dataset configuration (same as benchmark_full.py)
# ---------------------------------------------------------------------------

DATASETS = {
    "benzene_s0":  {"mecis": "data/benzene_s0/mecis"},
    "benzene_s1":  {"mecis": "data/benzene_s1/mecis"},
    "ethylene":    {"mecis": "data/ethylene/mecis"},
    "butadiene_s0":{"mecis": "data/butadiene_s0/mecis"},
    "butadiene_s1":{"mecis": "data/butadiene_s1/mecis"},
}

# Current single references (used as initial common frame for medoid search)
CURRENT_REFS = {
    "benzene_s0":   "data/benzene_s0/mecis/Type_2.xyz",
    "benzene_s1":   "data/benzene_s1/mecis/Type_3.xyz",
    "ethylene":     "data/ethylene/mecis/Twist.xyz",
    "butadiene_s0": "data/butadiene_s0/mecis/type12.xyz",
    "butadiene_s1": "data/butadiene_s1/mecis/type7.xyz",
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


def align_to_reference(meci, reference):
    """Brute-force align meci to reference. Returns (aligned_coords, perm, rmsd)."""
    perm, rmsd, _, _, aligned, _, _ = _search_bruteforce_elementwise(
        reference.coordinates, meci.coordinates,
        reference.atoms, meci.atoms,
        ALLOW_REFLECTION,
    )
    return aligned, perm, rmsd


def compute_pairwise_rmsd(aligned_coords_list: list, ref_atoms: list) -> np.ndarray:
    """
    Pairwise Kabsch RMSD between all aligned MECIs (atom ordering already fixed).
    Returns symmetric N×N matrix.
    """
    n   = len(aligned_coords_list)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            _, rmsd = kabsch_align_rmsd(
                aligned_coords_list[i], aligned_coords_list[j],
                ref_atoms, ref_atoms,
                use_all_atoms=True,
                allow_reflection=ALLOW_REFLECTION,
            )
            mat[i, j] = mat[j, i] = rmsd
    return mat


def find_medoid(names, rmsd_matrix):
    mean_rmsds = rmsd_matrix.mean(axis=1)
    idx = int(np.argmin(mean_rmsds))
    return idx, names[idx], mean_rmsds


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset_name: str, cfg: dict):
    mecis_dir  = Path(cfg["mecis"])
    ref_path   = Path(CURRENT_REFS[dataset_name])
    out_dir    = Path("data") / dataset_name / "mecis_aligned"
    report_path = Path("data") / dataset_name / "mecis_medoid_report.txt"

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}")

    mecis     = read_all_geometries(mecis_dir)
    reference = read_xyz_file(ref_path)
    n         = len(mecis)
    print(f"MECIs   : {n}")

    # Step 1 — align all MECIs to current reference (common frame for medoid search)
    print("Step 1: aligning all MECIs to current reference (brute-force)...")
    aligned_to_ref = []   # list of (filename, aligned_coords, perm)
    for meci in mecis:
        if meci.filename == reference.filename:
            aligned_to_ref.append((meci.filename, reference.coordinates.copy(), tuple(range(n))))
        else:
            coords, perm, _ = align_to_reference(meci, reference)
            aligned_to_ref.append((meci.filename, coords, perm))

    names      = [x[0] for x in aligned_to_ref]
    coords_ref = [x[1] for x in aligned_to_ref]

    # Step 2 — pairwise RMSD matrix → medoid
    print("Step 2: computing pairwise RMSD matrix...")
    rmsd_matrix = compute_pairwise_rmsd(coords_ref, reference.atoms)
    med_idx, medoid_name, mean_rmsds = find_medoid(names, rmsd_matrix)
    medoid_meci = mecis[names.index(medoid_name)]

    print(f"  Medoid : {medoid_name}  (mean RMSD to others = {mean_rmsds[med_idx]:.4f} Å)")

    # Step 3 — align all MECIs to the medoid (fresh brute-force alignment)
    print("Step 3: aligning all MECIs to medoid (brute-force)...")
    out_dir.mkdir(parents=True, exist_ok=True)

    rmsd_to_medoid = {}
    for meci in mecis:
        if meci.filename == medoid_name:
            # medoid: write as-is
            aligned_coords = meci.coordinates.copy()
            rmsd_val       = 0.0
        else:
            aligned_coords, _, rmsd_val = align_to_reference(meci, medoid_meci)

        rmsd_to_medoid[meci.filename] = rmsd_val
        _write_xyz(
            out_dir / meci.filename,
            medoid_meci.atoms,       # atom labels in medoid ordering
            aligned_coords,
            f"{meci.metadata}  | aligned_to={medoid_name}  rmsd={rmsd_val:.4f}",
        )

    # Step 4 — write text report
    _write_report(
        report_path, dataset_name, medoid_name, mean_rmsds,
        names, rmsd_matrix, rmsd_to_medoid,
    )

    print(f"  Aligned MECIs → {out_dir}/")
    print(f"  Report        → {report_path}")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(path, dataset_name, medoid_name, mean_rmsds_all,
                  names, rmsd_matrix, rmsd_to_medoid):
    buf = io.StringIO()

    buf.write(f"MECI Medoid Report — {dataset_name}\n")
    buf.write("=" * 60 + "\n\n")

    buf.write(f"Medoid : {medoid_name}\n")
    buf.write(f"  Mean RMSD to all other MECIs : "
              f"{mean_rmsds_all[names.index(medoid_name)]:.4f} Å\n\n")

    # Per-MECI RMSD to medoid
    buf.write("RMSD to medoid (after brute-force alignment)\n")
    buf.write("-" * 40 + "\n")
    sorted_by_rmsd = sorted(rmsd_to_medoid.items(), key=lambda x: x[1])
    for fname, rmsd in sorted_by_rmsd:
        marker = "  ← MEDOID" if fname == medoid_name else ""
        buf.write(f"  {fname:<30s}  {rmsd:.4f} Å{marker}\n")
    buf.write("\n")

    # Mean RMSD of each MECI to all others (ranking)
    buf.write("Mean RMSD to all other MECIs (ranking)\n")
    buf.write("-" * 40 + "\n")
    ranked = sorted(zip(names, mean_rmsds_all), key=lambda x: x[1])
    for fname, mean_r in ranked:
        marker = "  ← MEDOID" if fname == medoid_name else ""
        buf.write(f"  {fname:<30s}  {mean_r:.4f} Å{marker}\n")
    buf.write("\n")

    # Full pairwise RMSD matrix
    n = len(names)
    col_w = max(len(nm.replace(".xyz", "")) for nm in names) + 2
    buf.write("Full pairwise RMSD matrix (Å)\n")
    buf.write("-" * 40 + "\n")
    header = " " * col_w + "  " + "  ".join(f"{nm.replace('.xyz',''):>{col_w}}" for nm in names)
    buf.write(header + "\n")
    for i, row_name in enumerate(names):
        row = f"{row_name.replace('.xyz',''):<{col_w}}  "
        row += "  ".join(f"{rmsd_matrix[i, j]:>{col_w}.4f}" for j in range(n))
        buf.write(row + "\n")

    path.write_text(buf.getvalue())


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
    args = parser.parse_args()

    datasets_to_run = (
        {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    )

    for dataset_name, cfg in datasets_to_run.items():
        run_dataset(dataset_name, cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
