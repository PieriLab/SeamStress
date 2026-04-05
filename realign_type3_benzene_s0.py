"""
Redo brute-force permutation alignment of all benzene_s0 spawns tagged
closest_meci=Type_3.xyz, using Type_3.xyz (from mecis_aligned) as the sole
reference. Reflection is allowed (matching align_dataset.py ALLOW_REFLECTION=True).

Results are saved to data/benzene_s0/aligned_spawns/multi_ref_type3_realigned/.
A comparison CSV is written to data/benzene_s0/type3_realignment_comparison.csv
showing old RMSD, new RMSD, and whether coordinates changed.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from seamstress.alignment import _search_bruteforce_elementwise
from seamstress.geometry import read_xyz_file

# ── Config ────────────────────────────────────────────────────────────────────
MULTI_REF_DIR = Path("data/benzene_s0/aligned_spawns/multi_ref")
OUT_DIR       = Path("data/benzene_s0/aligned_spawns/multi_ref_type3_realigned")
REFERENCE_XYZ = Path("data/benzene_s0/mecis_aligned/Type_3.xyz")
COMPARISON_CSV = Path("data/benzene_s0/type3_realignment_comparison.csv")
ALLOW_REFLECTION = True
N_WORKERS = 8

# ── Helpers ───────────────────────────────────────────────────────────────────

def write_xyz(path, atoms, coords, comment):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for atom, (x, y, z) in zip(atoms, coords):
            f.write(f" {atom}  {x:.9f}  {y:.9f}  {z:.9f}\n")


def worker(args):
    fpath, ref_coords, ref_atoms = args
    tgt = read_xyz_file(fpath)
    old_rmsd_match = re.search(r"rmsd=([\d.]+)", tgt.metadata or "")
    old_rmsd = float(old_rmsd_match.group(1)) if old_rmsd_match else float("nan")

    _, new_rmsd, _, _, aligned_coords, _, reordered_atoms = \
        _search_bruteforce_elementwise(
            ref_coords, tgt.coordinates, ref_atoms, tgt.atoms, ALLOW_REFLECTION
        )

    return {
        "filename":    fpath.name,
        "old_rmsd":    old_rmsd,
        "new_rmsd":    new_rmsd,
        "rmsd_change": new_rmsd - old_rmsd,
        "metadata":    tgt.metadata,
        "reordered_atoms": reordered_atoms,
        "aligned_coords":  aligned_coords,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Type_3 reference
    ref = read_xyz_file(REFERENCE_XYZ)
    print(f"Reference: {REFERENCE_XYZ}  ({len(ref.atoms)} atoms)")

    # Find all type_3-tagged spawns
    type3_files = [
        p for p in sorted(MULTI_REF_DIR.glob("*.xyz"))
        if "closest_meci=Type_3.xyz" in p.read_text().split("\n")[1]
    ]
    print(f"Type_3-tagged spawns: {len(type3_files)}")

    args = [(f, ref.coordinates, ref.atoms) for f in type3_files]

    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(worker, a): a[0].name for a in args}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Realigning", unit="geom"):
            r = future.result()
            # Write new XYZ
            new_comment = re.sub(r"closest_meci=\S+\s*rmsd=[\d.]+",
                                 f"closest_meci=Type_3.xyz rmsd={r['new_rmsd']:.4f} [realigned]",
                                 r["metadata"] or "")
            write_xyz(OUT_DIR / r["filename"],
                      r["reordered_atoms"], r["aligned_coords"], new_comment)
            rows.append({k: v for k, v in r.items()
                         if k not in ("reordered_atoms", "aligned_coords")})

    df = pd.DataFrame(rows).sort_values("filename")
    df["coords_changed"] = df["rmsd_change"].abs() > 1e-4

    df.to_csv(COMPARISON_CSV, index=False)
    print(f"\nSaved comparison -> {COMPARISON_CSV}")
    print(f"\nSummary:")
    print(f"  Spawns realigned        : {len(df)}")
    print(f"  RMSD improved (new<old) : {(df.rmsd_change < -1e-4).sum()}")
    print(f"  RMSD worsened (new>old) : {(df.rmsd_change >  1e-4).sum()}")
    print(f"  Unchanged               : {(df.rmsd_change.abs() <= 1e-4).sum()}")
    print(f"  Mean old RMSD           : {df.old_rmsd.mean():.4f}")
    print(f"  Mean new RMSD           : {df.new_rmsd.mean():.4f}")
    print(f"\nLargest improvements:")
    print(df.nsmallest(10, "rmsd_change")[["filename","old_rmsd","new_rmsd","rmsd_change"]].to_string(index=False))


if __name__ == "__main__":
    main()
