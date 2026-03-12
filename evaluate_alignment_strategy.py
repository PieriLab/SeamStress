"""
evaluate_alignment_strategy.py
================================
Evaluate single-reference vs. multi-reference-RMSD alignment strategies by
comparing their induced pairwise distance matrices to the optimal (brute-force)
pairwise RMSD matrix D_opt.

For each dataset and alignment strategy m we compute D^(m) — the matrix of
pairwise Kabsch RMSDs between aligned geometries (no permutation search needed
since atom ordering is already fixed by the alignment step) — and then measure
how closely D^(m) reproduces D_opt using:

  • Spearman correlation of upper-triangle elements
  • Pearson  correlation of upper-triangle elements
  • Frobenius norm  ||D^(m) - D_opt||_F  (absolute and relative)
  • Mean squared error
  • k-NN overlap  (fraction of k nearest neighbours shared with D_opt)
    for k = 5 %, 10 %, 20 % of n

D_opt is computed by brute-force permutation + Kabsch on either the full spawn
set (small datasets) or a random sample of n_sample geometries (benzene, where
518 400 permutations make the all-pairs computation expensive).

Usage
-----
    uv run python evaluate_alignment_strategy.py -d ethylene
    uv run python evaluate_alignment_strategy.py -d benzene_s0 -n 100
    uv run python evaluate_alignment_strategy.py          # all datasets
    uv run python evaluate_alignment_strategy.py --load-dopt reports/dopt_ethylene.csv -d ethylene
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from seamstress.alignment import _search_bruteforce_elementwise, kabsch_align_rmsd
from seamstress.geometry import read_xyz_file

# ── Dataset configuration ────────────────────────────────────────────────────

DATASETS = {
    "benzene_s0": {
        "spawns":     Path("data/benzene_s0/spawns"),
        "single_ref": Path("data/benzene_s0/aligned_spawns/single_ref"),
        "multi_ref":  Path("data/benzene_s0/aligned_spawns/multi_ref"),
        "n_sample":   100,   # 518 400 perms/pair → subset for tractability
    },
    "benzene_s1": {
        "spawns":     Path("data/benzene_s1/spawns"),
        "single_ref": Path("data/benzene_s1/aligned_spawns/single_ref"),
        "multi_ref":  Path("data/benzene_s1/aligned_spawns/multi_ref"),
        "n_sample":   100,
    },
    "ethylene": {
        "spawns":     Path("data/ethylene/spawns"),
        "single_ref": Path("data/ethylene/aligned_spawns/single_ref"),
        "multi_ref":  Path("data/ethylene/aligned_spawns/multi_ref"),
        "n_sample":   None,  # 48 perms/pair → full set is fast
    },
    "butadiene_s0": {
        "spawns":     Path("data/butadiene_s0/spawns"),
        "single_ref": Path("data/butadiene_s0/aligned_spawns/single_ref"),
        "multi_ref":  Path("data/butadiene_s0/aligned_spawns/multi_ref"),
        "n_sample":   200,   # 17 280 perms/pair → manageable subset
    },
    "butadiene_s1": {
        "spawns":     Path("data/butadiene_s1/spawns"),
        "single_ref": Path("data/butadiene_s1/aligned_spawns/single_ref"),
        "multi_ref":  Path("data/butadiene_s1/aligned_spawns/multi_ref"),
        "n_sample":   200,
    },
}

ALLOW_REFLECTION = True


# ── Pairwise D_opt (brute-force) ─────────────────────────────────────────────

def _bf_pair(args):
    """Worker: brute-force RMSD for a single geometry pair (serial, no nested MP)."""
    coords_i, coords_j, atoms = args
    _, rmsd, *_ = _search_bruteforce_elementwise(
        coords_i, coords_j, atoms, atoms, allow_reflection=ALLOW_REFLECTION
    )
    return rmsd


def compute_D_opt(spawn_dir: Path, sample_indices, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """
    Compute the brute-force pairwise RMSD matrix D_opt for the sampled geometries.

    Parameters
    ----------
    spawn_dir     : directory of raw (un-aligned) XYZ spawns
    sample_indices: list/array of integer indices selecting which spawns to use
    seed          : random seed (used only for documentation; sampling is done
                    outside this function)

    Returns
    -------
    D   : (n, n) symmetric matrix of brute-force pairwise RMSDs
    names : list of spawn filenames (stems) in row/column order
    """
    paths = sorted(spawn_dir.glob("*.xyz"))
    selected = [paths[i] for i in sample_indices]
    geoms    = [read_xyz_file(p) for p in selected]
    names    = [Path(p).stem for p in selected]
    atoms    = geoms[0].atoms

    n = len(geoms)
    D = np.zeros((n, n))

    pairs = list(combinations(range(n), 2))
    for i, j in tqdm(pairs, desc="  D_opt pairs", leave=False):
        rmsd = _bf_pair((geoms[i].coordinates, geoms[j].coordinates, atoms))
        D[i, j] = D[j, i] = rmsd

    return D, names


# ── Pairwise D^(m) from aligned geometries ───────────────────────────────────

def compute_D_strategy(
    aligned_dir: Path,
    filenames: list[str],
) -> np.ndarray:
    """
    Compute pairwise Kabsch RMSD from already-aligned geometries.

    No permutation search is performed — atom ordering is already fixed by the
    alignment step (all geometries permuted to the same reference ordering).
    Kabsch rotation is still applied to handle residual frame differences
    (important for multi_ref where each geometry was Kabsch-aligned to a
    different MECI centroid).

    Parameters
    ----------
    aligned_dir : directory containing one XYZ file per spawn (filenames with
                  .xyz extension)
    filenames   : stems (no extension) of the specific geometries to include,
                  in the desired order

    Returns
    -------
    D : (n, n) symmetric pairwise RMSD matrix
    """
    geoms = []
    missing = []
    for stem in filenames:
        p = aligned_dir / f"{stem}.xyz"
        if not p.exists():
            missing.append(stem)
        else:
            geoms.append(read_xyz_file(p))

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} geometries not found in {aligned_dir}: {missing[:5]} ..."
        )

    n = len(geoms)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            _, rmsd = kabsch_align_rmsd(
                geoms[i].coordinates,
                geoms[j].coordinates,
                atoms1=geoms[i].atoms,
                atoms2=geoms[j].atoms,
                use_all_atoms=True,
                allow_reflection=ALLOW_REFLECTION,
            )
            D[i, j] = D[j, i] = rmsd

    return D


# ── Comparison metrics ───────────────────────────────────────────────────────

def knn_overlap(D1: np.ndarray, D2: np.ndarray, k: int) -> float:
    """
    Mean k-nearest-neighbour set overlap between two distance matrices.

    For each point i, computes |N_k(D1, i) ∩ N_k(D2, i)| / k and returns
    the mean over all points.  Self-distances (diagonal) are excluded.
    """
    n = D1.shape[0]
    k = min(k, n - 1)
    # argsort rows, skip index 0 (self)
    nn1 = np.argsort(D1, axis=1)[:, 1: k + 1]
    nn2 = np.argsort(D2, axis=1)[:, 1: k + 1]
    overlaps = [len(set(nn1[i]) & set(nn2[i])) / k for i in range(n)]
    return float(np.mean(overlaps))


def compare_matrices(
    D_opt: np.ndarray,
    D_m: np.ndarray,
    label: str,
    n_total: int,
) -> dict:
    """
    Compare D^(m) to D_opt with global and local metrics.

    Parameters
    ----------
    D_opt   : (n, n) reference brute-force RMSD matrix
    D_m     : (n, n) alignment-strategy RMSD matrix
    label   : strategy name (e.g. "single_ref", "multi_ref")
    n_total : total number of spawns in the dataset (for reporting)
    """
    n = D_opt.shape[0]
    triu = np.triu_indices(n, k=1)
    d_opt = D_opt[triu]
    d_m   = D_m[triu]

    spearman = spearmanr(d_opt, d_m).statistic
    pearson  = pearsonr(d_opt, d_m).statistic
    frob_abs = float(np.linalg.norm(D_opt - D_m, ord="fro"))
    frob_rel = frob_abs / float(np.linalg.norm(D_opt, ord="fro"))
    mse      = float(np.mean((d_opt - d_m) ** 2))

    # k = 5 %, 10 %, 20 % of n (at least 1)
    k_values = {
        "k5pct":  max(1, int(0.05 * n)),
        "k10pct": max(1, int(0.10 * n)),
        "k20pct": max(1, int(0.20 * n)),
    }
    knn = {tag: knn_overlap(D_opt, D_m, k) for tag, k in k_values.items()}

    return {
        "strategy":        label,
        "n_sample":        n,
        "n_total":         n_total,
        "spearman":        round(spearman, 6),
        "pearson":         round(pearson,  6),
        "frobenius_abs":   round(frob_abs, 6),
        "frobenius_rel":   round(frob_rel, 6),
        "mse":             round(mse,      8),
        **{f"knn_overlap_{tag}": round(v, 6) for tag, v in knn.items()},
        **{f"k_{tag}": k_values[tag] for tag in k_values},
    }


# ── Per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(
    name: str,
    cfg: dict,
    n_sample_override: int | None = None,
    seed: int = 42,
    dopt_cache: Path | None = None,
    output_dir: Path = Path("reports"),
) -> pd.DataFrame:
    """
    Run the full evaluation for one dataset.

    Steps
    -----
    1. Sample geometries (or use all if n_sample is None).
    2. Compute D_opt via brute-force (or load from cache CSV).
    3. Compute D^(single_ref) and D^(multi_ref) from aligned geometries.
    4. Compare each to D_opt and collect metrics.
    5. Save D_opt, D^(m) matrices and metrics to output_dir.

    Returns
    -------
    DataFrame with one row per strategy.
    """
    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"{'=' * 60}")

    n_sample = n_sample_override if n_sample_override is not None else cfg["n_sample"]
    spawn_dir = cfg["spawns"]

    all_paths = sorted(spawn_dir.glob("*.xyz"))
    n_total   = len(all_paths)
    print(f"  Total spawns : {n_total}")

    # ── Sample ──────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    if n_sample is None or n_sample >= n_total:
        sample_idx = list(range(n_total))
        print(f"  Sample       : all {n_total} geometries")
    else:
        sample_idx = sorted(rng.choice(n_total, size=n_sample, replace=False).tolist())
        print(f"  Sample       : {n_sample} geometries (seed={seed})")

    sample_names = [all_paths[i].stem for i in sample_idx]

    # ── D_opt ────────────────────────────────────────────────────────────────
    dopt_path = output_dir / f"dopt_{name}.csv"

    if dopt_cache is not None and Path(dopt_cache).exists():
        print(f"  D_opt        : loading from {dopt_cache}")
        df_dopt = pd.read_csv(dopt_cache, index_col=0)
        # reorder to match sample_names
        common = [n for n in sample_names if n in df_dopt.index]
        if len(common) < len(sample_names):
            print(f"  Warning: {len(sample_names) - len(common)} sample names missing "
                  f"from D_opt cache — using {len(common)} common geometries")
            sample_names = common
        D_opt = df_dopt.loc[sample_names, sample_names].values
    else:
        print(f"  D_opt        : computing brute-force ({len(sample_idx)} × {len(sample_idx) - 1} // 2 pairs) ...")
        D_opt, sample_names = compute_D_opt(spawn_dir, sample_idx, seed=seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(D_opt, index=sample_names, columns=sample_names).to_csv(dopt_path)
        print(f"  D_opt        : saved to {dopt_path}")

    n = len(sample_names)
    print(f"  n (used)     : {n}")
    print(f"  D_opt stats  : mean={np.mean(D_opt[np.triu_indices(n,1)]):.4f} Å  "
          f"max={np.max(D_opt):.4f} Å")

    # ── D^(m) for each strategy ──────────────────────────────────────────────
    strategies = {
        "single_ref": cfg["single_ref"],
        "multi_ref":  cfg["multi_ref"],
    }

    rows = []
    for strat_name, strat_dir in strategies.items():
        if not strat_dir.exists():
            print(f"  [{strat_name}] directory not found: {strat_dir} — skipping")
            continue

        print(f"  [{strat_name}] computing D^(m) ...")
        try:
            D_m = compute_D_strategy(strat_dir, sample_names)
        except FileNotFoundError as e:
            print(f"    Error: {e}")
            continue

        dm_path = output_dir / f"d_{strat_name}_{name}.csv"
        pd.DataFrame(D_m, index=sample_names, columns=sample_names).to_csv(dm_path)
        print(f"    Saved       : {dm_path}")
        print(f"    D^(m) stats : mean={np.mean(D_m[np.triu_indices(n,1)]):.4f} Å  "
              f"max={np.max(D_m):.4f} Å")

        metrics = compare_matrices(D_opt, D_m, label=strat_name, n_total=n_total)
        metrics["dataset"] = name
        rows.append(metrics)

        print(f"    Spearman    : {metrics['spearman']:.4f}")
        print(f"    Frobenius   : {metrics['frobenius_abs']:.4f} ({metrics['frobenius_rel']:.2%} relative)")
        print(f"    k-NN overlap: k5%={metrics['knn_overlap_k5pct']:.4f}  "
              f"k10%={metrics['knn_overlap_k10pct']:.4f}  "
              f"k20%={metrics['knn_overlap_k20pct']:.4f}")

    df = pd.DataFrame(rows)
    if not df.empty:
        out_csv = output_dir / f"alignment_strategy_eval_{name}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Metrics saved: {out_csv}")

    return df


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results: list[pd.DataFrame]) -> None:
    if not all_results:
        return
    combined = pd.concat(all_results, ignore_index=True)

    cols = ["dataset", "strategy", "n_sample", "spearman", "frobenius_rel",
            "knn_overlap_k5pct", "knn_overlap_k10pct", "knn_overlap_k20pct"]
    print("\n" + "=" * 80)
    print("SUMMARY — alignment strategy evaluation")
    print("=" * 80)
    print(combined[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Winner per dataset (higher Spearman + higher mean kNN = better)
    print("\nRecommended strategy per dataset (best Spearman):")
    for ds, grp in combined.groupby("dataset"):
        best = grp.loc[grp["spearman"].idxmax(), "strategy"]
        s_vals = grp.set_index("strategy")["spearman"].to_dict()
        print(f"  {ds:<14} → {best}  "
              f"(single_ref={s_vals.get('single_ref','n/a'):.4f}, "
              f"multi_ref={s_vals.get('multi_ref','n/a'):.4f})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate single-ref vs multi-ref alignment strategy using D_opt."
    )
    parser.add_argument(
        "-d", "--dataset",
        choices=list(DATASETS.keys()),
        help="Run only this dataset (default: all)",
    )
    parser.add_argument(
        "-n", "--nsamples",
        type=int,
        default=None,
        help="Override n_sample for all datasets (default: per-dataset config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for geometry sampling (default: 42)",
    )
    parser.add_argument(
        "--load-dopt",
        type=Path,
        default=None,
        metavar="CSV",
        help="Load a pre-computed D_opt from this CSV instead of recomputing "
             "(only used with -d)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for output CSVs (default: reports/)",
    )
    args = parser.parse_args()

    active = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    all_results = []
    for name, cfg in active.items():
        df = run_dataset(
            name=name,
            cfg=cfg,
            n_sample_override=args.nsamples,
            seed=args.seed,
            dopt_cache=args.load_dopt if args.dataset else None,
            output_dir=args.output_dir,
        )
        if not df.empty:
            all_results.append(df)

    print_summary(all_results)


if __name__ == "__main__":
    main()
