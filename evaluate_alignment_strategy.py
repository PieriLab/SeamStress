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

Output layout
-------------
reference_evaluation/
  <dataset>/
    dopt.csv
    d_single_ref.csv
    d_multi_ref.csv
    metrics.csv
    report.txt
    scatter_single_ref.png
    scatter_multi_ref.png
    knn_overlap.png
    distance_distributions.png

Usage
-----
    uv run python evaluate_alignment_strategy.py -d ethylene
    uv run python evaluate_alignment_strategy.py -d benzene_s0 -n 100
    uv run python evaluate_alignment_strategy.py          # all datasets
    uv run python evaluate_alignment_strategy.py --load-dopt reference_evaluation/ethylene/dopt.csv -d ethylene
"""

import argparse
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
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
STRATEGY_COLORS  = {"single_ref": "#2196F3", "multi_ref": "#FF5722"}
STRATEGY_LABELS  = {"single_ref": "Single-reference", "multi_ref": "Multi-reference RMSD"}


# ── Pairwise D_opt (brute-force) ─────────────────────────────────────────────

def compute_D_opt(spawn_dir: Path, sample_indices) -> tuple[np.ndarray, list[str]]:
    """
    Compute the brute-force pairwise RMSD matrix D_opt for the sampled geometries.

    Returns
    -------
    D     : (n, n) symmetric matrix of brute-force pairwise RMSDs
    names : filenames stems in row/column order
    """
    paths  = sorted(spawn_dir.glob("*.xyz"))
    geoms  = [read_xyz_file(paths[i]) for i in sample_indices]
    names  = [paths[i].stem for i in sample_indices]
    atoms  = geoms[0].atoms
    n      = len(geoms)
    D      = np.zeros((n, n))

    for i, j in tqdm(list(combinations(range(n), 2)), desc="  D_opt pairs", leave=False):
        _, rmsd, *_ = _search_bruteforce_elementwise(
            geoms[i].coordinates, geoms[j].coordinates,
            atoms, atoms, allow_reflection=ALLOW_REFLECTION,
        )
        D[i, j] = D[j, i] = rmsd

    return D, names


# ── Pairwise D^(m) from aligned geometries ───────────────────────────────────

def compute_D_strategy(aligned_dir: Path, filenames: list[str]) -> np.ndarray:
    """
    Compute pairwise Kabsch RMSD from already-aligned geometries.

    No permutation search is performed — atom ordering is already fixed by the
    alignment step. Kabsch rotation handles residual frame differences
    (important for multi_ref where each geometry was aligned to a different MECI).
    """
    geoms, missing = [], []
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
                geoms[i].coordinates, geoms[j].coordinates,
                atoms1=geoms[i].atoms, atoms2=geoms[j].atoms,
                use_all_atoms=True, allow_reflection=ALLOW_REFLECTION,
            )
            D[i, j] = D[j, i] = rmsd
    return D


# ── Comparison metrics ───────────────────────────────────────────────────────

def knn_overlap(D1: np.ndarray, D2: np.ndarray, k: int) -> float:
    """Mean k-NN set overlap: fraction of k nearest neighbours shared between D1 and D2."""
    n = D1.shape[0]
    k = min(k, n - 1)
    nn1 = np.argsort(D1, axis=1)[:, 1: k + 1]
    nn2 = np.argsort(D2, axis=1)[:, 1: k + 1]
    return float(np.mean([len(set(nn1[i]) & set(nn2[i])) / k for i in range(n)]))


def compare_matrices(D_opt: np.ndarray, D_m: np.ndarray, label: str, n_total: int) -> dict:
    """Compare D^(m) to D_opt; returns dict of metrics."""
    n    = D_opt.shape[0]
    triu = np.triu_indices(n, k=1)
    d_opt, d_m = D_opt[triu], D_m[triu]

    spearman = spearmanr(d_opt, d_m).statistic
    pearson  = pearsonr(d_opt, d_m).statistic
    frob_abs = float(np.linalg.norm(D_opt - D_m, ord="fro"))
    frob_rel = frob_abs / float(np.linalg.norm(D_opt, ord="fro"))
    mse      = float(np.mean((d_opt - d_m) ** 2))

    k_vals = {
        "k5pct":  max(1, int(0.05 * n)),
        "k10pct": max(1, int(0.10 * n)),
        "k20pct": max(1, int(0.20 * n)),
    }
    knn = {tag: knn_overlap(D_opt, D_m, k) for tag, k in k_vals.items()}

    return {
        "strategy":             label,
        "n_sample":             n,
        "n_total":              n_total,
        "spearman":             round(spearman, 6),
        "pearson":              round(pearson,  6),
        "frobenius_abs":        round(frob_abs, 6),
        "frobenius_rel":        round(frob_rel, 6),
        "mse":                  round(mse,      8),
        **{f"knn_overlap_{tag}": round(v, 6) for tag, v in knn.items()},
        **{f"k_{tag}": k_vals[tag] for tag in k_vals},
    }


# ── Plots ────────────────────────────────────────────────────────────────────

def _scatter_plot(D_opt, D_m, strat_name, out_path: Path, metrics: dict) -> None:
    """Scatter plot of upper-triangle D^(m) vs D_opt with identity line."""
    triu  = np.triu_indices(D_opt.shape[0], k=1)
    x, y  = D_opt[triu], D_m[triu]
    color = STRATEGY_COLORS[strat_name]
    label = STRATEGY_LABELS[strat_name]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=6, alpha=0.4, color=color, linewidths=0)
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="Identity")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r"$D^{\mathrm{opt}}_{ij}$ (Å)")
    ax.set_ylabel(r"$D^{(m)}_{ij}$ (Å)")
    ax.set_title(f"{label}\nSpearman={metrics['spearman']:.4f}  "
                 f"Frob={metrics['frobenius_rel']:.2%}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _knn_bar_plot(metrics_list: list[dict], out_path: Path) -> None:
    """Grouped bar chart of k-NN overlap at k=5%, 10%, 20% for each strategy."""
    k_tags  = ["k5pct", "k10pct", "k20pct"]
    k_labels = ["k = 5%", "k = 10%", "k = 20%"]
    x = np.arange(len(k_tags))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(metrics_list):
        vals   = [m[f"knn_overlap_{t}"] for t in k_tags]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=STRATEGY_LABELS[m["strategy"]],
               color=STRATEGY_COLORS[m["strategy"]], alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(k_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("k-NN overlap")
    ax.set_title("Local neighbourhood preservation vs $D^{\\mathrm{opt}}$")
    ax.legend(fontsize=9)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _distance_distribution_plot(
    D_opt: np.ndarray,
    strategy_matrices: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """KDE/histogram of upper-triangle pairwise distances for D_opt and each D^(m)."""
    triu = np.triu_indices(D_opt.shape[0], k=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(D_opt[triu], bins=40, density=True, alpha=0.5,
            color="gray", label=r"$D^{\mathrm{opt}}$", edgecolor="none")
    for strat, D_m in strategy_matrices.items():
        ax.hist(D_m[triu], bins=40, density=True, alpha=0.5,
                color=STRATEGY_COLORS[strat],
                label=STRATEGY_LABELS[strat], edgecolor="none")

    ax.set_xlabel("Pairwise RMSD (Å)")
    ax.set_ylabel("Density")
    ax.set_title("Pairwise distance distributions")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Text report ──────────────────────────────────────────────────────────────

def _write_report(
    name: str,
    n_total: int,
    n_sample: int,
    seed: int,
    rows: list[dict],
    out_path: Path,
) -> None:
    lines = [
        f"Alignment Strategy Evaluation — {name}",
        "=" * 60,
        f"Run at      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset     : {name}",
        f"Total spawns: {n_total}",
        f"Sample      : {n_sample}  (seed={seed})",
        "",
        "D_opt computed by brute-force permutation + Kabsch for every pair.",
        "D^(m) computed by Kabsch-only RMSD on pre-aligned geometries.",
        "",
    ]

    for m in rows:
        strat = m["strategy"]
        lines += [
            f"Strategy: {STRATEGY_LABELS.get(strat, strat)}",
            "-" * 40,
            f"  Spearman corr      : {m['spearman']:.4f}",
            f"  Pearson  corr      : {m['pearson']:.4f}",
            f"  Frobenius norm     : {m['frobenius_abs']:.4f} Å  "
            f"({m['frobenius_rel']:.2%} relative)",
            f"  MSE                : {m['mse']:.6f} Å²",
            f"  k-NN overlap k=5%  : {m['knn_overlap_k5pct']:.4f}  "
            f"(k={m['k_k5pct']})",
            f"  k-NN overlap k=10% : {m['knn_overlap_k10pct']:.4f}  "
            f"(k={m['k_k10pct']})",
            f"  k-NN overlap k=20% : {m['knn_overlap_k20pct']:.4f}  "
            f"(k={m['k_k20pct']})",
            "",
        ]

    # Recommendation
    if len(rows) == 2:
        s_vals = {m["strategy"]: m for m in rows}
        sr, mr = s_vals.get("single_ref"), s_vals.get("multi_ref")
        if sr and mr:
            winner_spearman = "single_ref" if sr["spearman"] >= mr["spearman"] else "multi_ref"
            winner_knn      = ("single_ref"
                               if sr["knn_overlap_k10pct"] >= mr["knn_overlap_k10pct"]
                               else "multi_ref")
            lines += [
                "Recommendation",
                "-" * 40,
                f"  Best Spearman      : {STRATEGY_LABELS[winner_spearman]}",
                f"  Best k-NN (k=10%)  : {STRATEGY_LABELS[winner_knn]}",
            ]
            if winner_spearman == winner_knn:
                lines.append(
                    f"  → Both metrics agree: use {STRATEGY_LABELS[winner_spearman]}"
                )
            else:
                lines.append(
                    "  → Metrics disagree: prioritise local structure "
                    f"→ use {STRATEGY_LABELS[winner_knn]}"
                )

    out_path.write_text("\n".join(lines) + "\n")


# ── Per-dataset runner ────────────────────────────────────────────────────────

def run_dataset(
    name: str,
    cfg: dict,
    n_sample_override: int | None = None,
    seed: int = 42,
    dopt_cache: Path | None = None,
    base_output_dir: Path = Path("reference_evaluation"),
) -> pd.DataFrame:
    """Run the full evaluation for one dataset and write all output files."""

    out_dir = base_output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}  →  {out_dir}")
    print(f"{'=' * 60}")

    n_sample  = n_sample_override if n_sample_override is not None else cfg["n_sample"]
    spawn_dir = cfg["spawns"]

    all_paths = sorted(spawn_dir.glob("*.xyz"))
    n_total   = len(all_paths)
    print(f"  Total spawns : {n_total}")

    # ── Sample ──────────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    if n_sample is None or n_sample >= n_total:
        sample_idx  = list(range(n_total))
        n_used      = n_total
        print(f"  Sample       : all {n_total} geometries")
    else:
        sample_idx  = sorted(rng.choice(n_total, size=n_sample, replace=False).tolist())
        n_used      = n_sample
        print(f"  Sample       : {n_sample} geometries (seed={seed})")

    sample_names = [all_paths[i].stem for i in sample_idx]

    # ── D_opt ────────────────────────────────────────────────────────────────
    dopt_path = out_dir / "dopt.csv"
    cache_src = dopt_cache if (dopt_cache and Path(dopt_cache).exists()) else (
        dopt_path if dopt_path.exists() else None
    )

    if cache_src:
        print(f"  D_opt        : loading from {cache_src}")
        df_dopt = pd.read_csv(cache_src, index_col=0)
        common  = [n for n in sample_names if n in df_dopt.index]
        if len(common) < len(sample_names):
            print(f"  Warning: {len(sample_names)-len(common)} names missing from cache "
                  f"— using {len(common)} common geometries")
            sample_names = common
        D_opt = df_dopt.loc[sample_names, sample_names].values
    else:
        n_pairs = len(sample_idx) * (len(sample_idx) - 1) // 2
        print(f"  D_opt        : computing brute-force ({n_pairs} pairs) ...")
        D_opt, sample_names = compute_D_opt(spawn_dir, sample_idx)
        pd.DataFrame(D_opt, index=sample_names, columns=sample_names).to_csv(dopt_path)
        print(f"  D_opt        : saved → {dopt_path}")

    n = len(sample_names)
    triu = np.triu_indices(n, k=1)
    print(f"  n (used)     : {n}")
    print(f"  D_opt        : mean={D_opt[triu].mean():.4f} Å  max={D_opt.max():.4f} Å")

    # ── D^(m) and metrics ────────────────────────────────────────────────────
    strategies = {"single_ref": cfg["single_ref"], "multi_ref": cfg["multi_ref"]}
    rows, D_matrices = [], {}

    for strat_name, strat_dir in strategies.items():
        if not strat_dir.exists():
            print(f"  [{strat_name}] not found: {strat_dir} — skipping")
            continue

        print(f"  [{strat_name}] computing D^(m) ...")
        try:
            D_m = compute_D_strategy(strat_dir, sample_names)
        except FileNotFoundError as e:
            print(f"    Error: {e}")
            continue

        pd.DataFrame(D_m, index=sample_names, columns=sample_names).to_csv(
            out_dir / f"d_{strat_name}.csv"
        )
        D_matrices[strat_name] = D_m

        metrics = compare_matrices(D_opt, D_m, label=strat_name, n_total=n_total)
        metrics["dataset"] = name
        rows.append(metrics)

        print(f"    Spearman    : {metrics['spearman']:.4f}")
        print(f"    Frobenius   : {metrics['frobenius_abs']:.4f} "
              f"({metrics['frobenius_rel']:.2%} relative)")
        print(f"    k-NN overlap: k5%={metrics['knn_overlap_k5pct']:.4f}  "
              f"k10%={metrics['knn_overlap_k10pct']:.4f}  "
              f"k20%={metrics['knn_overlap_k20pct']:.4f}")

        # scatter plot
        _scatter_plot(
            D_opt, D_m, strat_name,
            out_dir / f"scatter_{strat_name}.png",
            metrics,
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)

    # shared plots
    if D_matrices:
        _knn_bar_plot(rows, out_dir / "knn_overlap.png")
        _distance_distribution_plot(D_opt, D_matrices, out_dir / "distance_distributions.png")

    # text report
    _write_report(name, n_total, n, seed, rows, out_dir / "report.txt")

    print(f"  Output       : {out_dir}/")
    return df


# ── Summary (cross-dataset) ──────────────────────────────────────────────────

def write_summary(all_results: list[pd.DataFrame], base_dir: Path) -> None:
    if not all_results:
        return
    combined = pd.concat(all_results, ignore_index=True)

    cols = ["dataset", "strategy", "n_sample", "spearman", "frobenius_rel",
            "knn_overlap_k5pct", "knn_overlap_k10pct", "knn_overlap_k20pct"]

    # stdout
    print("\n" + "=" * 80)
    print("SUMMARY — alignment strategy evaluation")
    print("=" * 80)
    print(combined[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # winner per dataset
    print("\nRecommended strategy per dataset (local k-NN k=10%, tiebreak: Spearman):")
    winner_lines = []
    for ds, grp in combined.groupby("dataset"):
        best_knn = grp.loc[grp["knn_overlap_k10pct"].idxmax(), "strategy"]
        best_sp  = grp.loc[grp["spearman"].idxmax(), "strategy"]
        winner   = best_knn  # priority to local structure per paper
        line = (f"  {ds:<14} → {STRATEGY_LABELS.get(winner, winner):<28}  "
                f"(kNN={grp.set_index('strategy')['knn_overlap_k10pct'].to_dict()}  "
                f"Spearman={grp.set_index('strategy')['spearman'].to_dict()})")
        print(line)
        winner_lines.append(line)

    # summary txt
    summary_path = base_dir / "summary.txt"
    lines = [
        "Alignment Strategy Evaluation — Cross-Dataset Summary",
        "=" * 70,
        f"Run at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        combined[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "Recommended strategy per dataset (local k-NN k=10%, tiebreak: Spearman):",
        *winner_lines,
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary saved → {summary_path}")

    # summary CSV
    combined.to_csv(base_dir / "summary.csv", index=False)


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
        type=int, default=None,
        help="Override n_sample for all datasets (default: per-dataset config)",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed for geometry sampling (default: 42)",
    )
    parser.add_argument(
        "--load-dopt",
        type=Path, default=None, metavar="CSV",
        help="Load a pre-computed D_opt CSV instead of recomputing (requires -d)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path, default=Path("reference_evaluation"),
        help="Base output directory (default: reference_evaluation/)",
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
            base_output_dir=args.output_dir,
        )
        if not df.empty:
            all_results.append(df)

    write_summary(all_results, args.output_dir)


if __name__ == "__main__":
    main()
