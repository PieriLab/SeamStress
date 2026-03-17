# evaluate_alignment_strategy_mp_full.py

import argparse
from datetime import datetime
from itertools import combinations
from pathlib import Path
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from seamstress.alignment import _search_bruteforce_elementwise, weighted_rmsd
from seamstress.geometry import read_xyz_file


# ── Dataset configuration ────────────────────────────────────────────────────

DATASETS = {
    "benzene_s0": {
        "spawns":     Path("data/benzene_s0/spawns/"),
        "single_ref": Path("data/benzene_s0/aligned_spawns/single_ref"),
        "multi_ref":  Path("data/benzene_s0/aligned_spawns/multi_ref"),
        "n_sample":   100,
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
        "n_sample":   None,
    },
    "butadiene_s0": {
        "spawns":     Path("data/butadiene_s0/spawns"),
        "single_ref": Path("data/butadiene_s0/aligned_spawns/single_ref"),
        "multi_ref":  Path("data/butadiene_s0/aligned_spawns/multi_ref"),
        "n_sample":   200,
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


# ── Multiprocessing worker ───────────────────────────────────────────────────

def _dopt_worker(args):
    i, j, coords, atoms = args
    _, rmsd, *_ = _search_bruteforce_elementwise(
        coords[i], coords[j],
        atoms, atoms,
        allow_reflection=ALLOW_REFLECTION,
    )
    return i, j, rmsd


# ── Parallel D_opt ───────────────────────────────────────────────────────────

def compute_D_opt(spawn_dir, sample_indices, n_proc=None):
    paths  = sorted(spawn_dir.glob("*.xyz"))
    geoms  = [read_xyz_file(paths[i]) for i in sample_indices]
    names  = [paths[i].stem for i in sample_indices]

    coords = [g.coordinates for g in geoms]
    atoms  = geoms[0].atoms
    n      = len(geoms)
    D      = np.zeros((n, n))

    pairs = list(combinations(range(n), 2))

    if n_proc is None:
        n_proc = max(1, mp.cpu_count() - 1)

    print(f"  Using {n_proc} processes for D_opt")

    with mp.Pool(processes=n_proc) as pool:
        results = pool.imap_unordered(
            _dopt_worker,
            ((i, j, coords, atoms) for i, j in pairs),
            chunksize=50,
        )

        for i, j, rmsd in tqdm(results, total=len(pairs), desc="  D_opt pairs", leave=False):
            D[i, j] = D[j, i] = rmsd

    return D, names


# ── Serial D^(m) ─────────────────────────────────────────────────────────────

def compute_D_strategy(aligned_dir, filenames):
    geoms, missing = [], []

    for stem in filenames:
        p = aligned_dir / f"{stem}.xyz"
        if not p.exists():
            missing.append(stem)
        else:
            geoms.append(read_xyz_file(p))

    if missing:
        raise FileNotFoundError(f"{len(missing)} missing geometries")

    n = len(geoms)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            rmsd = weighted_rmsd(
                geoms[i].coordinates,
                geoms[j].coordinates,
                atoms1=geoms[i].atoms,
                atoms2=geoms[j].atoms,
                use_all_atoms=True,
                weight_type="mass",
                heavy_atom_factor=1.0,
            )
            D[i, j] = D[j, i] = rmsd

    return D


# ── Metrics ──────────────────────────────────────────────────────────────────

def knn_overlap(D1, D2, k):
    n = D1.shape[0]
    k = min(k, n - 1)
    nn1 = np.argsort(D1, axis=1)[:, 1:k+1]
    nn2 = np.argsort(D2, axis=1)[:, 1:k+1]
    return float(np.mean([len(set(nn1[i]) & set(nn2[i])) / k for i in range(n)]))


def compare_matrices(D_opt, D_m, label, n_total):
    n = D_opt.shape[0]
    triu = np.triu_indices(n, k=1)
    d_opt, d_m = D_opt[triu], D_m[triu]

    spearman = spearmanr(d_opt, d_m).statistic
    pearson  = pearsonr(d_opt, d_m).statistic
    frob_abs = float(np.linalg.norm(D_opt - D_m))
    frob_rel = frob_abs / float(np.linalg.norm(D_opt))
    mse      = float(np.mean((d_opt - d_m) ** 2))

    k_vals = {
        "k5pct":  max(1, int(0.05 * n)),
        "k10pct": max(1, int(0.10 * n)),
        "k20pct": max(1, int(0.20 * n)),
    }
    knn = {tag: knn_overlap(D_opt, D_m, k) for tag, k in k_vals.items()}

    return {
        "strategy": label,
        "n_sample": n,
        "n_total": n_total,
        "spearman": round(spearman, 6),
        "pearson":  round(pearson, 6),
        "frobenius_abs": frob_abs,
        "frobenius_rel": frob_rel,
        "mse": mse,
        **{f"knn_overlap_{k}": v for k, v in knn.items()},
    }


# ── Plots ────────────────────────────────────────────────────────────────────

def scatter_plot(D_opt, D_m, strat, out):
    triu = np.triu_indices(D_opt.shape[0], k=1)
    x, y = D_opt[triu], D_m[triu]

    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(x, y, s=5, alpha=0.4, color=STRATEGY_COLORS[strat])
    lim = max(x.max(), y.max())
    ax.plot([0,lim],[0,lim],'k--')
    ax.set_title(strat)
    fig.savefig(out, dpi=150)
    plt.close()


def knn_bar(metrics_list, out):
    tags = ["k5pct","k10pct","k20pct"]
    x = np.arange(len(tags))

    fig, ax = plt.subplots()
    for i,m in enumerate(metrics_list):
        vals = [m[f"knn_overlap_{t}"] for t in tags]
        ax.bar(x + i*0.3, vals, 0.3, label=m["strategy"])

    ax.legend()
    fig.savefig(out, dpi=150)
    plt.close()


# ── Report ───────────────────────────────────────────────────────────────────

def write_report(name, rows, out):
    lines = [
        f"Dataset: {name}",
        "="*40,
    ]
    for r in rows:
        lines.append(str(r))
    out.write_text("\n".join(lines))


# ── Runner ───────────────────────────────────────────────────────────────────

def run_dataset(name, cfg, n_proc=None, seed=42, base_out=Path("reference_evaluation")):
    print(f"\n{'='*60}\n{name}\n{'='*60}")

    spawn_dir = cfg["spawns"]
    all_paths = sorted(spawn_dir.glob("*.xyz"))
    n_total = len(all_paths)

    rng = np.random.default_rng(seed)
    if cfg["n_sample"] and cfg["n_sample"] < n_total:
        idx = sorted(rng.choice(n_total, size=cfg["n_sample"], replace=False))
    else:
        idx = list(range(n_total))

    names = [all_paths[i].stem for i in idx]

    # D_opt
    D_opt, names = compute_D_opt(spawn_dir, idx, n_proc=n_proc)

    out_dir = base_out / name
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(D_opt, index=names, columns=names).to_csv(out_dir/"dopt.csv")

    rows = []
    Dms = {}

    for strat, path in {"single_ref":cfg["single_ref"], "multi_ref":cfg["multi_ref"]}.items():
        Dm = compute_D_strategy(path, names)
        pd.DataFrame(Dm, index=names, columns=names).to_csv(out_dir/f"d_{strat}.csv")

        metrics = compare_matrices(D_opt, Dm, strat, n_total)
        rows.append(metrics)
        Dms[strat] = Dm

        scatter_plot(D_opt, Dm, strat, out_dir/f"scatter_{strat}.png")

    pd.DataFrame(rows).to_csv(out_dir/"metrics.csv", index=False)

    knn_bar(rows, out_dir/"knn_overlap.png")
    write_report(name, rows, out_dir/"report.txt")

    print(f"Saved → {out_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", choices=list(DATASETS.keys()))
    parser.add_argument("--nproc", type=int, default=None)

    args = parser.parse_args()

    run_dataset(args.dataset, DATASETS[args.dataset], n_proc=args.nproc)


if __name__ == "__main__":
    main()