"""
Three-way correspondence analysis between:
  1. CSV assignment   : which MECI each spawn optimized to (from data/*.csv)
  2. Alignment assignment : closest MECI by geometry (from multi_ref spawn headers)
  3. Basin assignment : nearest basin centroid in 3D densMAP embedding

Outputs per-dataset CSV and summary table to seam_basin/<dataset>/correspondence.csv
and prints a LaTeX-ready summary table.
"""

import numpy as np
import pandas as pd
import glob, re
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ── Name normalisation maps ───────────────────────────────────────────────────

RENAME = {
    "ethylene": {
        "ch3-ch_dissociation": "ch_dissociation",
        "ethylidene_straight": "ethylidene_straight",   # keep — just strip
        "ethylidene_bent ":    "ethylidene_bent",
    },
}

def normalise(name, dataset):
    """Strip whitespace, lowercase, apply dataset-specific renames."""
    n = str(name).strip().lower()
    return RENAME.get(dataset, {}).get(n, n)


# ── Dataset config ────────────────────────────────────────────────────────────

DATASETS = {
    "ethylene": {
        "csv":        "data/ethylene.csv",
        "spawn_dir":  "data/ethylene/aligned_spawns/multi_ref",
        "emb_dir":    "seam_basin/ethylene",
        "csv_meci_col": "meci_type",
    },
    "butadiene_s0": {
        "csv":        "data/butadiene_s0.csv",
        "spawn_dir":  "data/butadiene_s0/aligned_spawns/multi_ref",
        "emb_dir":    "seam_basin/butadiene_s0",
        "csv_meci_col": "meci_type",
    },
    "butadiene_s1": {
        "csv":        "data/butadiene_s1.csv",
        "spawn_dir":  "data/butadiene_s1/aligned_spawns/multi_ref",
        "emb_dir":    "seam_basin/butadiene_s1",
        "csv_meci_col": "meci_type",
    },
    "benzene_s0": {
        "csv":        "data/benzene_s0.csv",
        "spawn_dir":  "data/benzene_s0/aligned_spawns/multi_ref",
        "emb_dir":    "seam_basin/benzene_s0",
        "csv_meci_col": "meci_type",
    },
    "benzene_s1": {
        "csv":        "data/benzene_s1.csv",
        "spawn_dir":  "data/benzene_s1/aligned_spawns/multi_ref",
        "emb_dir":    "seam_basin/benzene_s1",
        "csv_meci_col": "meci_type",
    },
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_csv(path, meci_col, dataset):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["spawn_id"]  = df["IC_Spawn"].str.strip()
    df["csv_meci"]  = df[meci_col].apply(lambda x: normalise(x, dataset))
    return df.set_index("spawn_id")[["csv_meci"]]


def load_alignment(spawn_dir, dataset):
    """Read closest_meci from multi_ref spawn headers."""
    records = {}
    for fpath in sorted(glob.glob(str(Path(spawn_dir) / "*.xyz"))):
        spawn_id = Path(fpath).stem
        with open(fpath) as f:
            lines = f.readlines()
        m = re.search(r"closest_meci=(\S+?)(?:\s|$)", lines[1])
        tag = m.group(1).replace(".xyz", "") if m else "unknown"
        records[spawn_id] = normalise(tag, dataset)
    return pd.DataFrame.from_dict(records, orient="index", columns=["align_meci"])


def load_basin(emb_dir, dataset):
    """Assign each spawn to nearest basin centroid in 3D embedding."""
    out = Path(emb_dir)
    emb    = np.load(out / "spawn_embedding_3d.npy")
    basins = pd.read_csv(out / "basin_summary_grid_3d.csv")
    centers = basins[["centroid_x", "centroid_y", "centroid_z"]].values
    dists   = np.linalg.norm(emb[:, None, :] - centers[None, :, :], axis=2)
    basin_labels = np.argmin(dists, axis=1)

    # Map basin_id → set of MECIs assigned to it (normalised)
    metrics = pd.read_csv(out / "meci_metrics_grid_3d.csv")
    basin_mecis = (
        metrics.groupby("nearest_basin")["meci"]
        .apply(lambda s: set(s.apply(lambda x: normalise(x, dataset))))
        .to_dict()
    )

    # Spawn filenames (sorted, same order as embedding)
    spawn_files = sorted(glob.glob(
        str(Path(DATASETS[dataset]["spawn_dir"]) / "*.xyz")
    ))
    spawn_ids = [Path(f).stem for f in spawn_files]

    df = pd.DataFrame({
        "spawn_id":  spawn_ids,
        "basin_id":  basin_labels,
    }).set_index("spawn_id")
    df["basin_mecis"] = df["basin_id"].map(basin_mecis)
    return df


# ── Per-dataset analysis ──────────────────────────────────────────────────────

def analyse(dataset, cfg):
    print(f"\n{'='*60}")
    print(f"  {dataset}")
    print(f"{'='*60}")

    csv_df   = load_csv(cfg["csv"], cfg["csv_meci_col"], dataset)
    aln_df   = load_alignment(cfg["spawn_dir"], dataset)
    bas_df   = load_basin(cfg["emb_dir"], dataset)

    # Merge on spawn_id — inner join so benzene partial CSV is handled
    df = csv_df.join(aln_df, how="inner").join(bas_df, how="inner")
    print(f"  Spawns in all three sources: {len(df)}")

    # ── Pairwise agreements ───────────────────────────────────────────────────
    pairs = [
        ("csv_meci",   "align_meci", "CSV ↔ Alignment"),
        ("csv_meci",   "basin_id",   "CSV ↔ Basin"),
        ("align_meci", "basin_id",   "Alignment ↔ Basin"),
    ]
    print("\n  Pairwise agreement:")
    results = {}
    for a, b, label in pairs:
        ari = adjusted_rand_score(df[a], df[b])
        nmi = normalized_mutual_info_score(df[a].astype(str), df[b].astype(str))
        print(f"    {label:30s}  ARI={ari:.3f}  NMI={nmi:.3f}")
        results[label] = (ari, nmi)

    # ── Three-way exact agreement ─────────────────────────────────────────────
    # A spawn is "fully consistent" if csv_meci == align_meci AND
    # csv_meci is among the MECIs assigned to its basin
    df["csv_eq_align"]   = df["csv_meci"] == df["align_meci"]
    df["csv_in_basin"]   = df.apply(
        lambda r: r["csv_meci"] in (r["basin_mecis"] if isinstance(r["basin_mecis"], set) else set()), axis=1
    )
    df["align_in_basin"] = df.apply(
        lambda r: r["align_meci"] in (r["basin_mecis"] if isinstance(r["basin_mecis"], set) else set()), axis=1
    )
    df["all_agree"]      = df["csv_eq_align"] & df["csv_in_basin"]

    n = len(df)
    print(f"\n  Three-way consistency:")
    print(f"    CSV == Alignment             : {df['csv_eq_align'].sum():4d} / {n}  ({100*df['csv_eq_align'].mean():.1f}%)")
    print(f"    CSV MECI in basin            : {df['csv_in_basin'].sum():4d} / {n}  ({100*df['csv_in_basin'].mean():.1f}%)")
    print(f"    Alignment MECI in basin      : {df['align_in_basin'].sum():4d} / {n}  ({100*df['align_in_basin'].mean():.1f}%)")
    print(f"    All three agree              : {df['all_agree'].sum():4d} / {n}  ({100*df['all_agree'].mean():.1f}%)")

    # ── Save per-spawn table ──────────────────────────────────────────────────
    out_path = Path(cfg["emb_dir"]) / "correspondence.csv"
    df[["csv_meci", "align_meci", "basin_id",
        "csv_eq_align", "csv_in_basin", "align_in_basin", "all_agree"]].to_csv(out_path)
    print(f"\n  Saved → {out_path}")

    # ── Per-MECI summary table ────────────────────────────────────────────────
    # For each MECI: N spawns optimized to it, N spawns aligned to it,
    # majority basin for each assignment, and agreement rate
    all_mecis = sorted(set(df["csv_meci"]) | set(df["align_meci"]))
    summary_rows = []
    for meci in all_mecis:
        csv_mask = df["csv_meci"]   == meci
        aln_mask = df["align_meci"] == meci
        n_csv = csv_mask.sum()
        n_aln = aln_mask.sum()
        # Majority basin for spawns that optimized to this MECI
        if n_csv > 0:
            maj_basin_csv = df.loc[csv_mask, "basin_id"].value_counts().idxmax()
            pct_maj_csv   = 100 * df.loc[csv_mask, "basin_id"].value_counts().iloc[0] / n_csv
        else:
            maj_basin_csv, pct_maj_csv = "-", float("nan")
        # Majority basin for spawns aligned to this MECI
        if n_aln > 0:
            maj_basin_aln = df.loc[aln_mask, "basin_id"].value_counts().idxmax()
            pct_maj_aln   = 100 * df.loc[aln_mask, "basin_id"].value_counts().iloc[0] / n_aln
        else:
            maj_basin_aln, pct_maj_aln = "-", float("nan")
        # Agreement: of spawns aligned to this MECI, fraction that also optimized to it
        if n_aln > 0:
            agree_rate = 100 * (df.loc[aln_mask, "csv_meci"] == meci).mean()
        else:
            agree_rate = float("nan")
        summary_rows.append({
            "meci":           meci,
            "n_csv":          n_csv,
            "n_align":        n_aln,
            "basin_csv":      maj_basin_csv,
            "pct_basin_csv":  pct_maj_csv,
            "basin_align":    maj_basin_aln,
            "pct_basin_align": pct_maj_aln,
            "csv_eq_align_pct": agree_rate,
        })
    meci_summary = pd.DataFrame(summary_rows).set_index("meci")
    meci_out = Path(cfg["emb_dir"]) / "correspondence_per_meci.csv"
    meci_summary.to_csv(meci_out)
    print(f"  Saved → {meci_out}")
    print(f"\n  Per-MECI summary:")
    print(meci_summary.to_string(float_format=lambda x: f"{x:.1f}"))

    return {
        "dataset": dataset,
        "n": n,
        "ari_csv_aln":   results["CSV ↔ Alignment"][0],
        "nmi_csv_aln":   results["CSV ↔ Alignment"][1],
        "ari_csv_bas":   results["CSV ↔ Basin"][0],
        "nmi_csv_bas":   results["CSV ↔ Basin"][1],
        "ari_aln_bas":   results["Alignment ↔ Basin"][0],
        "nmi_aln_bas":   results["Alignment ↔ Basin"][1],
        "pct_csv_eq_aln":    100 * df["csv_eq_align"].mean(),
        "pct_csv_in_basin":  100 * df["csv_in_basin"].mean(),
        "pct_aln_in_basin":  100 * df["align_in_basin"].mean(),
        "pct_all_agree":     100 * df["all_agree"].mean(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rows = []
    for dataset, cfg in DATASETS.items():
        rows.append(analyse(dataset, cfg))

    summary = pd.DataFrame(rows).set_index("dataset")

    print("\n\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(summary.to_string())

    # ── LaTeX table ───────────────────────────────────────────────────────────
    labels = {
        "ethylene":     "Ethylene",
        "butadiene_s0": "Butadiene S$_0$",
        "butadiene_s1": "Butadiene S$_1$",
        "benzene_s0":   "Benzene S$_0$",
        "benzene_s1":   "Benzene S$_1$",
    }

    print("\n\n% LaTeX table — pairwise ARI")
    print(r"\begin{tabular}{lccccccc}")
    print(r"\hline")
    print(r"Dataset & \multicolumn{2}{c}{CSV $\leftrightarrow$ Alignment} & \multicolumn{2}{c}{CSV $\leftrightarrow$ Basin} & \multicolumn{2}{c}{Alignment $\leftrightarrow$ Basin} \\")
    print(r" & ARI & NMI & ARI & NMI & ARI & NMI \\")
    print(r"\hline")
    for ds, row in summary.iterrows():
        print(f"{labels[ds]} & {row.ari_csv_aln:.3f} & {row.nmi_csv_aln:.3f} & "
              f"{row.ari_csv_bas:.3f} & {row.nmi_csv_bas:.3f} & "
              f"{row.ari_aln_bas:.3f} & {row.nmi_aln_bas:.3f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")

    print("\n\n% LaTeX table — three-way consistency")
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Dataset & CSV $=$ Align. (\%) & CSV in basin (\%) & Align. in basin (\%) & All agree (\%) \\")
    print(r"\hline")
    for ds, row in summary.iterrows():
        print(f"{labels[ds]} & {row.pct_csv_eq_aln:.1f} & {row.pct_csv_in_basin:.1f} & "
              f"{row.pct_aln_in_basin:.1f} & {row.pct_all_agree:.1f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
