import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

datasets = {
    "ethylene":     r"Ethylene",
    "butadiene_s0": r"Butadiene S$_0$",
    "butadiene_s1": r"Butadiene S$_1$",
    "benzene_s0":   r"Benzene S$_0$",
    "benzene_s1":   r"Benzene S$_1$",
}

# ── Table 1 (main): three-way consistency ────────────────────────────────────
print("% Table 1 (main text): three-way consistency\n")
print(r"\begin{tabular}{lcccc}")
print(r"\hline")
print(r"Dataset & CSV $=$ Align.\ (\%) & CSV in basin (\%) & Align.\ in basin (\%) & All agree (\%) \\")
print(r"\hline")
for ds, label in datasets.items():
    df = pd.read_csv(f"seam_basin/{ds}/correspondence.csv", index_col=0)
    a = 100 * df["csv_eq_align"].mean()
    b = 100 * df["csv_in_basin"].mean()
    c = 100 * df["align_in_basin"].mean()
    d = 100 * df["all_agree"].mean()
    print(f"{label} & {a:.1f} & {b:.1f} & {c:.1f} & {d:.1f} \\\\")
print(r"\hline")
print(r"\end{tabular}")

# ── Table 2 (main): d_m summary ──────────────────────────────────────────────
print("\n\n% Table 2 (main text): d_m summary\n")
print(r"\begin{tabular}{lcccc}")
print(r"\hline")
print(r"Dataset & min $d_m$ & median $d_m$ & max $d_m$ & $\langle N_k \rangle_{\mathrm{MECI}}$ \\")
print(r"\hline")
for ds, label in datasets.items():
    df = pd.read_csv(f"seam_basin/{ds}/meci_metrics_grid_3d.csv")
    dm = df["d_m"]
    nk = df["basin_N_k"].mean()
    print(f"{label} & {dm.min():.3f} & {dm.median():.3f} & {dm.max():.3f} & {nk:.1f} \\\\")
print(r"\hline")
print(r"\end{tabular}")

# ── Table SI: pairwise ARI/NMI ───────────────────────────────────────────────
print("\n\n% Table SI: pairwise ARI/NMI\n")
print(r"\begin{tabular}{lcccccc}")
print(r"\hline")
print(r"Dataset & \multicolumn{2}{c}{CSV $\leftrightarrow$ Alignment} & \multicolumn{2}{c}{CSV $\leftrightarrow$ Basin} & \multicolumn{2}{c}{Alignment $\leftrightarrow$ Basin} \\")
print(r" & ARI & NMI & ARI & NMI & ARI & NMI \\")
print(r"\hline")
for ds, label in datasets.items():
    df = pd.read_csv(f"seam_basin/{ds}/correspondence.csv", index_col=0)
    ari_ca = adjusted_rand_score(df["csv_meci"], df["align_meci"])
    nmi_ca = normalized_mutual_info_score(df["csv_meci"], df["align_meci"])
    ari_cb = adjusted_rand_score(df["csv_meci"], df["basin_id"])
    nmi_cb = normalized_mutual_info_score(df["csv_meci"], df["basin_id"].astype(str))
    ari_ab = adjusted_rand_score(df["align_meci"], df["basin_id"])
    nmi_ab = normalized_mutual_info_score(df["align_meci"], df["basin_id"].astype(str))
    print(f"{label} & {ari_ca:.3f} & {nmi_ca:.3f} & {ari_cb:.3f} & {nmi_cb:.3f} & {ari_ab:.3f} & {nmi_ab:.3f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
