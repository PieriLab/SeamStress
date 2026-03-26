import os
import glob
import argparse
import shutil
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from umap import UMAP

from ase.io import read, write
from ase import Atoms
from matplotlib.cm import get_cmap


# ---------------------------
# IO (flatten XYZ + keep files)
# ---------------------------
def load_xyz_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.xyz")))
    coords = []
    n_atoms = None

    for f in files:
        atoms = read(f)
        positions = atoms.positions

        if n_atoms is None:
            n_atoms = positions.shape[0]
        elif positions.shape[0] != n_atoms:
            raise ValueError(f"Inconsistent atom count in {f}")

        coords.append(positions.reshape(-1))

    return np.array(coords), files


# ---------------------------
# Bandwidth selection
# ---------------------------
def select_bandwidth(X, mode):
    if mode == "manual":
        return 0.90

    elif mode == "cv":
        params = {"bandwidth": np.logspace(-2, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(X)
        return grid.best_params_["bandwidth"]

    elif mode == "multimodal":
        std = np.std(X, axis=0).mean()
        n = len(X)
        return 1.06 * std * n ** (-1 / 5) * 0.5

    else:
        raise ValueError("Unknown bandwidth mode")


# ---------------------------
# KDE density
# ---------------------------
def compute_kde_density(X, bandwidth):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(X)

    log_density = kde.score_samples(X)
    density = np.exp(log_density)

    density /= density.sum()

    return density, kde


# ---------------------------
# Mean Shift derived from KDE
# ---------------------------
def mean_shift_from_kde(X, bandwidth=0.1, tol=1e-3, max_iter=300):
    N, d = X.shape
    assert d == 2

    modes = []

    for i in range(N):
        x = X[i].copy()

        for _ in range(max_iter):
            diff = X - x
            dist2 = np.sum(diff**2, axis=1)

            weights = np.exp(-dist2 / (2 * bandwidth**2))

            x_new = np.sum(weights[:, None] * X, axis=0) / np.sum(weights)

            if np.linalg.norm(x_new - x) < tol:
                break

            x = x_new

        modes.append(x)

    modes = np.array(modes)

    unique_modes = []

    for m in modes:
        if len(unique_modes) == 0:
            unique_modes.append(m)
        else:
            dists = np.linalg.norm(np.array(unique_modes) - m, axis=1)
            idx = np.argmin(dists)

            if dists[idx] >= bandwidth / 2:
                unique_modes.append(m)

    unique_modes = np.array(unique_modes)

    if len(unique_modes) == 0:
        return np.array([]), []

    assignments = []
    for x in X:
        dists = np.linalg.norm(unique_modes - x, axis=1)
        assignments.append(np.argmin(dists))
    assignments = np.array(assignments)

    valid_modes = []
    valid_indices = []

    for i in range(len(unique_modes)):
        count = np.sum(assignments == i)
        if count >= 5:
            valid_modes.append(unique_modes[i])
            valid_indices.append(i)

    valid_modes = np.array(valid_modes)

    if len(valid_modes) == 0:
        return np.array([]), []

    return valid_modes, valid_indices


# ---------------------------
# Save modes by copying original XYZ + annotating
# ---------------------------
def save_modes_as_xyz(modes_emb, X_emb, xyz_files, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    saved_info = []

    for i, mode in enumerate(modes_emb):
        dists = np.linalg.norm(X_emb - mode, axis=1)
        idx = np.argmin(dists)

        src_file = xyz_files[idx]
        src_atoms = read(src_file)

        mode_index = i + 1

        base_name = os.path.basename(src_file)
        out_name = f"mode_{mode_index:03d}__{base_name}"
        out_path = os.path.join(out_dir, out_name)

        src_atoms.info["comment"] = f"Density Mode {mode_index} (from {base_name})"

        write(out_path, src_atoms)

        saved_info.append((mode_index, out_path))

    return saved_info


# ---------------------------
# Scatter plot
# ---------------------------
def plot_results(embedding, meci_emb, modes, density_values, meci_files, out_dir):
    plt.figure(figsize=(8, 6))

    sc = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=density_values,
        s=10,
        cmap="viridis",
        alpha=0.7,
    )

    plt.colorbar(sc, label="Normalized Density")

    # MECIs (red stars)
    plt.scatter(
        meci_emb[:, 0],
        meci_emb[:, 1],
        marker="*",
        s=150,
        c="red",
        edgecolor="black",
        label="MECIs",
    )

    # Maxima (yellow stars)
    plt.scatter(
        modes[:, 0],
        modes[:, 1],
        marker="*",
        s=150,
        c="yellow",
        edgecolor="black",
        label="Maxima",
    )

    plt.legend(fontsize=9)
    plt.title("UMAP + KDE Density Landscape")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "density_scatter.png")
    plt.savefig(out_path, dpi=300)
    plt.show()


# ---------------------------
# Contour plot
# ---------------------------
def plot_density_contours(embedding, meci_emb, modes, kde, meci_files, out_dir):
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )

    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    zz = np.exp(kde.score_samples(grid)).reshape(xx.shape)

    dx = (x_max - x_min) / 200
    dy = (y_max - y_min) / 200
    zz = zz / np.sum(zz * dx * dy)

    plt.figure(figsize=(8, 6))

    plt.contourf(xx, yy, zz, levels=50, cmap="viridis")
    plt.contour(xx, yy, zz, levels=10, colors="black", linewidths=0.5)

    # MECIs (red stars)
    plt.scatter(
        meci_emb[:, 0],
        meci_emb[:, 1],
        marker="*",
        s=150,
        c="red",
        edgecolor="black",
        label="MECIs",
    )

    # Maxima (yellow stars)
    plt.scatter(
        modes[:, 0],
        modes[:, 1],
        marker="*",
        s=150,
        c="yellow",
        edgecolor="black",
        label="Maxima",
    )

    plt.legend(fontsize=9)
    plt.title("KDE Density Contour Landscape (Normalized)")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "density_contours.png")
    plt.savefig(out_path, dpi=300)
    plt.show()


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--xyz_dir", required=True)
    parser.add_argument("--meci_dir", required=True)
    parser.add_argument("--bandwidth_mode", choices=["manual", "cv", "multimodal"], default="cv")
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    X, xyz_files = load_xyz_folder(args.xyz_dir)
    X_meci, meci_files = load_xyz_folder(args.meci_dir)

    # Combine
    X_all = np.vstack([X, X_meci])

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # UMAP
    n_neighbors = min(15, X_scaled.shape[0] - 1)

    umap_model = UMAP(
        densmap=True,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        random_state=42,
    )

    embedding_all = umap_model.fit_transform(X_scaled)

    emb = embedding_all[:len(X)]
    meci_emb = embedding_all[len(X):]

    # Bandwidth
    bw = select_bandwidth(emb, args.bandwidth_mode)

    # KDE
    density_values, kde = compute_kde_density(emb, bw)

    # Mean shift modes
    modes, _ = mean_shift_from_kde(emb, bandwidth=bw)

    # Save modes
    modes_dir = os.path.join(args.out_dir, "density_modes_xyz")
    save_modes_as_xyz(modes, emb, xyz_files, modes_dir)

    # Plots
    plot_results(emb, meci_emb, modes, density_values, meci_files, args.out_dir)
    plot_density_contours(emb, meci_emb, modes, kde, meci_files, args.out_dir)


if __name__ == "__main__":
    main()