import os
import argparse
import time
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap, trustworthiness
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from umap import UMAP


# ========================
# Feature construction
# ========================

def coords_to_flat_cartesian(coords: np.ndarray) -> np.ndarray:
    return coords.reshape(coords.shape[0], -1)


# ========================
# Dimensionality reduction
# ========================

def run_pca(features, scaler, n_comp=2):
    scaled = scaler.transform(features)
    pca = PCA(n_components=min(n_comp, features.shape[1], features.shape[0]))
    return pca.fit_transform(scaled), pca


def run_tsne(features, scaler, n_comp=2):
    scaled = scaler.transform(features)
    perp = min(30.0, (features.shape[0] - 1) / 3)
    return TSNE(n_components=n_comp, perplexity=perp,
                random_state=42, max_iter=1000).fit_transform(scaled)


def run_umap(features, scaler, n_comp=2):
    scaled = scaler.transform(features)
    n_neighbors = min(15, features.shape[0] - 1)
    return UMAP(n_neighbors=n_neighbors, min_dist=0.1,
                n_components=n_comp, random_state=42).fit_transform(scaled)


def run_densmap(features, scaler, n_comp=2):
    scaled = scaler.transform(features)
    n_neighbors = min(15, features.shape[0] - 1)
    return UMAP(densmap=True, n_neighbors=n_neighbors,
                min_dist=0.1, n_components=n_comp,
                random_state=42).fit_transform(scaled)


def run_diffusion_map(features, scaler, n_comp=2):
    scaled = scaler.transform(features)
    n_neighbors = min(10, features.shape[0] - 1)
    emb = SpectralEmbedding(n_components=n_comp + 1,
                            affinity='nearest_neighbors',
                            n_neighbors=n_neighbors,
                            random_state=42)
    return emb.fit_transform(scaled)[:, 1:]


def run_isomap(features, scaler, n_comp=2):
    scaled = scaler.transform(features)
    n_neighbors = min(10, features.shape[0] - 1)
    return Isomap(n_components=n_comp,
                  n_neighbors=n_neighbors).fit_transform(scaled)


def run_laplacian_eigenmaps(features, scaler, n_comp=2):
    scaled = scaler.transform(features)
    n_neighbors = min(10, features.shape[0] - 1)
    emb = SpectralEmbedding(n_components=n_comp,
                            affinity='nearest_neighbors',
                            n_neighbors=n_neighbors,
                            random_state=42)
    return emb.fit_transform(scaled)


# ========================
# Metrics
# ========================

def compute_distance_matrix(X):
    return squareform(pdist(X))


def compute_stress(D_low, D_high):
    triu_idx = np.triu_indices_from(D_low, k=1)
    diff = D_low[triu_idx] - D_high[triu_idx]
    return np.sqrt(np.sum(diff**2) / np.sum(D_high[triu_idx]**2))


import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma
from math import pi
from scipy.stats import pearsonr, spearmanr

def knn_density(X, k=5):
    """
    Compute classic kNN density estimates for a dataset X.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input points.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    densities : ndarray of shape (n_samples,)
        Estimated density for each point.
    """
    n, d = X.shape

    # Fit nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)  # include self
    distances, _ = nbrs.kneighbors(X)
    
    # k-th neighbor distance (skip the first one, which is zero)
    R_k = distances[:, k]

    # Volume of unit d-dimensional ball
    V_d = (pi ** (d / 2)) / gamma(d / 2 + 1)

    # Classic kNN density estimator
    densities = k / (n * V_d * (R_k ** d))

    return densities


def compute_density_preservation(X_high, X_low, k=10):
    """
    Correlate local densities between high-D and low-D embeddings.
    
    Parameters
    ----------
    X_high : ndarray
        Original high-dimensional points.
    X_low : ndarray
        Low-dimensional embedding points.
    k : int
        Number of neighbors for density estimation.

    Returns
    -------
    dict
        Dictionary with Pearson and Spearman correlations of local densities.
    """
    # Compute kNN densities in high-D and low-D
    rho_high = knn_density(X_high, k)
    rho_low = knn_density(X_low, k)

    # Compute correlations
    pearson_corr = pearsonr(rho_high, rho_low)[0]
    spearman_corr = spearmanr(rho_high, rho_low)[0]

    return {
        "density_spearman": spearman_corr,
    }

def compute_continuity(X_high, X_low, k=10):
    n = X_high.shape[0]

    nbrs_high = NearestNeighbors(n_neighbors=k + 1).fit(X_high)
    nbrs_low = NearestNeighbors(n_neighbors=k + 1).fit(X_low)

    high_neighbors = nbrs_high.kneighbors(return_distance=False)[:, 1:]
    low_neighbors = nbrs_low.kneighbors(return_distance=False)[:, 1:]

    continuity_sum = 0

    for i in range(n):
        missing = set(high_neighbors[i]) - set(low_neighbors[i])
        if not missing:
            continue

        low_ranking = nbrs_low.kneighbors(
            X_low[i].reshape(1, -1),
            n_neighbors=n,
            return_distance=False
        )[0]

        for j in missing:
            rank = np.where(low_ranking == j)[0][0]
            if rank > k:
                continuity_sum += rank - k

    norm = 2 / (n * k * (2 * n - 3 * k - 1))
    return 1 - norm * continuity_sum


def full_embedding_analysis(X_high, X_low, k=10):
    dist_high = pdist(X_high)
    dist_low = pdist(X_low)

    density_metrics = compute_density_preservation(X_high, X_low, k)

    return {
        "trustworthiness": trustworthiness(X_high, X_low, n_neighbors=k),
        "continuity": compute_continuity(X_high, X_low, k),
        "pearson_dist_corr": pearsonr(dist_high, dist_low)[0],
        "spearman_dist_corr": spearmanr(dist_high, dist_low)[0],
        **density_metrics
    }


# ========================
# XYZ Loader
# ========================

def load_geometries(folder):
    data, names = [], []

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".xyz"):
            continue

        path = os.path.join(folder, fname)

        try:
            with open(path) as f:
                lines = f.readlines()

            n_atoms = int(lines[0].strip())
            coords = [
                list(map(float, line.split()[1:4]))
                for line in lines[2:2 + n_atoms]
            ]

            coords = np.array(coords)

            if data and coords.shape != data[0].shape:
                raise ValueError("Inconsistent atom count")

            data.append(coords)
            names.append(fname)

        except Exception as e:
            print(f"Skipping {fname}: {e}")

    if not data:
        raise ValueError("No valid .xyz files found.")

    return np.array(data), names


# ========================
# Pipeline
# ========================

def run_pipeline(args):
    coords, _ = load_geometries(args.input)

    features = coords_to_flat_cartesian(coords)

    scaler = StandardScaler().fit(features)
    X_scaled = scaler.transform(features)

    D_high = compute_distance_matrix(X_scaled)

    methods = {
        "PCA": lambda X: run_pca(X, scaler, args.n_components)[0],
        "tSNE": lambda X: run_tsne(X, scaler, args.n_components),
        "UMAP": lambda X: run_umap(X, scaler, args.n_components),
        "densMAP": lambda X: run_densmap(X, scaler, args.n_components),
        "DiffusionMap": lambda X: run_diffusion_map(X, scaler, args.n_components),
        "Isomap": lambda X: run_isomap(X, scaler, args.n_components),
        "LaplacianEigenmaps": lambda X: run_laplacian_eigenmaps(X, scaler, args.n_components),
    }

    # ✅ OUTPUT DIR (fixed behavior)
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory: {os.path.abspath(output_dir)}")

    results = []

    for name, func in methods.items():
        print(f"\nRunning {name}...")
        start = time.time()

        try:
            X_low = func(features)
            runtime = time.time() - start

            D_low = compute_distance_matrix(X_low)

            metrics = full_embedding_analysis(X_scaled, X_low, k=args.k)

            results.append({
                "method": name,
                "stress": compute_stress(D_low, D_high),
                "runtime_sec": runtime,
                **metrics
            })

            if args.save_embeddings:
                np.save(os.path.join(output_dir, f"{name}_embedding.npy"), X_low)

        except Exception as e:
            print(f"{name} failed: {e}")

    df = pd.DataFrame(results)
    out_csv = os.path.join(output_dir, "metrics.csv")
    df.to_csv(out_csv, index=False)

    print(f"\nSaved metrics to {out_csv}")


# ========================
# CLI
# ========================

def main():
    parser = argparse.ArgumentParser(
        description="Dimensionality Reduction Benchmark"
    )

    parser.add_argument("-i", "--input", required=True,
                        help="Folder containing .xyz files")

    parser.add_argument("-o", "--output", required=True,
                        help="Output directory")

    parser.add_argument("-k", type=int, default=10,
                        help="Neighbors for metrics")

    parser.add_argument("-n", "--n-components", type=int, default=2,
                        help="Embedding dimension")

    parser.add_argument("--save-embeddings", action="store_true")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()