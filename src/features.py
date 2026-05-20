"""
features.py
-----------
Preprocessing, dimensionality reduction, and clustering pipeline.

Used by app.py to process uploaded CSVs into cluster assignments
and 2D coordinates for visualisation.

Pipeline:
    Raw features → StandardScaler → PCA (50 dims) → t-SNE (2D) → Clusters
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def preprocess(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    Select features, drop missing values, and standardise.
    Returns scaled numpy array.
    """
    X = df[feature_cols].dropna().values
    scaler = StandardScaler()
    return scaler.fit_transform(X), df[feature_cols].dropna().index


def run_pca(X: np.ndarray, n_components: int = 50) -> np.ndarray:
    """
    Reduce to n_components dimensions with PCA.
    If n_components >= n_features, skip (already low-dimensional).
    """
    n_components = min(n_components, X.shape[1], X.shape[0])
    if n_components >= X.shape[1]:
        return X  # already low-dimensional, skip PCA
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)


def run_tsne(X: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """
    Reduce to 2D with t-SNE.
    Perplexity is roughly the number of nearest neighbours considered.
    """
    perplexity = min(perplexity, X.shape[0] - 1)
    # Cap at 2,000 rows — t-SNE scales poorly beyond this on CPU
    if X.shape[0] > 2000:
        idx = np.random.default_rng(42).choice(X.shape[0], 2000, replace=False)
        X = X[idx]

    perplexity = min(perplexity, X.shape[0] - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=300,
        init="pca",
        learning_rate="auto",
        method="barnes_hut",
    )
    return tsne.fit_transform(X)


def cluster_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    """Fit K-Means and return cluster labels."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(X)


def cluster_dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Fit DBSCAN and return cluster labels.
    Label -1 means noise (points that don't belong to any cluster).
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    return db.fit_predict(X)


def elbow_scores(X: np.ndarray, k_range: range) -> list[float]:
    """Compute inertia for each k — used to plot the elbow curve."""
    return [
        KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
        for k in k_range
    ]


def run_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    algorithm: str,
    perplexity: int = 30,
    k: int = 3,
    eps: float = 0.5,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Full pipeline: preprocess → PCA → t-SNE → cluster.

    Returns a DataFrame with columns:
        x, y         — 2D t-SNE coordinates
        cluster      — cluster label (int, -1 = noise for DBSCAN)
        + all original feature columns
    """
    X_scaled, valid_idx = preprocess(df, feature_cols)
    X_pca  = run_pca(X_scaled)
    X_tsne = run_tsne(X_pca, perplexity=perplexity)

    if algorithm == "K-Means":
        labels = cluster_kmeans(X_scaled, k=k)
    else:
        labels = cluster_dbscan(X_scaled, eps=eps, min_samples=min_samples)

    result = df.loc[valid_idx, feature_cols].copy().reset_index(drop=True)
    result["x"]       = X_tsne[:, 0]
    result["y"]       = X_tsne[:, 1]
    result["cluster"] = labels

    return result