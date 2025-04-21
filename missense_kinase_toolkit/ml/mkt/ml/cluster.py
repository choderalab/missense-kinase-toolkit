import logging

import numpy as np
import torch
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

logger = logging.getLogger(__name__)


DICT_METHODS = {
    "k-Means": {
        "model": KMeans,
        # need to manually add n_clusterss
        "dict_kwarg": {
            "init": "random",
            "n_init": 10,
            "max_iter": 300,
            "random_state": 42,
        },
        "scaler": StandardScaler,
    },
    "PCA": {
        "model": PCA,
        "dict_kwarg": {
            "n_components": 2,
        },
        "scaler": StandardScaler,
    },
    "t-SNE": {
        "model": TSNE,
        "dict_kwarg": {
            "n_components": 2,
            "learning_rate": "auto",
            "init": "random",
            "perplexity": 3,
        },
        "scaler": StandardScaler,
    },
    "DBSCAN": {
        "model": DBSCAN,
        "dict_kwarg": {
            "eps": 0.0001,
            "metric": "cosine",
        },
        "scaler": StandardScaler,
    },
    "Spectral": {
        "model": SpectralClustering,
        "dict_kwargs": {
            "n_clusters": 2,
            "assign_labels": "kmeans",
            "random_state": 42,
        },
        "scaler": StandardScaler,
    },
    "UMAP": {
        "model": UMAP,
        "dict_kwarg": {
            "n_components": 2,
        },
        "scaler": StandardScaler,
    },
}


def generate_clustering(
    method: str,
    mx_input: torch.Tensor | np.ndarray,
    dict_kwarg: dict | None = None,
    bool_scale=False,
):
    if dict_kwarg is None:
        try:
            dict_kwarg = DICT_METHODS[method]["dict_kwarg"]
        except KeyError:
            raise logger.error(f"Method {method} not found in DICT_METHODS")

    if bool_scale:
        scaler = DICT_METHODS[method]["scaler"]
        mx_input = scaler().fit_transform(mx_input)

    model = DICT_METHODS[method]["model"](**dict_kwarg)
    model.fit(mx_input)

    return model


def find_kmeans(
    mx_input: np.ndarray,
    dict_kwarg: str | None = None,
    bool_scale: bool = False,
    n_clust=30,
) -> tuple[object, list, list]:
    """Find optimal number of clusters using elbow method

    Parameters:
    -----------
    mx_input : np.ndarray
        Input matrix
    dict_kwarg : dict
        Dictionary with KMeans parameters
    n_clust : int, default=30
        Maximum number of clusters to test

    Returns:
    --------
    kmeans : KMeans object
        Fitted KMeans object
    list_sse : list
        List of sum of squared errors
    list_silhouette : list
        List of silhouette scores
    """
    from kneed import KneeLocator
    from sklearn.metrics import silhouette_score

    if dict_kwarg is None:
        dict_kwarg = DICT_METHODS["k-Means"]["dict_kwarg"]

    if bool_scale:
        scaler = DICT_METHODS["k-Means"]["scaler"]
        mx_input = scaler().fit_transform(mx_input)

    model = DICT_METHODS["k-Means"]["model"]

    np.random.seed(dict_kwarg["random_state"])

    # Generate k-means for 1:n_clust
    list_sse, list_silhouette = [], []
    for k in range(1, n_clust + 1):
        kmeans = model(n_clusters=k, **dict_kwarg)
        kmeans.fit(mx_input)

        list_sse.append(kmeans.inertia_)

        if k > 1:
            score = silhouette_score(mx_input, kmeans.labels_)
            list_silhouette.append(score)

    # Find elbow
    kl = KneeLocator(
        range(1, n_clust + 1),
        list_sse,
        curve="convex",
        direction="decreasing",
    )
    n_clust = kl.elbow

    kmeans = model(n_clusters=n_clust, **dict_kwarg)
    kmeans.fit(mx_input)

    return kmeans, list_sse, list_silhouette


def generate_cosine_similarity_matrix(
    mx_input: torch.Tensor,
    bool_row: bool = True,
):
    """Generate similarity matrix

    Params:
    -------
    mx_input: torch.Tensor
        Input matrix
    bool_row: bool; default=True
        Whether to calculate similarity by row (True) or by column (False)

    Returns:
    --------
    mx_similarity: torch.Tensor
        Square, symmetrix similarity matrix containing pairwise cosine similarities
    """
    if bool_row:
        mx_norm = mx_input / mx_input.norm(dim=1, p=2, keepdim=True)
        mx_similarity = mx_norm @ mx_norm.T
    else:
        mx_norm = mx_input / mx_input.norm(dim=0, p=2, keepdim=True)
        mx_similarity = mx_norm.T @ mx_norm

    return mx_similarity


def create_laplacian(
    mx_input: torch.Tensor,
):
    """Create graph Laplacian

    Params:
    -------
    mx_input: torch.Tensor
        Input matrix

    Returns:
    --------
    mx_laplacian: torch.Tensor
        Graph Laplacian
    """
    from scipy import sparse

    mx_laplacian = sparse.csgraph.laplacian(csgraph=mx_input, normed=True)

    return mx_laplacian
