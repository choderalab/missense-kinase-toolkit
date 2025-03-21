import logging
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from umap import UMAP # need to install umap-learn if want to use

logger = logging.getLogger(__name__)


DICT_METHODS = {
    "model": KMeans,
    "k-Means": {
        "dict_kwarg": {
            "init": "random", 
            "n_init": 10, 
            "max_iter": 300, 
            "random_state": 42
        }
    },
    "PCA": {
        "model": PCA,
        "dict_kwarg": {
            "n_components": 2,
        }
    },
    "tSNE": {
        "model": TSNE,
        "dict_kwarg": {
            "n_components": 2, 
            "learning_rate": "auto", 
            "init": "random"
            "perplexity": 3,
        }

    },
    "DBSCAN": {
        "model": DBSCAN,
        "dict_kwarg": {
            "eps": 0.0001,
            "metric": "cosine",
        },
    },
    "Spectral": {
        "model": SpectralClustering,
        "dict_kwargs": {
            "n_clusters": 2,
            "assign_labels": "kmeans",
            "random_state": 42,
        }

    },
    # "UMAP": {
    #     "model": UMAP,
    #     "dict_kwarg": {
    #         "n_components": 2,
    #     }
    # },
}


def generate_clustering(
    method: str,
    mx_input: torch.Tensor | np.array,
    dict_kwarg: dict | None = None,
):
    if dict_kwarg is None:
        try:
            dict_kwarg = DICT_METHODS[method]["dict_kwarg"]
        except KeyError:
            raise logger.error(f"Method {method} not found in DICT_METHODS")

    

def find_kmeans(
    mx_input: np.array,
    dict_kwarg: str | None = None,
    n_clust=30,
) -> tuple[ object, list, list ]:
    """Find optimal number of clusters using elbow method

    Parameters:
    -----------
    mx_input : np.array
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
    from sklearn.metrics import silhouette_score
    from kneed import KneeLocator

    if dict_kwarg is None:
        dict_kwarg = DICT_METHODS["k-Means"]["dict_kwarg"]
    
    np.random.seed(dict_kwarg["random_state"])

    list_sse = [], list_silhouette = []
    for k in range(1, n_clust + 1):
        kmeans = DICT_METHODS["k-Means"]["model"](
            n_clusters=k, 
            **dict_kwarg
        )
        kmeans.fit(mx_input)
        list_sse.append(kmeans.inertia_)
        if k > 1:
            score = silhouette_score(mx_input, kmeans.labels_)
            list_silhouette.append(score)

    kl = KneeLocator(
        range(1, n_clust + 1), list_sse, curve="convex", direction="decreasing"
    )
    n_clust = kl.elbow

    kmeans = KMeans(n_clusters=n_clust, **dict_kwarg)
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
    bool_row: bool
        Whether to calculate similarity by row; default is True

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
