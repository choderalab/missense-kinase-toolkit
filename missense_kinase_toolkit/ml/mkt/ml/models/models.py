from ast import literal_eval
from itertools import chain

import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kneed import KneeLocator
from mkt.ml.utils import generate_similarity_matrix, return_device  # noqa: F401
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

# from umap import UMAP # need to install umap-learn if want to use


# TODO:
# 1. MLP
# 2. Factor model
# 3. Separate tokenzier?

# FUNCTIONS


def find_kmeans(
    mx_input,
    dict_kwarg,
    n_clust=30,
):
    list_sse = []
    list_silhouette = []

    for k in range(1, n_clust + 1):
        kmeans = KMeans(n_clusters=k, **dict_kwarg)
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


def plot_dim_red_scatter(
    df_input: pd.DataFrame,
    kmeans: KMeans,
    method: str = "PCA",
):
    n_clusters = len(np.unique(kmeans.labels_))
    cmap = plt.get_cmap("Set1", n_clusters)

    if df_input.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")
    else:
        col1, col2 = df_input.columns
        plt.scatter(
            data=df_input,
            x=col1,
            y=col2,
            c=kmeans.labels_ + 1,
            cmap=cmap,
            alpha=0.5,
        )

    patches = [
        mpatches.Patch(color=cmap(i), label=str(i + 1)) for i in range(n_clusters)
    ]
    plt.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.14, 1))

    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f"{method} of Kinase Embeddings")

    plt.savefig(f"./plots/{method.lower()}.png")
    plt.close()


def try_except_entry_in_list(str_in, list_in):
    try:
        return str_in in list_in
    except:
        return False


def scatter_plot_binary_grid(
    dfs_dict: pd.DataFrame,
    df_input: pd.DataFrame,
    kmeans: KMeans,
    col: str = "group",
    method: str = "tSNE",
    n_cutoff: int = 5,
    n_cols: int = 5,
) -> None:
    """
    Create a grid of scatter plots for multiple families with kmeans clustering in first position

    Parameters:
    -----------
    dfs_dict : dict
        Dictionary with family names as keys and their counts as values
    df_input : DataFrame
        DataFrame containing the t-SNE coordinates
    kmeans : KMeans object
        Fitted KMeans object with labels_ attribute
    n_cols : int, default=6
        Number of columns in the grid
    """
    # Filter families according to cutoff
    filtered_families = {
        fam: count for fam, count in dfs_dict.items() if count >= n_cutoff
    }

    # Calculate grid dimensions
    n_plots = len(filtered_families) + 1  # +1 for the kmeans plot
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Add top padding to prevent title overlap
    fig.tight_layout(
        pad=4.0, rect=[0, 0, 1, 0.975]
    )  # Adjust rect to leave space for suptitle

    # Flatten axes array for easy indexing
    axes = axes.flatten() if n_rows > 1 else axes

    if df_input.shape[1] != 2:
        raise ValueError("Input DataFrame must have exactly two columns")

    col1, col2 = df_input.columns
    # Create kmeans plot in position (1,1)
    n_clusters = len(np.unique(kmeans.labels_))
    cmap = plt.get_cmap("Set1", n_clusters)
    axes[0].scatter(
        df_input[col1], 
        df_input[col2], 
        c=kmeans.labels_ + 1, 
        cmap=cmap, 
        alpha=0.5,
    )
    axes[0].set_xlabel(col1)
    axes[0].set_ylabel(col2)
    axes[0].set_title(f"KMeans Clustering (k={n_clusters})")

    # Create colormap for binary plots
    cmap_bin = colors.ListedColormap(["grey", "red"])

    # Create plots for each family
    for i, (fam, count) in enumerate(filtered_families.items()):
        if i + 1 < len(axes):  # +1 because we used the first position for kmeans
            ax = axes[i + 1]
            vx_fam = df_annot[col].apply(lambda x: try_except_entry_in_list(fam, x))

            ax.scatter(
                df_input[col1], 
                df_input[col2],
                c=vx_fam, 
                cmap=cmap_bin, 
                alpha=0.5
            )
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            ax.set_title(f"{fam} (n={count})")

    # Hide any unused subplots
    for j in range(
        i + 2, len(axes)
    ):  # +2 because we used the first position for kmeans
        axes[j].axis("off")

    # Add overall title with additional top padding
    fig.suptitle(f"{method} of Kinase Embeddings by {col.upper()}", fontsize=16, y=0.98)

    method_print = ''.join(ch.lower() for ch in method if ch.isalnum())
    plt.savefig(f"./plots/{method_print}_{col}_grid.png", dpi=300, bbox_inches="tight")
    plt.close()


# SET-UP

device = return_device()

# LOAD DATA AND PREPROCESS

df_pkis2 = pd.read_csv(
    "https://raw.githubusercontent.com/openkinome/kinoml/refs/heads/master/kinoml/data/kinomescan/journal.pone.0181585.s004.csv"
)

df_pkis_rev = df_pkis2.set_index("Smiles").iloc[:, 6:]

# added kinase groups manually
# TODO: automate this
df_annot = pd.read_csv(
    "/data1/tanseyw/projects/whitej/missense-kinase-toolkit/data/pkis2_annotated_group.csv"
)
list_to_drop = ["-phosphorylated", "-cyclin", "-autoinhibited"]
idx_drop = df_annot["DiscoverX Gene Symbol"].apply(
    lambda x: any([i in x for i in list_to_drop])
)
df_annot = df_annot.loc[~idx_drop, :].reset_index(drop=True)
df_annot = df_annot.loc[df_annot["sequence_partial"].notnull(), :].reset_index(
    drop=True
)

# all NA (e.g., no start/end)
# list_dup = df_annot.loc[df_annot["sequence_partial"].duplicated(), "sequence_partial"].to_list()
# df_annot.loc[df_annot["sequence_partial"].isin(list_dup), "accession"]

# KINASE MODEL

model_kinase_name = "facebook/esm2_t6_8M_UR50D"

model_kinase = AutoModel.from_pretrained(
    model_kinase_name,
    # device_map="auto",
).to(device)

tokenizer_kinase = AutoTokenizer.from_pretrained(model_kinase_name)

kinase_tokens = tokenizer_kinase(
    df_annot["sequence_partial"].to_list(),
    return_tensors="pt",
    padding=True,
).to(device)

with torch.no_grad():
    outputs_kinase = model_kinase(**kinase_tokens, output_hidden_states=True)

# for layer in outputs_kinase.hidden_states:
#     print(layer.shape)

# mx_kinase_sim = generate_similarity_matrix(outputs_kinase.pooler_output.cpu())
X = outputs_kinase.pooler_output.cpu().numpy()
# save X locally
np.save("./kinase_pooler_layer.npy", X)
# mx_kinase_euclidan = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2))

# CLUSTERING

# model_DB = DBSCAN(eps=0.0001, metric="cosine").fit(mx_kinase_sim.cpu().numpy())
# labels = model_DB.labels_
# unique, counts = np.unique(labels, return_counts=True)

seed = 42
np.random.seed(seed)

kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": seed}

kmeans, list_sse, list_silhouette = find_kmeans(X, kmeans_kwargs)
n_clusters = len(np.unique(kmeans.labels_))
cmap = plt.get_cmap("Set1", n_clusters)

# PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)  # Specify the number of components
principal_components = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(
    n_components=2, learning_rate="auto", init="random", perplexity=3
).fit_transform(X)

# umap = UMAP(n_components=2)
# umap_components = umap.fit_transform(X_scaled)

# PLOT

# PCA
plot_dim_red_scatter(
    pd.DataFrame(principal_components, columns=["PC1", "PC2"]),
    kmeans,
    method="PCA",
)
# centers = pca.transform(kmeans.cluster_centers_)
# plt.scatter(centers[:, 0], centers[:, 1], c="red", s=100, alpha=0.5)

# t-SNE
plot_dim_red_scatter(
    pd.DataFrame(tsne, columns=["tSNE1", "tSNE2"]),
    kmeans,
    method="tSNE",
)

# UMAP
# plot_dim_red_scatter(
#     pd.DataFrame(umap_components, columns=["UMAP1", "UMAP2"]),
#     kmeans,
#     method="UMAP",
# )

col = "group"
df_annot.loc[~df_annot[col].isnull(), col] = df_annot.loc[
    ~df_annot[col].isnull(), col
].apply(literal_eval)
set_annot = set(chain(*df_annot.loc[~df_annot[col].isnull(), col].tolist()))
dict_annot = {
    entry: sum([entry in i for i in df_annot.loc[~df_annot[col].isnull(), col]])
    for entry in set_annot
}
dict_annot = dict(sorted(dict_annot.items(), key=lambda item: item[1], reverse=True))

# Call the function with your dictionary of families and kmeans object
df_plot = pd.DataFrame(tsne, columns=["tSNE1", "tSNE2"])
scatter_plot_binary_grid(dict_annot, df_plot, kmeans)

# NOT IN USE
# config_automodel = AutoConfig.from_pretrained(model_name)
# [i for i in model_kinase.state_dict().keys()]
# [dir(i) for i in model_kinase.state_dict().keys()]
# [i.split(".")[0] for i in model_kinase.state_dict().keys()]
# list(model_kinase.state_dict().values())[-1]
# model_kinase.embeddings
# model_kinase.encoder
# del model_kinase.pooler
# del model_kinase.contact_head

# config_automodel.architectures

# type(model_kinase)
# isinstance(model_kinase, torch.nn.Module)

# list_keep = ["encoder", "embeddings"]
# set_drop = set()
# for k, v in model_kinase.state_dict().items():
#     if k.split(".")[0] not in list_keep:
#         set_drop.add(k.split(".")[0])
# for drop in set_drop:
#     delattr(model_kinase, drop)


# for k, v in model_kinase.state_dict().items():
#     print(k)
#     print(v.shape)
#     print()

# DRUG MODEL

# MTR = multi-task regression
model_drug_name = "DeepChem/ChemBERTa-77M-MTR"
# MLM = masked language model
# model_drug_name = "DeepChem/ChemBERTa-77M-MLM"

model_drug = AutoModel.from_pretrained(
    model_drug_name,
    device_map="auto",
)
# model_drug

tokenizer_drug = AutoTokenizer.from_pretrained(model_drug_name)
# {v: k for k, v in tokenizer_drug.vocab.items()}[12] # [CLS]

drug_tokens = tokenizer_drug(
    df_pkis_rev.index.to_list(),
    return_tensors="pt",
    padding=True,
).to(device)

with torch.no_grad():
    outputs_drug = model_drug(**drug_tokens, output_hidden_states=True)

for layer in outputs_drug.hidden_states:
    print(layer.shape)

mx_drug_sim = generate_similarity_matrix(outputs_drug.pooler_output)

# torch.allclose(mx_similarity, mx_similarity.T)
# torch.all(torch.diag(mx_similarity) == 1.0000)

# torch.allclose(mx_dotprod, mx_dotprod.T)
