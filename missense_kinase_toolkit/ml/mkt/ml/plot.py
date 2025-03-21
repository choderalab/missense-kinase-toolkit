from os import path

import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def plot_dim_red_scatter(
    df_input: pd.DataFrame,
    kmeans: KMeans,
    palette: str = "Set1",
    method: str = "PCA",
    path_out: str = "./plots",
) -> None:
    """Plot scatter plot of dimensionality reduction

    Parameters:
    -----------
    df_input : DataFrame
        DataFrame containing the dimensionality reduction coordinates
    kmeans : KMeans object
        Fitted KMeans object with labels_ attribute
    palette : str, default="Set1"
        Color palette to use for the plot (make sure >= n_clusters)
    method : str, default="PCA"
        Method used for dimensionality reduction
    path_out : str, default="./plots"
        Path to save the plot
    """
    n_clusters = len(np.unique(kmeans.labels_))
    cmap = plt.get_cmap(palette, n_clusters)

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

    plt.savefig(path.join(path_out, f"{method.lower()}.png"))
    plt.close()


def scatter_plot_binary_grid(
    dfs_dict: pd.DataFrame,
    df_input: pd.DataFrame,
    kmeans: KMeans,
    n_cutoff: int = 5,
    n_cols: int = 5,
    col: str = "group",
    method: str = "tSNE",
    path_out: str = "./plots",
) -> None:
    """
    Create a grid of scatter plots for multiple families with kmeans clustering in first position

    Parameters:
    -----------
    dfs_dict : dict
        Dictionary with property (e.g., group) names as keys and their counts as values
    df_input : DataFrame
        DataFrame containing the t-SNE coordinates
    kmeans : KMeans object
        Fitted KMeans object with labels_ attribute
    n_cols : int, default=6
        Number of columns in the grid
    n_cutoff : int, default=5
        Minimum number of entries in a family to include in the grid
    col : str, default="group"
        Column name in df_annot to use for filtering
    method : str, default="tSNE"
        Method used for dimensionality reduction
    path_out : str, default="./plots"
        Path to save the plot
    """
    # Filter property according to cutoff
    dict_property = {
        prop: count for prop, count in dfs_dict.items() if count >= n_cutoff
    }
    dict_property = dict(
        sorted(dict_property.items(), key=lambda item: item[1], reverse=True)
    )

    # Calculate grid dimensions
    n_plots = len(dict_property) + 1  # +1 for the kmeans plot
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
    for i, (fam, count) in enumerate(dict_property.items()):
        if i + 1 < len(axes):  # +1 because we used the first position for kmeans
            ax = axes[i + 1]
            vx_fam = df_annot[col].apply(lambda x: try_except_string_in_list(fam, x))

            ax.scatter(
                df_input[col1], df_input[col2], c=vx_fam, cmap=cmap_bin, alpha=0.5
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

    method_print = "".join(ch.lower() for ch in method if ch.isalnum())
    plt.savefig(
        path.join(path_out, f"{method_print}_{col}_grid.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
