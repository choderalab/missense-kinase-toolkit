from itertools import chain
from os import path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mkt.ml.utils import try_except_string_in_list
from sklearn.cluster import KMeans


def plot_knee(
    list_in: list[int | float],
    vline=int,
    xlabel: str = "Number of clusters",
    ylabel: str = "Sum of squared errors",
    title: str = "Elbow Method",
    filename: str = "elbow.png",
    path_out: str = "./plots",
):
    """Plot knee method

    Parameters:
    -----------
    list_in : list
        List of values to plot (e.g., sum of squared errors)
    vline : int
        Vertical line position
    path_out : str, default="./plots"
        Path to save the plot
    """
    plt.plot(range(1, len(list_in) + 1), list_in, marker="o")
    plt.axvline(x=vline, color="red", linestyle="--")
    plt.text(vline + 2, max(list_in[1:]), f"n={vline}", fontsize=12, color="red")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    filepathname = path.join(path_out, filename)
    plt.savefig(filepathname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dim_red_scatter(
    df_input: pd.DataFrame,
    kmeans: KMeans,
    method: str,
    palette: str = "Set1",
    path_out: str = "./plots",
) -> None:
    """Plot scatter plot of dimensionality reduction

    Parameters:
    -----------
    df_input : DataFrame
        DataFrame containing the dimensionality reduction coordinates
    kmeans : KMeans object
        Fitted KMeans object with labels_ attribute
    method : str
        Method used for dimensionality reduction
    palette : str, default="Set1"
        Color palette to use for the plot (make sure >= n_clusters)
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

    filepathname = path.join(path_out, f"{method.lower()}.png")
    plt.savefig(filepathname, dpi=300, bbox_inches="tight")
    plt.close()


def generate_dict_count(
    df: pd.DataFrame,
    col: str,
    n_cutoff: int,
    bool_iterable: bool,
) -> dict[str, int]:
    """
    Generate a dictionary with the count of each unique entry in a column of a DataFrame

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with property (e.g., group) column
    col : str
        Column name in df_annot to use for filtering
    n_cutoff : int, default=5
        Minimum number of entries in a property to include in the grid

    Returns:
    --------
    dict_property : dict
        Dictionary with the count of each unique entry in the column
    """
    # unpack column of iterables (e.g., group) via list of iterables
    if bool_iterable:
        list_col = df.loc[~df[col].isnull(), col].tolist()
        set_annot = set(chain(*list_col))
        dict_annot = {
            entry: sum([entry in i for i in df.loc[~df[col].isnull(), col]])
            for entry in set_annot
        }
    # if not a list of iterables, will not be able to unpack list_col
    else:
        dict_annot = dict(df.loc[~df[col].isnull(), col].value_counts())

    dict_count = {
        prop: count for prop, count in dict_annot.items() if count >= n_cutoff
    }
    dict_count = dict(
        sorted(dict_count.items(), key=lambda item: item[1], reverse=True)
    )

    return dict_count


def plot_scatter_grid(
    df_annot: pd.DataFrame,
    df_input: pd.DataFrame,
    kmeans: KMeans,
    method: str,
    col: str = "group_consensus",
    bool_iterable: bool = True,
    n_cutoff: int = 5,
    n_cols: int = 5,
    path_out: str = "./plots",
) -> None:
    """
    Create a grid of scatter plots for categorical properties with kmeans clustering in first position.
    Points within each property are colored by their k-means cluster, others are light grey.

    Parameters:
    -----------
    df_annot : pd.DataFrame
        Dataframe with property (e.g., group) column
    df_input : DataFrame
        DataFrame containing the 2D dim red coordinates in same order as df_annot
    kmeans : KMeans object
        Fitted KMeans object with labels_ attribute
    method : str
        Method used for dimensionality reduction for labeling puroposes
    col : str, default="group_consensus"
        Column name in df_annot to use for filtering
    bool_iterable : bool, default=True
        Whether the column is an iterable (e.g., group) or not
    n_cols : int, default=5
        Number of columns in the grid
    n_cutoff : int, default=5
        Minimum number of entries in a property to include in the grid
    path_out : str, default="./plots"
        Path to save the plot
    """
    # Filter property according to cutoff
    dict_property = generate_dict_count(df_annot, col, n_cutoff, bool_iterable)

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

    # Get k-means clusters
    n_clusters = len(np.unique(kmeans.labels_))
    cluster_labels = kmeans.labels_

    # Create kmeans plot in position (1,1)
    cmap = plt.get_cmap("Set1", n_clusters)
    axes[0].scatter(
        df_input[col1],
        df_input[col2],
        c=cluster_labels
        + 1,  # +1 to avoid cluster 0 being treated differently in colormap
        cmap=cmap,
        alpha=0.5,
    )
    axes[0].set_xlabel(col1, fontsize=12)
    axes[0].set_ylabel(col2, fontsize=12)
    axes[0].set_title(
        f"KMeans Clustering (k={n_clusters})", fontsize=14, fontweight="bold"
    )

    # Create plots for each property
    for i, (prop, count) in enumerate(dict_property.items()):
        if i + 1 < len(axes):  # +1 because we used the first position for kmeans
            ax = axes[i + 1]

            # Get property membership vector (True/False)
            prop_mask = df_annot[col].apply(
                lambda x: try_except_string_in_list(prop, x)
            )

            # Plot non-property points first (light grey)
            ax.scatter(
                df_input.loc[~prop_mask, col1],
                df_input.loc[~prop_mask, col2],
                c="lightgrey",
                alpha=0.3,
            )

            # Plot property points colored by cluster
            for cluster_id in range(n_clusters):
                # Get points that are both in this property and in this cluster
                cluster_prop_mask = prop_mask & (cluster_labels == cluster_id)

                # Plot only if there are any points in this combination
                if cluster_prop_mask.any():
                    ax.scatter(
                        df_input.loc[cluster_prop_mask, col1],
                        df_input.loc[cluster_prop_mask, col2],
                        c=[
                            cmap(cluster_id / n_clusters)
                        ],  # Use same color as in the cluster plot
                        alpha=0.7,
                        label=(
                            f"Cluster {cluster_id+1}" if i == 0 else None
                        ),  # Only add label in first plot
                    )

            ax.set_xlabel(col1, fontsize=12)
            ax.set_ylabel(col2, fontsize=12)
            ax.set_title(f"{prop} (n={count})", fontsize=14, fontweight="bold")

    # Hide any unused subplots
    for j in range(
        i + 2, len(axes)
    ):  # +2 because we used the first position for kmeans
        axes[j].axis("off")

    # Add overall title with additional top padding
    title_print = col.replace("_", " ").title()
    # title_print = "".join(
    #     ch.replace.upper() if idx == 0 else ch.lower() for idx, ch in enumerate(col)
    # )
    # bold font

    fig.suptitle(
        f"{method} of Kinase Embeddings by {title_print}",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )

    method_print = "".join(ch.lower() for ch in method if ch.isalnum())
    filepathname = path.join(path_out, f"{method_print}_{col}_grid.svg")
    plt.savefig(filepathname, dpi=300, bbox_inches="tight")
    plt.close()
