#!/usr/bin/env python

from os import path
import argparse
from ast import literal_eval
import logging
import numpy as np
import pandas as pd
from mkt.ml.cluster import find_kmeans, generate_clustering
from mkt.ml.plot import plot_knee, plot_dim_red_scatter, plot_scatter_grid


logger = logging.getLogger(__name__)

def parse_args():
    parse_args = argparse.ArgumentParser()

    parse_args.add_argument(
        "--path_mx_npy",
        type=str, 
        default="/data1/tanseyw/projects/whitej/missense-kinase-toolkit/kinase_pooler_layer.npy", 
        help="Path to embedding matrix",
    )
    
    parse_args.add_argument(
        "--path_annot_csv", 
        type=str, 
        default="/data1/tanseyw/projects/whitej/missense-kinase-toolkit/data/pkis2_annotated_group.csv", 
        help="Path to annotated CSV",
    )

    parse_args.add_argument(
        "--path_out", 
        type=str, 
        default="/data1/tanseyw/projects/whitej/missense-kinase-toolkit/plots", 
        help="Path to output directory"
    )

    parse_args.add_argument(
        "--bool_scale", 
        action="store_true", 
        help="Scale input matrix",
    )

    args = parse_args.parse_args()

    return args


def main():
    args = parse_args()

    for file in [args.path_mx_npy, args.path_annot_csv]:
        if not path.isfile(file):
            logger.error(f"File {file} not found")

    if not path.isdir(args.path_out):
        logger.error(f"Directory {args.path_out} not found")

    X = np.load(args.path_mx_npy)
    df_annot = pd.read_csv(args.path_annot_csv)

    list_to_drop = ["-phosphorylated", "-cyclin", "-autoinhibited"]
    idx_drop = df_annot["DiscoverX Gene Symbol"].apply(
        lambda x: any([i in x for i in list_to_drop])
    )
    df_annot = df_annot.loc[~idx_drop, :].reset_index(drop=True)
    df_annot = df_annot.loc[df_annot["sequence_partial"].notnull(), :].reset_index(
        drop=True
    )

    col = "group"
    df_annot.loc[~df_annot[col].isnull(), col] = df_annot.loc[
        ~df_annot[col].isnull(), col
    ].apply(literal_eval)

    if args.bool_scale:
        scale_bool = True
    else:
        scale_bool = False
    
    kmeans, list_sse, list_silhouette = find_kmeans(mx_input=X, bool_scale=scale_bool)
    n_clusters = len(np.unique(kmeans.labels_))
    plot_knee(list_sse, n_clusters, filename="elbow.png", path_out=args.path_out)

    # PCA
    pca = generate_clustering("PCA", X.T, bool_scale=scale_bool)
    df_pca = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"])
    plot_dim_red_scatter(df_pca, kmeans, method="PCA", path_out=args.path_out)
    plot_scatter_grid(df_annot, df_pca, kmeans, "PCA", path_out=args.path_out)

    # t-SNE
    tsne = generate_clustering("t-SNE", X, bool_scale=scale_bool)
    df_tsne = pd.DataFrame(tsne.embedding_, columns=["tSNE1", "tSNE2"])
    plot_dim_red_scatter(df_tsne, kmeans, method="tSNE", path_out=args.path_out)
    plot_scatter_grid(df_annot, df_tsne, kmeans, "t-SNE", path_out=args.path_out)


if __name__ == "__main__":
    main()