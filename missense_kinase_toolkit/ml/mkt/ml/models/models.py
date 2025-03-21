from ast import literal_eval

import numpy as np
import pandas as pd
import torch

from mkt.ml.utils import (
    return_device, 
    try_except_string_in_list,
)
from mkt.ml.plot import (
    plot_knee,
    plot_dim_red_scatter,
    plot_scatter_grid,
)
from mkt.ml.cluster import find_kmeans, generate_clustering
from transformers import AutoModel, AutoTokenizer


# TODO:
# 1. MLP
# 2. Factor model
# 3. Separate tokenzier?

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

col = "group"
df_annot.loc[~df_annot[col].isnull(), col] = df_annot.loc[
    ~df_annot[col].isnull(), col
].apply(literal_eval)

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

for layer in outputs_kinase.hidden_states:
    print(layer.shape)

X = outputs_kinase.pooler_output.cpu().numpy()
# save X locally
np.save("./kinase_pooler_layer.npy", X)
# mx_kinase_sim = generate_similarity_matrix(outputs_kinase.pooler_output.cpu())
# mx_kinase_euclidan = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2))

# CLUSTERING

# model_DB = DBSCAN(eps=0.0001, metric="cosine").fit(mx_kinase_sim.cpu().numpy())
# labels = model_DB.labels_
# unique, counts = np.unique(labels, return_counts=True)

X = np.load("./kinase_pooler_layer.npy")

scale_bool = False
kmeans, list_sse, list_silhouette = find_kmeans(mx_input=X, bool_scale=scale_bool)
n_clusters = len(np.unique(kmeans.labels_))
plot_knee(list_sse, n_clusters, filename="elbow.png")

# PCA
pca = generate_clustering("PCA", X.T, bool_scale=scale_bool)
df_pca = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"])
plot_dim_red_scatter(df_pca, kmeans, method="PCA")
plot_scatter_grid(df_annot, df_pca, kmeans, "PCA")

# t-SNE
tsne = generate_clustering("t-SNE", X, bool_scale=scale_bool)
df_tsne = pd.DataFrame(tsne.embedding_, columns=["tSNE1", "tSNE2"])
plot_dim_red_scatter(df_tsne, kmeans, method="tSNE")
plot_scatter_grid(df_annot, df_tsne, kmeans, "t-SNE")

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
