import numpy as np
import pandas as pd
import torch
from mkt.ml.utils import generate_similarity_matrix, return_device  # noqa: F401
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import DBSCAN, SpectralClustering # noqa: F401

# TODO:
# 1. MLP
# 2. Factor model
# 3. Separate tokenzier?

# SET-UP

device = return_device()

# LOAD DATA

df_pkis2 = pd.read_csv(
    "https://raw.githubusercontent.com/openkinome/kinoml/refs/heads/master/kinoml/data/kinomescan/journal.pone.0181585.s004.csv"
)

df_pkis_rev = df_pkis2.set_index("Smiles").iloc[:, 6:]
# df_pkis_rev.index

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

# KINASE MODEL

model_kinase_name = "facebook/esm2_t6_8M_UR50D"

model_kinase = AutoModel.from_pretrained(
    model_kinase_name,
    device_map="auto",
)

tokenizer_kinase = AutoTokenizer.from_pretrained(model_kinase_name)

kinase_tokens = tokenizer_kinase(
    df_pkis_rev.columns.to_list(),
    return_tensors="pt",
    padding=True,
).to(device)

with torch.no_grad():
    outputs_kinase = model_drug(**kinase_tokens, output_hidden_states=True)

for layer in outputs_kinase.hidden_states:
    print(layer.shape)

mx_kinase_sim = generate_similarity_matrix(outputs_kinase.pooler_output)

# CLUSTERING

model_DB = DBSCAN(eps=0.1, metric="cosine").fit(mx_kinase_sim.cpu().numpy())
labels = model_DB.labels_
unique, counts = np.unique(labels, return_counts=True)


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
