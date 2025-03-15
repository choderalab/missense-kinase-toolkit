import pandas as pd
import torch
from mkt.ml.utils import generate_similarity_matrix, return_device
from transformers import AutoConfig, AutoModel, AutoTokenizer, EsmForMaskedLM

# TODO:
# 1. MLP
# 2. Factor model
# 3. Separate tokenzier?

### SET-UP ###

device = return_device()

### LOAD DATA ###

df_pkis2 = pd.read_csv(
    "https://raw.githubusercontent.com/openkinome/kinoml/refs/heads/master/kinoml/data/kinomescan/journal.pone.0181585.s004.csv"
)

df_pkis_rev = df_pkis2.set_index("Smiles").iloc[:, 6:]
# df_pkis_rev.index

### DRUG MODEL ###

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
    outputs = model_drug(**drug_tokens, output_hidden_states=True)

for layer in outputs.hidden_states:
    print(layer.shape)

mx_similarity = generate_similarity_matrix(outputs.pooler_output)

torch.allclose(mx_similarity, mx_similarity.T)
torch.all(torch.diag(mx_similarity) == 1.0000)

# cosi = torch.nn.CosineSimilarity(dim=1)
# cosi(outputs.pooler_output, outputs.pooler_output).shape

mx_norm = outputs.pooler_output / outputs.pooler_output.norm(dim=1, p=2, keepdim=True)
mx_dotprod = mx_norm @ mx_norm.T


def generate_similarity_matrix(
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

    return mx_dotprod


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

    mx_laplacian = sparse.csgraph.laplacian(csgraph=mx_adjacency, normed=True)
    return mx_laplacian


dir(outputs)
outputs.last_hidden_state.squeeze().shape
outputs.last_hidden_state.shape
len(outputs.hidden_states)

torch.allclose(mx_dotprod, mx_dotprod.T)

### KINASE MODEL ###

model_kinase_name = "facebook/esm2_t6_8M_UR50D"

model_kinase = AutoModel.from_pretrained(
    model_kinase_name,
    device_map="auto",
)
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

type(model_kinase)
isinstance(model_kinase, torch.nn.Module)

list_keep = ["encoder", "embeddings"]
set_drop = set()
for k, v in model_kinase.state_dict().items():
    if k.split(".")[0] not in list_keep:
        set_drop.add(k.split(".")[0])
for drop in set_drop:
    delattr(model_kinase, drop)


for k, v in model_kinase.state_dict().items():
    print(k)
    print(v.shape)
    print()
