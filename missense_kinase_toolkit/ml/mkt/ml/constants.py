from collections import defaultdict

from mkt.ml.datasets.pkis2 import PKIS2Dataset
from mkt.mk.models.pooling import CombinedPoolingModel

DICT_DATASET = {
    "pkis2": PKIS2Dataset,
}
"""dict[str, type]: Dictionary mapping dataset names to their respective classes."""
DICT_DATASET = defaultdict(None, DICT_DATASET)

DICT_MODELS = {
    "pooling": CombinedPoolingModel,
}
"""dict[str, type]: Dictionary mapping model names to their respective classes."""
DICT_MODELS = defaultdict(None, DICT_MODELS)

LIST_DRUG_MODELS = [
    "DeepChem/ChemBERTa-77M-MTR",
    
]

LIST_KINASE_MODELS = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
]

