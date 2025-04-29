# from collections import defaultdict
from enum import Enum

from mkt.ml.datasets.pkis2 import PKIS2CrossValidation, PKIS2KinaseSplit
from mkt.ml.models.pooling import CombinedPoolingModel
from strenum import StrEnum

# DICT_DATASET = {
#     "pkis2": PKIS2Dataset,
# }
# """dict[str, type]: Dictionary mapping dataset names to their respective classes."""
# DICT_DATASET = defaultdict(None, DICT_DATASET)

# DICT_MODELS = {
#     "pooling": CombinedPoolingModel,
# }
# """dict[str, type]: Dictionary mapping model names to their respective classes."""
# DICT_MODELS = defaultdict(None, DICT_MODELS)


class DataSet(Enum):
    """Enum for dataset names."""

    PKIS2_Kinase_Split = PKIS2KinaseSplit
    PKIS2_CV_Split = PKIS2CrossValidation


class ModelType(Enum):
    """Enum for model types."""

    pooling = CombinedPoolingModel


class DrugModel(StrEnum):
    """StrEnum for drug models."""

    CHEMBERTA_MTR = "DeepChem/ChemBERTa-77M-MTR"
    CHEMBERTA_MLM = "DeepChem/ChemBERTa-77M-MLM"


class KinaseModel(StrEnum):
    """StrEnum for kinase models."""

    ESM2_T6_8M = "facebook/esm2_t6_8M_UR50D"
    ESM2_T12_35M = "facebook/esm2_t12_35M_UR50D"
    ESM2_T30_150M = "facebook/esm2_t30_150M_UR50D"
    ESM2_T33_650M = "facebook/esm2_t33_650M_UR50D"
    ESM2_T36_3B = "facebook/esm2_t36_3B_UR50D"
    ESM2_T48_15B = "facebook/esm2_t48_15B_UR50D"
