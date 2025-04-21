from os import path
from dataclasses import dataclass, field
from mkt.ml.utils import get_repo_root
from mkt.ml.datasets.finetune import FineTuneDataset

@dataclass
class PKIS2Dataset(FineTuneDataset):
    """PKIS2 dataset for kinase and drug interactions."""
    filepath: str = path.join(get_repo_root(), "data/pkis_data.csv")
    col_kinase_split: str = "kincore_group"
    col_yval: str = "percent_displacement"
    col_kinase: str = "klifs"
    col_drug: str = "Smiles"
    list_kinase_split: list[str] | None = field(
        default_factory=lambda: ["TK", "TKL"]
    )

    def __post_init__(self):
        """Post-initialization method to load the dataset."""
        super().__post_init__()
