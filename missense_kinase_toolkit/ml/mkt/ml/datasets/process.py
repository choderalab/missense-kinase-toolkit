import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd
from mkt.ml.constnts import KinaseGroupSource
from mkt.ml.utils import get_repo_root, rgetattr
from mkt.schema.io_utils import deserialize_kinase_dict
from pydantic import BaseModel

logger = logging.getLogger(__name__)


DICT_KINASE = deserialize_kinase_dict()


class DatasetConfig(BaseModel):
    name: str
    """Name of the dataset."""
    url: str | None
    """URL to the dataset file."""
    col_drug: str
    """Column name for drug identifiers in the dataset."""
    col_kinase: str
    """Column name for kinase identifiers in the dataset."""
    col_y: str
    """Column name for the target variable in the dataset."""
    attr_group: KinaseGroupSource = KinaseGroupSource.consensus
    """Attribute to be used to group kinases. If None, no grouping is applied."""
    bool_save: bool = True
    """Whether to save the processed dataset to a CSV file."""
    df: pd.DataFrame | None = None
    """DataFrame containing the dataset."""


class PKIS2Config(DatasetConfig):
    """Configuration for the PKIS2 dataset."""

    name: str = "PKIS2"
    url: str = (
        "https://raw.githubusercontent.com/openkinome/kinoml/refs/heads/master/kinoml/data/kinomescan/journal.pone.0181585.s004.csv"
    )
    col_drug: str = "Smiles"
    col_kinase: str = "Kinase"
    col_y: str = "Percent Displacement"


class DavisConfig(DatasetConfig):
    """Configuration for the Davis dataset."""

    name: str = "DAVIS"
    url: str | None = None  # URL will be set in the DavisDataset class
    col_drug: str = "Drug"
    col_kinase: str = "Target_ID"
    col_y: str = "Y"


@dataclass
class ProcessDataset(ABC):
    """DataSet class for handling dataset configurations."""

    bool_save: bool = True
    """Whether to save the processed dataset to a CSV file."""
    df: pd.DataFrame | None = None
    """DataFrame containing the dataset."""

    def __post_init__(self):
        """Post-initialization method to load the dataset."""
        self.df = self.process()
        self.df = self.add_klifs_column(self.df)
        self.df = self.add_kinase_group_column(self.df)

        if self.bool_save:
            self.save_data2csv()

    @abstractmethod
    def process(self):
        """Process the dataset."""
        ...

    def save_data2csv(self) -> None:
        """Save the processed dataset to a CSV file."""
        if self.df is None:
            logger.error("DataFrame is None. Cannot save to CSV.")
            return
        filepath = path.join(get_repo_root(), f"data/{self.name.lower()}_processed.csv")
        self.df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")

    def add_klifs_column(self) -> pd.DataFrame:
        """Add KLIFS column to the DataFrame."""
        df = self.df.copy()
        df["klifs"] = df[self.col_kinase].apply(
            lambda x: rgetattr(DICT_KINASE.get(x, None), "klifs.pocket_seq")
        )
        return df

    def add_kinase_group_column(self) -> pd.DataFrame:
        """Add kinase group column to the DataFrame."""
        df = self.df.copy()
        if self.attr_group.value is None:
            df["group"] = df[self.col_kinase].apply(
                lambda x: (
                    DICT_KINASE.get(x, None).adjudicate_group()
                    if DICT_KINASE.get(x, None)
                    else None
                )
            )
        else:
            df["group"] = df[self.col_kinase].apply(
                lambda x: rgetattr(
                    DICT_KINASE.get(x, None), self.attr_group.value, None
                )
            )
        return df


@dataclass
class PKIS2Dataset(ProcessDataset, PKIS2Config):
    """PKIS2 dataset processing class."""

    name: str = "PKIS2"
    """Name of the dataset."""
    url: str = (
        "https://raw.githubusercontent.com/openkinome/kinoml/refs/heads/master/kinoml/data/kinomescan/journal.pone.0181585.s004.csv"
    )
    """URL to the PKIS2 dataset CSV file."""
    col_drug: str = "Smiles"
    """Column name for drug SMILES in the dataset."""
    col_kinase: str = "Kinase"
    """Column name for kinase in the dataset."""
    col_y: str = "Percent Displacement"
    """Column name for the target variable in the dataset."""

    def __post_init__(self):
        """Post-initialization method to load the dataset."""
        super().__post_init__()
        self.df = self.process()
        self.df = self.add_klifs_column(self.df)

    def process(self) -> pd.DataFrame:
        """Process the PKIS2 dataset."""
        df = pd.read_csv(self.url)

        # first 7 columns are metadata, rest are kinase targets
        df_pivot = df.iloc[:, 7:]

        df_pivot.index = df[self.col_drug]

        df_melt = df_pivot.reset_index().melt(
            id_vars=self.col_drug,
            var_name=self.col_kinase,
            value_name=self.col_y,
        )

        return df_melt


@dataclass
class DavisDataset(ProcessDataset, DavisConfig):
    """Davis dataset processing class."""

    def __post_init__(self):
        """Post-initialization method to load the dataset."""
        super().__post_init__()

    def process(self) -> pd.DataFrame:
        """Process the Davis dataset."""
        col_davis_drug = "Drug"
        col_davis_target = "Target_ID"
        col_davis_y = "Y"
        col_davis_y_transformed = "Y_trans"

        data_davis = DTI(name="DAVIS")
        data_davis.harmonize_affinities("mean")
        df_davis = data_davis.get_data()
        df_davis_pivot = df_davis.pivot(
            index=col_davis_drug, columns=col_davis_target, values=col_davis_y
        )
        df_davis[col_davis_y_transformed] = convert_to_percentile(df_davis[col_davis_y])
        temp = convert_from_percentile(
            df_davis[col_davis_y_transformed], df_davis[col_davis_y]
        )
