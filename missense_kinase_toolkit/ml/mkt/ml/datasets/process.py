import logging
from abc import ABC, abstractmethod
from os import path

import pandas as pd
from mkt.ml.constants import KinaseGroupSource
from mkt.schema.io_utils import deserialize_kinase_dict, get_repo_root
from mkt.schema.utils import rgetattr
from pydantic import BaseModel, Field
from rdkit import Chem

logger = logging.getLogger(__name__)


DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE")


class DatasetConfig(BaseModel):
    name: str
    """Name of the dataset."""
    url_main: str | None
    """URL to the dataset file."""
    url_supp: str | None
    """URL to the supplementary dataset file."""
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
    bool_isomeric: bool = False
    """Whether to use isomeric SMILES for drug identifiers."""

    class Config:
        arbitrary_types_allowed = True


class PKIS2Config(DatasetConfig):
    """Configuration for the PKIS2 dataset."""

    name = "PKIS2"
    url_main = "https://raw.githubusercontent.com/openkinome/kinoml/refs/heads/master/kinoml/data/kinomescan/journal.pone.0181585.s004.csv"
    col_drug = "Smiles"
    col_kinase = "Kinase"
    col_y = "% Inhibition"


class DavisConfig(DatasetConfig):
    """Configuration for the Davis dataset."""

    name = "Davis"
    url_main = None  # URL will be set in the DavisDataset class
    col_drug = "Drug"
    col_kinase = "Target_ID"
    col_y = "Y"


class ProcessDataset(ABC):
    """DataSet class for handling dataset configurations."""

    df: pd.DataFrame = Field(
        default=None, exclude=True, validate_default=False, repr=False
    )
    """DataFrame to hold the processed dataset."""

    def __init__(self, **kwargs):
        """Initialize the dataset processor."""
        self.__dict__.update(kwargs)

        self.df = self.process()
        self.df = self.add_klifs_column()
        self.df = self.add_kd_column()
        self.df = self.add_kinase_group_column()
        # TODO: better map Davis kinase names (e.g., mutant, phos, domains, etc.)
        # self.df = self.drop_na_rows()
        self.df = self.standardize_colnames()
        self.df = self.add_source_column()
        self.df = self.apply_smiles_standardization()

        if getattr(self, "bool_save", True):
            self.save_data2csv()

    @abstractmethod
    def process(self):
        """Process the dataset."""
        ...

    @abstractmethod
    def add_source_column(self) -> pd.DataFrame:
        """Add a source name column to the DataFrame."""
        ...

    def save_data2csv(self) -> None:
        """Save the processed dataset to a CSV file."""
        if self.df is None:
            logger.error("DataFrame is None. Cannot save to CSV.")
            return
        filepath = path.join(
            get_repo_root(), f"data/{self.name.lower()}_data_processed.csv"
        )
        self.df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")

    def add_klifs_column(self) -> pd.DataFrame:
        """Add KLIFS column to the DataFrame."""
        df = self.df.copy()
        df["klifs_seq"] = df[self.col_kinase].apply(
            lambda x: rgetattr(DICT_KINASE.get(x, None), "klifs.pocket_seq")
        )
        return df

    def add_kd_column(self) -> pd.DataFrame:
        """Add kinase domain column to the DataFrame."""
        df = self.df.copy()
        df["kd_seq"] = df[self.col_kinase].apply(
            lambda x: (
                DICT_KINASE.get(x, None).adjudicate_kd_sequence()
                if DICT_KINASE.get(x, None)
                else None
            )
        )
        return df

    def add_kinase_group_column(self) -> pd.DataFrame:
        """Add kinase group column to the DataFrame."""
        df = self.df.copy()
        if self.attr_group.value is None:
            df["group_consensus"] = df[self.col_kinase].apply(
                lambda x: (
                    DICT_KINASE.get(x, None).adjudicate_group()
                    if DICT_KINASE.get(x, None)
                    else None
                )
            )
        else:
            suffix = self.attr_group.value.split(".")[0]
            df[f"group_{suffix}"] = df[self.col_kinase].apply(
                lambda x: rgetattr(
                    DICT_KINASE.get(x, None), self.attr_group.value, None
                )
            )
        return df

    def drop_na_rows(self) -> pd.DataFrame:
        """Drop rows with NA values in the KLIFS column."""
        df = self.df.copy()
        df = df.dropna(subset=["klifs"]).reset_index(drop=True)
        return df

    def standardize_colnames(self) -> pd.DataFrame:
        """Standardize column names to lower case."""
        df = self.df.copy()
        df = df.rename(
            columns={
                self.col_drug: "smiles",
                self.col_kinase: "kinase_name",
                self.col_y: "y",
            },
        )
        return df

    def standardize_smiles(self, smiles: str) -> str:
        """Standardize SMILES strings."""

        if smiles is None or pd.isna(smiles):
            return None

        mol = Chem.MolFromSmiles(smiles)
        if self.bool_isomeric:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            return Chem.MolToSmiles(mol, isomericSmiles=False)

    def apply_smiles_standardization(self) -> pd.DataFrame:
        """Apply SMILES standardization to the DataFrame."""
        df = self.df.copy()
        df["smiles"] = df["smiles"].apply(self.standardize_smiles)
        return df


class PKIS2Dataset(PKIS2Config, ProcessDataset):
    """PKIS2 dataset processing class."""

    def __init__(self, **kwargs):
        """Initialize PKIS2 dataset."""
        super().__init__(**kwargs)

        config_dict = PKIS2Config().model_dump()
        config_dict.update(kwargs)

        ProcessDataset.__init__(self, **config_dict)

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

    def add_source_column(self) -> pd.DataFrame:
        """Add a PKIS2 name column to the DataFrame."""
        df = self.df.copy()
        df["source"] = self.name
        return df


class DavisDataset(DavisConfig, ProcessDataset):
    """Davis dataset processing class."""

    def __init__(self, **kwargs):
        """Initialize Davis dataset."""
        super().__init__(**kwargs)

        config_dict = DavisConfig().model_dump()
        config_dict.update(kwargs)

        ProcessDataset.__init__(self, **config_dict)

    def process(self) -> pd.DataFrame:
        """Process the Davis dataset."""
        from tdc.multi_pred import DTI

        data = DTI(name=self.name)
        data.harmonize_affinities("mean")

        df = data.get_data()
        df_keep = df[[self.col_drug, self.col_kinase, self.col_y]]

        return df_keep

    def add_source_column(self) -> pd.DataFrame:
        """Add a Davis name column to the DataFrame."""
        df = self.df.copy()
        df["source"] = self.name.title()
        return df
