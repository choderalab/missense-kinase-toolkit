import pandas as pd
from mkt.databases.datasets.process import (
    DatasetConfig,
    ProcessDataset,
)


class PKIS2Config(DatasetConfig):
    """Configuration for the PKIS2 dataset."""

    name: str = "PKIS2"
    url_main: str = (
        "https://raw.githubusercontent.com/openkinome/kinoml/refs/heads/master/kinoml/data/kinomescan/journal.pone.0181585.s004.csv"
    )
    col_drug_input: str = "Smiles"
    # no conventional names supplied - can try ChemBL later
    col_drug_name: str | None = None
    col_kinase_name: str = "Kinase"
    col_y: str = "% Inhibition"


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
        df = pd.read_csv(self.url_main)

        # first 7 columns are metadata, rest are kinase targets
        df_pivot = df.iloc[:, 7:]

        df_pivot.index = df[self.col_drug_input]

        df_melt = df_pivot.reset_index().melt(
            id_vars=self.col_drug_input,
            var_name=self.col_kinase_name,
            value_name=self.col_y,
        )

        return df_melt

    def add_source_column(self) -> pd.DataFrame:
        """Add a PKIS2 name column to the DataFrame."""
        df = self.df.copy()
        df["source"] = self.name
        return df


# drop values that cannot be matched to protein kinase sequence
LIST_KM_ATP_DROP = [
    "PI3-KINASE-ALPHA",  # lipid kinase
    "PI3-KINASE-DELTA",  # lipid kinase
    "PI3-KINASE-GAMMA",  # lipid kinase
    "PI4-K-BETA",  # lipid kinase
    "SPHK1",  # lipid kinase
    "SPHK2",  # lipid kinase
    "AMP-A1B1G1",  # B1G1 subunits not kinase entries
    "AMP-A2B1G1",  # B1G1 subunits not kinase entries
    "CK2",  # not sure if CSNK2A1 or CSNK2A2; catalog not available
]

# dictionary of {string match : string replace} for exact matches
DICT_KM_ATP_EXACT = {
    "ARK5": "NUAK1",
    "CK1": "CK1a",
    "CRAF": "RAF1",
    "LRRK": "LRRK2",
    "MEK1": "MAP2K1",
    "PAR-1B-ALPHA": "MARK2",
    "PKC-BETA1": "PKC-B",
    "PRAK": "MAPKAPK5",
    "PTK5": "FRK",
    "SNF1LK2": "QIK",
}

# dictionary of {string match : string replace} for partial matches
DICT_KM_ATP_PARTIAL = {
    "AURORA": "AURK",
    "ALPHA": "A",
    "BETA": "B",
    "DELTA": "D",
    "EPSILON": "E",
    "GAMMA": "G",
    "-ETA": "H",
    "IOTA": "I",
    "THETA": "T",
}


def read_xlsx_file(
    str_path: str = "../../../data/3. PKIS Nanosyn Assay Heatmaps.xlsx",
    str_sheet: str = "Assay and Panel information",
) -> pd.DataFrame:
    df = pd.read_excel(str_path, sheet_name=str_sheet)
    return df
