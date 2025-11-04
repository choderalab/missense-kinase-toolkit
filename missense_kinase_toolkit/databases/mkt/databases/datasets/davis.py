import logging

import numpy as np
import pandas as pd
from mkt.databases.chembl import ChEMBLMolecule, return_chembl_id
from mkt.databases.datasets.process import DatasetConfig, ProcessDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

tqdm.pandas()


DICT_ID2CHEMBL = {
    "GSK-1838705A": "CHEMBL464552",
    "MLN-120B": "CHEMBL608154",
    "PD-173955": "CHEMBL386051",
    "PP-242": "CHEMBL1241674",
}
"""Dictionary of manual ChEMBL ID mappings for Davis dataset drugs."""


class DavisConfig(DatasetConfig):
    """Configuration for the Davis dataset."""

    name: str = "Davis"
    url_main: str = (
        "https://raw.githubusercontent.com/choderalab/missense-kinase-toolkit/main/data/41587_2011_BFnbt1990_MOESM5_ESM.xls"
    )
    url_supp_drug: str = (
        "https://raw.githubusercontent.com/choderalab/missense-kinase-toolkit/main/data/41587_2011_BFnbt1990_MOESM4_ESM.xls"
    )
    col_drug_input: str = "smiles"
    col_drug_name: str | None = "drug_name"
    col_kinase_name: str = "discoverx_gene_symbol"
    col_y: str = "pKd"


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
        df = pd.read_excel(self.url_main, sheet_name=0)
        df_drug = pd.read_excel(self.url_supp_drug, sheet_name=0)

        # pre-process main dataframe
        df = df.iloc[:, 2:]
        df = df.melt(id_vars=["Kinase"], var_name="drug_name", value_name="Kd")
        df = df.rename(columns={"Kinase": "discoverx_gene_symbol"})
        # convert nM to pKd in µM
        df["pKd"] = df["Kd"].fillna(10000).apply(lambda x: -np.log10(x / 1000))
        df = df.drop(columns="Kd")

        # add ChEMBL info to drug dataframe
        df_drug = self.add_chembl_info(df_drug)

        # merge dataframes
        df_merge = df.merge(
            df_drug[["Compound Name", "pref_name", "smiles"]],
            how="left",
            left_on="drug_name",
            right_on="Compound Name",
        )
        df_merge = df_merge.drop(columns=["drug_name", "Compound Name"])
        df_merge = df_merge.rename(columns={"pref_name": "drug_name"})
        df_merge = df_merge[[i for i in df_merge.columns if i != "pKd"] + ["pKd"]]

        return df_merge

    def add_source_column(self) -> pd.DataFrame:
        """Add a Davis name column to the DataFrame."""
        df = self.df.copy()
        df["source"] = self.name.title()
        return df

    @staticmethod
    def add_chembl_info(df_in: pd.DataFrame) -> pd.DataFrame:
        """Add ChEMBL preferred IDs and canonical SMILES to the Davis dataset.

        Parameters:
        -----------
        df_in : pd.DataFrame
            Input DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame with ChEMBL SMILES added.
        """
        # get list of drugs with alternative names
        list_davis_drugs = list(
            map(
                lambda x, y: x if str(y) == "nan" else y,
                df_in["Compound Name"],
                df_in["Alternative Name"],
            )
        )

        # query ChEMBL for each drug - have manually adjudicated some changes
        dict_chembl_id = {
            drug: {"source": None, "ids": []} for drug in list_davis_drugs
        }
        for drug in tqdm(list_davis_drugs, desc="Querying drugs in ChEMBL"):
            drug_rev = drug.split(" (")[0]
            chembl_id, source = return_chembl_id(drug_rev)
            dict_chembl_id[drug]["source"] = source
            dict_chembl_id[drug]["ids"].extend(chembl_id)
        dict_chembl_id_rev = {k: v["ids"][0] for k, v in dict_chembl_id.items()}
        dict_chembl_id_rev.update(DICT_ID2CHEMBL)

        # query for ChEMBL ChEMBLMolecule objects
        list_chembl_molec = [
            ChEMBLMolecule(id=v)
            for v in tqdm(dict_chembl_id_rev.values(), desc="Querying ChEBMLMolecule")
        ]
        dict_chembl_molecule = {
            k: {"chembl_id": v, "molecule": mol}
            for (k, v), mol in zip(dict_chembl_id_rev.items(), list_chembl_molec)
        }

        df_in["pref_name"] = [
            v["molecule"].adjudicate_preferred_name(k)
            for k, v in dict_chembl_molecule.items()
        ]
        df_in["smiles"] = [
            v["molecule"].return_smiles() for v in dict_chembl_molecule.values()
        ]

        return df_in


# NOT IN USE #

# figured out boundary conditions for RPS6KA4 and RPS6KA5 N-term constructs
# for key in set([i.split("_")[0] for i in davis_dataset.df.loc[davis_dataset.df["key"].apply(lambda x: "RPS6KA" in x), "key"].unique()]):
#     print(davis_dataset.df.loc[davis_dataset.df["key"].apply(lambda x: key in x), "key"].values)
#     print(davis_dataset.df.loc[davis_dataset.df["key"].apply(lambda x: key in x), "DiscoverX Gene Symbol"].values)
#     temp = davis_dataset.df.loc[davis_dataset.df["key"].apply(lambda x: key in x), "AA Start/Stop"].values
#     print(temp)
#     if len(temp) > 1:
#         if temp[0] != "Null" and temp[1] != "Null":
#             idx_start_11, idx_end_12 = [int(i[1:]) for i in temp[0].split("/")]
#             idx_start_21, idx_end_22 = [int(i[1:]) for i in temp[1].split("/")]
#             print(idx_start_21 - idx_end_12)
#     print(davis_dataset.df.loc[davis_dataset.df["key"].apply(lambda x: key in x), "kd_end"].values)
#     print()
