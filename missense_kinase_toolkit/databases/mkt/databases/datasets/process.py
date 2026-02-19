import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from os import path

import pandas as pd
from mkt.databases import config
from mkt.databases.aligners import ClustalOmegaAligner
from mkt.databases.datasets.constants import KinaseGroupSource
from mkt.databases.datasets.discoverx import DiscoverXInfoGenerator
from mkt.databases.log_config import configure_logging
from mkt.schema.io_utils import deserialize_kinase_dict, get_repo_root
from mkt.schema.utils import rgetattr
from pydantic import BaseModel, Field
from rdkit import Chem
from tqdm import tqdm

logger = logging.getLogger(__name__)


DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE")

configure_logging()

try:
    config.set_request_cache(path.join(get_repo_root(), "requests_cache.sqlite"))
except Exception as e:
    logger.warning(f"Failed to set request cache, using current directory: {e}")
    config.set_request_cache(path.join(".", "requests_cache.sqlite"))

obj_discoverx = DiscoverXInfoGenerator()
DICT_DISCOVERX = obj_discoverx.dict_discoverx_info


DICT_PROCESS_STRATEGIES = {
    # align only KD region with KLIFS residues
    "kd_klifs_aligned": {
        "dict_unaligned": {
            k1: {
                k2: v2
                for k2, v2 in v1.dict_construct_sequences.items()
                if k2 not in ["kd_pre", "kd_post"]
            }
            for k1, v1 in DICT_DISCOVERX.items()
            if (
                (v1.dict_construct_sequences is not None)
                and (v1.bool_has_klifs)
                and (v1.bool_has_kd)
                and (
                    (v1.bool_mutations_in_kd_region is None)
                    or (v1.bool_mutations_in_kd_region)
                )
            )
        },
        "list_region_include": None,
        "list_region_align": ["kd_start", "kd_end", ":", "intra", "inter"],
    },
    # align only KLIFS region (no KD)
    "klifs_region_aligned": {
        "dict_unaligned": {
            k1: {k2: v2 for k2, v2 in v1.dict_construct_sequences.items()}
            for k1, v1 in DICT_DISCOVERX.items()
            if (
                (v1.dict_construct_sequences is not None)
                and (v1.bool_has_klifs)
                and (
                    (v1.bool_mutations_in_klifs_region is None)
                    or (v1.bool_mutations_in_klifs_region)
                )
            )
        },
        "list_region_include": DICT_KINASE[
            list(DICT_KINASE.keys())[0]
        ].KLIFS2UniProtSeq.keys(),
        "list_region_align": [":", "intra", "inter"],
    },
    # align only KLIFS residues (no KD or alignment of flanking regions)
    "klifs_residues_only": {
        "dict_unaligned": {
            k1: {k2: v2 for k2, v2 in v1.dict_construct_sequences.items()}
            for k1, v1 in DICT_DISCOVERX.items()
            if (
                (v1.dict_construct_sequences is not None)
                and (v1.bool_has_klifs)
                and (
                    (v1.bool_mutations_in_klifs_residues is None)
                    or (v1.bool_mutations_in_klifs_residues)
                )
            )
        },
        "list_region_include": [
            i
            for i in DICT_KINASE[list(DICT_KINASE.keys())[0]].KLIFS2UniProtSeq.keys()
            if all([j not in i for j in [":", "intra", "inter"]])
        ],
        "list_region_align": [":", "intra", "inter"],
    },
}


class DatasetConfig(BaseModel):
    name: str
    """str: Name of the dataset."""
    url_main: str
    """str: URL to the dataset file."""
    col_drug_input: str
    """str: Column name for drug input in the dataset."""
    col_drug_name: str | None
    """str: Column name for drug name in the dataset."""
    col_kinase_name: str
    """str: Column name for kinase name in the dataset."""
    col_y: str
    """str: Column name for the target variable in the dataset."""
    url_supp_drug: str | None = None
    """str: URL to the supplementary dataset file."""
    attr_group: KinaseGroupSource = KinaseGroupSource.consensus
    """KinaseGroupSource: Attribute to be used to group kinases. If None, no grouping is applied."""
    bool_save: bool = False
    """bool: Whether to save the processed dataset to a CSV file."""
    bool_isomeric: bool = False
    """bool: Whether to use isomeric SMILES for drug identifiers."""
    bool_offset: bool = True
    """bool: Whether to use 1-based indexing (True) or 0-based indexing (False). Default is True."""

    class Config:
        arbitrary_types_allowed = True


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
        self.df = self.drop_na_rows()

        self.df = self.add_kinase_group_column()
        self.df = self.add_wt_annotation()
        self.df = self.add_source_column()
        self.df = self.apply_smiles_standardization()

        self.df = self.add_construct_unaligned()
        for strategy in DICT_PROCESS_STRATEGIES:
            self.df = self.add_aligned_sequence(str_in=strategy)

        self.df = self.standardize_colnames()

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

    def drop_na_rows(self) -> pd.DataFrame:
        """Drop rows with no mapping to DICT_DISCOVERX."""
        df = self.df.copy()
        df = df[
            df[self.col_kinase_name].apply(lambda x: x in DICT_DISCOVERX)
        ].reset_index(drop=True)
        return df

    def save_data2csv(self) -> None:
        """Save the processed dataset to a CSV file."""
        if self.df is None:
            logger.error("DataFrame is empty. Cannot save to CSV.")
            return
        filepath = path.join(
            get_repo_root(), f"data/{self.name.lower()}_data_processed.csv"
        )
        self.df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")

    @staticmethod
    def convert_discoverx_to_kinase(str_in) -> pd.DataFrame:
        return rgetattr(DICT_DISCOVERX.get(str_in, None), "key")

    def add_kinase_group_column(self) -> pd.DataFrame:
        """Add kinase group column to the DataFrame."""
        # TODO: handle lipid kinases separately
        df = self.df.copy()
        if self.attr_group.value is None:
            df["group_consensus"] = df[self.col_kinase_name].apply(
                lambda x: (
                    DICT_KINASE[self.convert_discoverx_to_kinase(x)].adjudicate_group()
                    if self.convert_discoverx_to_kinase(x)
                    else None
                )
            )
        else:
            suffix = self.attr_group.value.split(".")[0]
            df[f"group_{suffix}"] = df[self.col_kinase_name].apply(
                lambda x: (
                    rgetattr(
                        DICT_KINASE[self.convert_discoverx_to_kinase(x)],
                        self.attr_group.value,
                        None,
                    )
                    if self.convert_discoverx_to_kinase(x)
                    else None
                )
            )
        return df

    def add_wt_annotation(self) -> pd.DataFrame:
        """Add a wild-type annotation column to the DataFrame."""
        df = self.df.copy()
        df["is_wt"] = df[self.col_kinase_name].apply(
            lambda x: DICT_DISCOVERX[x].bool_wt
        )
        return df

    def standardize_colnames(self) -> pd.DataFrame:
        """Standardize column names to lower case."""
        df = self.df.copy()
        df = df.rename(
            columns={
                self.col_drug_input: "smiles",
                self.col_kinase_name: "kinase_name",
                self.col_y: "y",
            },
        )
        return df

    def standardize_smiles(self, smiles: str) -> str:
        """Standardize SMILES strings.

        Parameters:
        -----------
        smiles : str
            Input SMILES string.

        Returns:
        --------
        str
            Standardized SMILES string.
        """

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
        df[self.col_drug_input] = df[self.col_drug_input].apply(self.standardize_smiles)
        return df

    def add_construct_unaligned(self) -> pd.DataFrame:
        """Add unaligned construct sequence column to the DataFrame."""
        list_idx = self.df[self.col_kinase_name].unique().tolist()
        dict_unaligned = {
            k: v.seq_refseq[v.return_index(v.idx_start) : v.idx_stop]
            for k, v in DICT_DISCOVERX.items()
            if k in list_idx and v.dict_construct_sequences is not None
        }
        df = self.df.copy()
        df["seq_construct_unaligned"] = df[self.col_kinase_name].apply(
            lambda x: dict_unaligned[x] if x in dict_unaligned else None
        )
        return df

    @staticmethod
    def align_deletion_using_wt(str_wt, str_del):
        """Align a deletion sequence to the wild-type sequence by inserting gaps.

        Parameters:
        -----------
        str_wt : str
            Wild-type sequence.
        str_del : str
            Deletion sequence.

        Returns:
        --------
        str
            Aligned deletion sequence with gaps inserted.
        """
        idx_del = 0
        str_temp = ""
        for i in str_wt:
            if i == "-":
                str_temp += i
            else:
                str_temp += str_del[idx_del]
                idx_del += 1

        assert len(str_wt) == len(str_temp)
        assert str_del in str_temp

        return str_temp

    def add_aligned_sequence(self, str_in: str) -> pd.DataFrame:
        """Add aligned sequence column to the DataFrame.

        Parameters:
        -----------
        str_in : str
            Strategy for processing sequences; key for DICT_PROCESS_STRATEGIES.
        """
        assert str_in in DICT_PROCESS_STRATEGIES, (
            f"Strategy {str_in} not recognized. "
            f"Available strategies: {list(DICT_PROCESS_STRATEGIES.keys())}"
        )

        list_idx = self.df[self.col_kinase_name].unique().tolist()

        dict_unaligned = DICT_PROCESS_STRATEGIES[str_in]["dict_unaligned"]
        list_include_region = DICT_PROCESS_STRATEGIES[str_in]["list_region_include"]
        list_align_region = DICT_PROCESS_STRATEGIES[str_in]["list_region_align"]

        dict_unaligned = {k: v for k, v in dict_unaligned.items() if k in list_idx}
        if list_include_region is not None:
            dict_unaligned = {
                k1: {k2: v2 for k2, v2 in v1.items() if k2 in list_include_region}
                for k1, v1 in dict_unaligned.items()
            }
        dict_aligned = deepcopy(dict_unaligned)

        list_region = list(dict_unaligned[list(dict_unaligned.keys())[0]].keys())
        for region in tqdm(
            list_region, desc=f"Aligning sequence regions using {str_in} strategy"
        ):
            if any([i in region for i in list_align_region]):
                dict_temp_region = {
                    k: (
                        v[region]
                        if ((v[region] is not None) and (v[region] != ""))
                        else None
                    )
                    for k, v in dict_unaligned.items()
                }

                # perform MSA
                # if seq not None and doesn't contain any deletions
                list_key_not_none = [
                    k
                    for k, v in dict_temp_region.items()
                    if ((v is not None) and ("-" not in v))
                ]
                list_val_not_none = [
                    i
                    for i in dict_temp_region.values()
                    if ((i is not None) and ("-" not in i))
                ]

                # if seq not None and does contain deletions
                list_key_del = [
                    k
                    for k, v in dict_temp_region.items()
                    if ((v is not None) and ("-" in v))
                ]
                list_val_del = [
                    i
                    for i in dict_temp_region.values()
                    if ((i is not None) and ("-" in i))
                ]

                # no sequences in region
                if len(list_val_not_none) == 0:
                    # remove region and move to next iteration
                    dict_aligned = {
                        k1: {k2: v2 for k2, v2 in v1.items() if k2 != region}
                        for k1, v1 in dict_aligned.items()
                    }
                    continue
                # more than one sequence in region
                if len(list_val_not_none) > 1:
                    alignment = ClustalOmegaAligner(list_val_not_none)
                    list_alignments = alignment.list_alignments
                # one sequence in region: len(list_val_not_none) == 1
                else:
                    list_alignments = list_val_not_none

                dict_replace = dict(zip(list_key_not_none, list_alignments))

                # deletions cannot be processed using ClustalOmega
                if len(list_val_del) > 0:
                    list_key_wt = [i.split("(")[0] for i in list_key_del]
                    list_val_wt = [dict_replace[i] for i in list_key_wt]
                    list_val_del_replace = [
                        self.align_deletion_using_wt(i, j)
                        for i, j in zip(list_val_wt, list_val_del)
                    ]
                    dict_replace.update(dict(zip(list_key_del, list_val_del_replace)))

                # replace values in dictionaries
                str_missing = "-" * len(list_alignments[0])
                dict_temp_region = {
                    k: dict_replace[k] if k in dict_replace else str_missing
                    for k, v in dict_temp_region.items()
                }

                dict_aligned = {
                    k1: {
                        k2: dict_temp_region[k1] if k2 == region else v2
                        for k2, v2 in v1.items()
                    }
                    for k1, v1 in dict_aligned.items()
                }

        # add aligned sequences to dataframe
        dict_full_seq = {k: "".join(v.values()) for k, v in dict_aligned.items()}
        # double check all sequences are same length
        assert all(
            [
                len(v) == len(list(dict_full_seq.values())[0])
                for v in dict_full_seq.values()
            ]
        )
        df = self.df.copy()
        df["seq_" + str_in] = df[self.col_kinase_name].apply(
            lambda x: dict_full_seq[x] if x in dict_full_seq else None
        )

        return df


def generate_ridgeline_df(df_in: pd.DataFrame, source: str):
    list_ids = df_in["kinase_name"].unique()

    list_frac_refseq = [
        (
            (DICT_DISCOVERX[x].idx_stop - DICT_DISCOVERX[x].idx_start + 1)
            / len(DICT_DISCOVERX[x].seq_refseq)
            if DICT_DISCOVERX[x].dict_construct_sequences is not None
            else None
        )
        for x in list_ids
    ]

    list_family = [
        (
            (
                "Lipid"
                if DICT_KINASE[DICT_DISCOVERX[x].key].is_lipid_kinase()
                else DICT_KINASE[DICT_DISCOVERX[x].key].adjudicate_group()
            )
            if DICT_DISCOVERX[x].key is not None
            else None
        )
        for x in list_ids
    ]

    df_out = pd.DataFrame(
        {
            "kinase_name": list_ids,
            "family": list_family,
            "fraction_construct": list_frac_refseq,
        }
    )

    df_out["source"] = source

    return df_out


def generate_stacked_barchart_df(df_in: pd.DataFrame, source: str):
    list_ids = df_in["kinase_name"].unique()

    list_bool_uniprot2refseq = [
        DICT_DISCOVERX[x].seq_refseq == DICT_DISCOVERX[x].seq_uniprot for x in list_ids
    ]

    list_family = [
        (
            (
                "Lipid"
                if DICT_KINASE[DICT_DISCOVERX[x].key].is_lipid_kinase()
                else DICT_KINASE[DICT_DISCOVERX[x].key].adjudicate_group()
            )
            if DICT_DISCOVERX[x].key is not None
            else None
        )
        for x in list_ids
    ]

    df_temp = pd.DataFrame(
        {
            "kinase_name": list_ids,
            "bool_uniprot2refseq": list_bool_uniprot2refseq,
            "family": list_family,
        }
    )

    df_out = pd.DataFrame(
        df_temp[["family", "bool_uniprot2refseq"]].value_counts()
    ).reset_index()

    df_out["source"] = source

    return df_out
