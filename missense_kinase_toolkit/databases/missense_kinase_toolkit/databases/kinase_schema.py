import logging
import os
from enum import Enum, StrEnum

import pandas as pd
from pydantic import BaseModel, ValidationError, constr, model_validator

from missense_kinase_toolkit.databases.kincore import (
    align_kincore2uniprot,
    extract_pk_fasta_info_as_dict,
)
from missense_kinase_toolkit.databases.utils import get_repo_root

logger = logging.getLogger(__name__)


LIST_PFAM_KD = [
    "Protein kinase domain",
    "Protein tyrosine and serine/threonine kinase",
]


class Group(str, Enum):
    """Enum class for kinase groups."""

    AGC = "AGC"  # Protein Kinase A, G, and C families
    Atypical = "Atypical"  # Atypical protein kinases
    CAMK = "CAMK"  # Calcium/calmodulin-dependent protein kinase family
    CK1 = "CK1"  # Casein kinase 1 family
    CMGC = "CMGC"  # Cyclin-dependent kinase, Mitogen-activated protein kinase, Glycogen synthase kinase, and CDK-like kinase families
    RGC = "RGC"  # Receptor guanylate cyclase family
    STE = "STE"  # Homologs of yeast Sterile 7, Sterile 11, Sterile 20 kinases
    TK = "TK"  # Tyrosine kinase family
    TKL = "TKL"  # Tyrosine kinase-like family
    Other = "Other"  # Other protein kinases


class Family(Enum):
    """Enum class for kinase families (>=5 in KinHub)."""

    STE20 = "STE20"
    CAMKL = "CAMKL"
    CDK = "CDK"
    Eph = "Eph"
    MAPK = "MAPK"
    STKR = "STKR"
    NEK = "NEK"
    Src = "Src"
    DYRK = "DYRK"
    PKC = "PKC"
    STE11 = "STE11"
    RSK = "RSK"
    MLK = "MLK"
    GRK = "GRK"
    CK1 = "CK1"
    DMPK = "DMPK"
    STE7 = "STE7"
    PIKK = "PIKK"
    RSKb = "RSKb"
    Alpha = "Alpha"
    Tec = "Tec"
    CAMK1 = "CAMK1"
    PDGFR = "PDGFR"
    ULK = "ULK"
    DAPK = "DAPK"
    RAF = "RAF"
    RIPK = "RIPK"
    MLCK = "MLCK"
    PKA = "PKA"
    MAPKAPK = "MAPKAPK"
    RGC = "RGC"
    CDKL = "CDKL"
    MAST = "MAST"
    TSSK = "TSSK"
    ABC1 = "ABC1"
    PDHK = "PDHK"
    Jak = "Jak"
    Jakb = "Jakb"
    Other = "Other"
    Null = None


KinaseDomainName = StrEnum(
    "KinaseDomainName", {"KD" + str(idx + 1): kd for idx, kd in enumerate(LIST_PFAM_KD)}
)

UniProtSeq = constr(pattern=r"^[ACDEFGHIKLMNPQRSTVWXY]+$")
"""Pydantic model for UniProt sequence constraints."""
KLIFSPocket = constr(pattern=r"^[ACDEFGHIKLMNPQRSTVWY\-]{85}$")
"""Pydantic model for KLIFS pocket sequence constraints."""
UniProtID = constr(pattern=r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$")
"""Pydantic model for UniProt ID constraints."""


class KinHub(BaseModel):
    """Pydantic model for KinHub information."""

    kinase_name: str
    manning_name: list[str]
    xname: list[str]
    group: list[Group]
    family: list[Family]


class UniProt(BaseModel):
    """Pydantic model for UniProt information."""

    canonical_seq: UniProtSeq


class KLIFS(BaseModel):
    """Pydantic model for KLIFS information."""

    gene_name: str
    name: str
    full_name: str
    group: Group
    family: Family
    iuphar: int
    kinase_id: int
    pocket_seq: KLIFSPocket | None


class Pfam(BaseModel):
    """Pydantic model for Pfam information."""

    domain_name: KinaseDomainName
    start: int
    end: int
    protein_length: int
    pfam_accession: str
    in_alphafold: bool


class KinCore(BaseModel):
    """Pydantic model for KinCore information."""

    seq: UniProtSeq
    start: int | None
    end: int | None
    mismatch: list[int] | None


class KinaseInfo(BaseModel):
    """Pydantic model for kinase information."""

    hgnc_name: str
    uniprot_id: UniProtID
    KinHub: KinHub
    UniProt: UniProt
    KLIFS: KLIFS | None
    Pfam: Pfam | None
    KinCore: KinCore | None

    # https://stackoverflow.com/questions/68082983/validating-a-nested-model-in-pydantic
    # skip if other validation errors occur in nested models first
    @model_validator(mode="after")
    @classmethod
    def validate_uniprot_length(cls, values):
        """Validate canonical UniProt sequence length matches Pfam length if Pfam not None."""
        pfam = values.Pfam
        uniprot = values.UniProt
        if pfam is not None:
            if len(uniprot.canonical_seq) != pfam.protein_length:
                raise ValidationError(
                    "UniProt sequence length does not match Pfam protein length."
                )
        return values


# TODO: Is this necessary? Just aggregate as a list of KinaseInfo objects or dict?
class CollectionKinaseInfo(BaseModel):
    """Pydantic model for kinase information."""

    kinase_dict: dict[str, KinaseInfo]


def check_if_file_exists_then_load_dataframe(str_file: str) -> pd.DataFrame | None:
    """Check if file exists and load dataframe.

    Parameters
    ----------
    str_file : str
        File to check and load.

    Returns
    -------
    pd.DataFrame | None
        Dataframe if file exists, otherwise None.
    """
    if os.path.isfile(str_file):
        return pd.read_csv(str_file)
    else:
        logger.error(f"File {str_file} does not exist.")


def concatenate_source_dataframe(
    kinhub_df: pd.DataFrame | None = None,
    uniprot_df: pd.DataFrame | None = None,
    klifs_df: pd.DataFrame | None = None,
    pfam_df: pd.DataFrame | None = None,
    col_kinhub_merge: str | None = None,
    col_uniprot_merge: str | None = None,
    col_klifs_merge: str | None = None,
    col_pfam_merge: str | None = None,
    col_pfam_include: list[str] | None = None,
    list_domains_include: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate database dataframes on UniProt ID.

    Parameters
    ----------
    kinhub_df : pd.DataFrame | None, optional
        KinHub dataframe, by default None and will be loaded from "data" dir.
    uniprot_df : pd.DataFrame | None, optional
        UniProt dataframe, by default None and will be loaded from "data" dir.
    klifs_df : pd.DataFrame | None, optional
        KLIFS dataframe, by default None and will be loaded from "data" dir.
    pfam_df : pd.DataFrame | None, optional
        Pfam dataframe, by default None and will be loaded from "data" dir.
    col_kinhub_merge : str | None, optional
        Column to merge KinHub dataframe, by default None.
    col_uniprot_merge : str | None, optional
        Column to merge UniProt dataframe, by default None.
    col_klifs_merge : str | None, optional
        Column to merge KLIFS dataframe, by default None.
    col_pfam_merge : str | None, optional
        Column to merge Pfam dataframe, by default None.
    col_pfam_include : list[str] | None, optional
        Columns to include in Pfam dataframe, by default None.
    list_domains_include : list[str] | None, optional
        List of Pfam domains to include, by default None.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe.
    """

    # load dataframes if not provided from "data" sub-directory
    path_data = os.path.join(get_repo_root(), "data")
    if kinhub_df is None:
        kinhub_df = check_if_file_exists_then_load_dataframe(
            os.path.join(path_data, "kinhub.csv")
        )
    if uniprot_df is None:
        uniprot_df = check_if_file_exists_then_load_dataframe(
            os.path.join(path_data, "kinhub_uniprot.csv")
        )
    if klifs_df is None:
        klifs_df = check_if_file_exists_then_load_dataframe(
            os.path.join(path_data, "kinhub_klifs.csv")
        )
    if pfam_df is None:
        pfam_df = check_if_file_exists_then_load_dataframe(
            os.path.join(path_data, "kinhub_pfam.csv")
        )
    list_df = [kinhub_df, uniprot_df, klifs_df, pfam_df]
    if any([True if i is None else False for i in list_df]):
        list_df_shape = [i.shape if i is not None else None for i in list_df]
        logger.error(f"One or more dataframes are None\n{list_df_shape}")
        return None

    # columns on which to merge dataframes
    if col_kinhub_merge is None:
        col_kinhub_merge = "UniprotID"
    if col_uniprot_merge is None:
        col_uniprot_merge = "uniprot_id"
    if col_klifs_merge is None:
        col_klifs_merge = "uniprot"
    if col_pfam_merge is None:
        col_pfam_merge = "uniprot"

    # columns to include in the final dataframe
    if col_pfam_include is None:
        col_pfam_include = [
            "name",
            "start",
            "end",
            "protein_length",
            "pfam_accession",
            "in_alphafold",
        ]

    # list of Pfam domains to include
    if list_domains_include is None:
        list_domains_include = LIST_PFAM_KD

    # set indices to merge columns
    kinhub_df_merge = kinhub_df.set_index(col_kinhub_merge)
    uniprot_df_merge = uniprot_df.set_index(col_uniprot_merge)
    klifs_df_merge = klifs_df.set_index(col_klifs_merge)
    pfam_df_merge = pfam_df.set_index(col_pfam_merge)

    # filter Pfam dataframe for KD domains and columns to include
    df_pfam_kd = pfam_df_merge.loc[
        pfam_df_merge["name"].isin(LIST_PFAM_KD), col_pfam_include
    ]

    # rename "name" column in Pfam so doesn't conflict with KLIFS name
    try:
        df_pfam_kd = df_pfam_kd.rename(columns={"name": "domain_name"})
    except KeyError:
        pass

    # concat dataframes
    df_merge = pd.concat(
        [kinhub_df_merge, uniprot_df_merge, klifs_df_merge, df_pfam_kd],
        join="outer",
        axis=1,
    ).reset_index()

    return df_merge


def is_not_valid_string(str_input: str) -> bool:
    if pd.isna(str_input) or str_input == "":
        return True
    else:
        return False


def convert_to_group(str_input: str, bool_list: bool = True) -> list[Group]:
    """Convert KinHub group to Group enum.

    Parameters
    ----------
    str_input : str
        KinHub group to convert.
    bool_list : bool, optional
        Whether to return list of Group enums (e.g., converting KinHub), by default True.

    Returns
    -------
    list[Group]
        List of Group enums.
    """
    if bool_list:
        return [Group(group) for group in str_input.split(", ")]
    else:
        return Group(str_input)


def convert_str2family(str_input: str) -> Family:
    """Convert string to Family enum.

    Parameters
    ----------
    str_input : str
        String to convert to Family enum.

    Returns
    -------
    Family
        Family enum.
    """
    try:
        return Family(str_input)
    except ValueError:
        if is_not_valid_string(str_input):
            return Family.Null
        else:
            return Family.Other


def convert_to_family(
    str_input: str,
    bool_list: bool = True,
) -> Family:
    """Convert KinHub family to Family enum.

    Parameters
    ----------
    str_input : str
        String to convert to Family enum.

    Returns
    -------
    Family
        Family enum.
    """
    if bool_list:
        return [convert_str2family(family) for family in str_input.split(", ")]
    else:
        return convert_str2family(str_input)


def create_kinase_models_from_df(
    df: pd.DataFrame | None = None,
) -> dict[str, BaseModel]:
    """Create Pydantic models for kinases from dataframes.

    Parameters
    ----------
    df : pd.DataFrame | None, optional
        Dataframe with merged kinase information, by default will be None.

    Returns
    -------
    dict[str, BaseModel]
        Dictionary of HGNC name key and kinase model key.
    """

    # load dataframe if not provided
    if df is None:
        df = concatenate_source_dataframe()
    if df is None:
        logger.error("Dataframe is None. Cannot create kinase models.")
        return None

    # create KinHub model
    dict_kinase_models = {}

    # create KinCore dictionary from fasta file
    DICT_KINCORE = extract_pk_fasta_info_as_dict()

    for _, row in df.iterrows():

        id_uniprot = row["index"]
        name_hgnc = row["HGNC Name"]

        # create KinHub model
        kinhub_model = KinHub(
            kinase_name=row["Kinase Name"],
            manning_name=row["Manning Name"].split(", "),
            xname=row["xName"].split(", "),
            group=convert_to_group(row["Group"]),
            family=convert_to_family(row["Family"]),
        )

        # create UniProt model
        uniprot_model = UniProt(canonical_seq=row["canonical_sequence"])

        # create KLIFS model
        if is_not_valid_string(row["family"]):
            klifs_model = None
        else:
            if is_not_valid_string(row["pocket"]):
                pocket = None
            else:
                pocket = row["pocket"]
            klifs_model = KLIFS(
                gene_name=row["gene_name"],
                name=row["name"],
                full_name=row["full_name"],
                group=convert_to_group(row["group"], bool_list=False),
                family=convert_to_family(row["family"], bool_list=False),
                iuphar=row["iuphar"],
                kinase_id=row["kinase_ID"],
                pocket_seq=pocket,
            )

        # create Pfam model
        if row["domain_name"] not in LIST_PFAM_KD:
            pfam_model = None
        else:
            pfam_model = Pfam(
                domain_name=row["domain_name"],
                start=row["start"],
                end=row["end"],
                protein_length=row["protein_length"],
                pfam_accession=row["pfam_accession"],
                in_alphafold=row["in_alphafold"],
            )

        # create KinCore model
        if id_uniprot in DICT_KINCORE.keys():
            dict_temp = align_kincore2uniprot(
                DICT_KINCORE[id_uniprot]["seq"],
                uniprot_model.canonical_seq,
            )
            kincore_model = KinCore(**dict_temp)
        else:
            kincore_model = None

        # create KinaseInfo model
        kinase_info = KinaseInfo(
            hgnc_name=name_hgnc,
            uniprot_id=id_uniprot,
            KinHub=kinhub_model,
            UniProt=uniprot_model,
            KLIFS=klifs_model,
            Pfam=pfam_model,
            KinCore=kincore_model,
        )

        dict_kinase_models[name_hgnc] = kinase_info

    # TODO: For entries in DICT_KINCORE that are not in df, add to dict_kinase_models

    return dict_kinase_models


### NOT IN USE - USE TO GENERATE ABOVE ###

# import numpy as np
# import pandas as pd
# from itertools import chain

# # generate these in databases.ipynb
# df_kinhub = pd.read_csv("../data/kinhub.csv")
# df_klifs = pd.read_csv("../data/kinhub_klifs.csv")
# df_uniprot = pd.read_csv("../data/kinhub_uniprot.csv")
# df_pfam = pd.read_csv("../data/kinhub_pfam.csv")

# # generate list of families for kinase_schema.Family Enum
# list_family = list(chain.from_iterable(df_kinhub["Family"].apply(lambda x: x.split(", ")).tolist()))
# dict_family = {item: list_family.count(item) for item in set(list_family)}
# dict_family = {k: v for k, v in sorted(dict_family.items(), key=lambda item: item[1], reverse=True)}
# [key for key, val in dict_family.items() if val >= 5] # kinase_schema.Family; manually added Jak and JakB since Jak + JakB > 5

# # see if should sub-family list Enum object
# list_subfamily = list(chain.from_iterable(df_kinhub["SubFamily"].apply(lambda x: str(x).split(", ")).tolist()))
# dict_subfamily = {item: list_subfamily.count(item) for item in set(list_subfamily)}
# dict_subfamily = {k: v for k, v in sorted(dict_subfamily.items(), key=lambda item: item[1], reverse=True)}
# [key for key, val in dict_subfamily.items() if val >= 5] # kinase_schema.SubFamily NOT IN USE AS N=3
# df_pivot = pd.DataFrame(df_kinhub[["Family", "SubFamily"]].value_counts()).reset_index().pivot(columns="Family", index="SubFamily", values="count")
# df_pivot.loc[df_pivot.index.isin([key for key, val in dict_subfamily.items() if val >= 5]),].dropna(axis=1, how="all")

# # kinase_schema.UniProtSeq
# "".join(sorted(list(set(chain.from_iterable(df_uniprot["canonical_sequence"].apply(lambda x: list(x)).tolist())))))

# # kinase_schema.KLIFSPocket
# "".join(sorted(list(set(chain.from_iterable(df_klifs_uniprot_narm["pocket"].apply(lambda x: list(x)).tolist())))))

# # look at Pfam kinase domain annotations
# list_kd = ["Protein kinase domain", "Protein tyrosine and serine/threonine kinase"]
# # print(max(df_pfam.loc[df_pfam["name"].isin(list_kd), "uniprot"].value_counts().tolist())) # only 1 KD for those in list_kd

# df_pfam_kd_simple = df_pfam.loc[df_pfam["name"].isin(list_kd), ]

# df_multi = pd.DataFrame(df_pfam.loc[(~df_pfam["name"].isin(list_kd) & \
#                                      df_pfam["name"].apply(lambda x: "kinase" in x.lower())), \
#                         ["uniprot", "name", "type"]].groupby(["uniprot"]).agg(list))

# df_multi["count"] = df_multi["name"].apply(len)
# df_multi.sort_values(["count"], ascending=False, inplace=True)

# list_single_kd = df_multi.loc[df_multi["count"] == 1, "name"].tolist()
# list_multi_kd = df_multi.loc[df_multi["count"] > 1, "name"].tolist()

# # [idx for idx, i in enumerate(list_single_kd) if "kinase" not in i[0].lower()] # []

# df_temp = df_multi.loc[(df_multi["type"].apply(lambda x: "".join(x) != "family") \
#                         & df_multi["type"].apply(lambda x: "family" in x)), ].reset_index()

# for _, row in df_temp.iterrows():
#     print(row["uniprot"])
#     for i, j in zip(row["type"], row["name"]):
#         print(f"{i} : {j}")
#     print("")

# df_multi.loc[df_multi["type"].apply(lambda x: "".join(x) == "family"), ]

# list_multi_domain = (
#     df_multi.loc[df_multi["type"]
#         .apply(lambda x: "".join(x) == "domain"), "name"]
#         .tolist()
# )
