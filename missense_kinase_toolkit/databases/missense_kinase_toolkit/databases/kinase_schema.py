from enum import Enum, StrEnum
from pydantic import BaseModel, constr
import numpy as np
import pandas as pd


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


KinaseDomainName = StrEnum("KinaseDomainName", 
                           {"KD" + str(idx + 1): kd \
                            for idx, kd in enumerate(LIST_PFAM_KD)})


class KinHub(BaseModel):
    """Pydantic model for KinHub information."""

    kinase_name: str
    manning_name: list[str]
    xname: list[str]
    group: list[Group]
    family: list[Family]


class UniProt(BaseModel):
    """Pydantic model for UniProt information."""

    canonical_seq: constr(pattern=r"^[ACDEFGHIKLMNPQRSTVWXY]+$")


class KLIFS(BaseModel):
    """Pydantic model for KLIFS information."""
    gene_name: str
    name: str
    full_name: str
    group: Group
    family: Family
    iuphar: int
    kinase_id: int
    pocket_seq: constr(pattern=r"^[ACDEFGHIKLMNPQRSTVWY\-]+$") | None


class Pfam(BaseModel):
    """Pydantic model for Pfam information."""
    domain_name: KinaseDomainName
    start: int
    end: int
    protein_length: int
    pfam_accession: str
    in_alphafold: bool


class KinaseInfo(BaseModel):
    """Pydantic model for kinase information."""

    hgnc_name: str
    uniprot_id: constr(pattern=r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$")
    KinHub: KinHub
    UniProt: UniProt
    KLIFS: KLIFS | None
    Pfam: Pfam | None


class CollectionKinaseInfo(BaseModel):
    """Pydantic model for kinase information."""
    kinase_dict: dict[str, KinaseInfo]


def concatenate_source_dataframe(
    kinhub_df: pd.DataFrame,
    uniprot_df: pd.DataFrame,
    klifs_df: pd.DataFrame,
    pfam_df: pd.DataFrame,
    col_kinhub_merge: str | None = None,
    col_uniprot_merge: str | None = None,
    col_klifs_merge: str | None = None,
    col_pfam_merge: str | None = None,
    col_pfam_include: list[str] | None = None,
    list_domains_include: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate database dataframes on UniProt ID."""

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
            "in_alphafold"
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
        pfam_df_merge["name"].isin(LIST_PFAM_KD), 
        col_pfam_include
    ]

    # rename "name" column in Pfam so doesn't conflict with KLIFS name
    try:
        df_pfam_kd = df_pfam_kd.rename(columns={"name": "domain_name"})
    except KeyError:
        pass

    # concat dataframes
    df_merge = pd.concat(
        [
            kinhub_df_merge,
            uniprot_df_merge,
            klifs_df_merge,
            df_pfam_kd
        ],
        join="outer",
        axis=1
    ).reset_index()    

    return df_merge    


def is_not_valid_string(str_input: str) -> bool:
    if pd.isna(str_input) or str_input == "":
        return True
    else:
        return False


def convert_to_group(
    str_input: str, 
    bool_list: bool = True
) -> list[Group]:
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
    df: pd.DataFrame,
) -> dict[str, BaseModel]:
    """Create Pydantic models for kinases from dataframes."""

    # create KinHub model
    dict_kinase_models = {}

    for _, row in df.iterrows(): 
        # create KinHub model           
        kinhub = KinHub(
            kinase_name=row["Kinase Name"],
            manning_name=row["Manning Name"].split(", "),
            xname=row["xName"].split(", "),
            group=convert_to_group(row["Group"]),
            family=convert_to_family(row["Family"])
        )

        # create UniProt model
        uniprot = UniProt(
            canonical_seq=row["canonical_sequence"]
        )

        # create KLIFS model
        if is_not_valid_string(row["family"]):
            klifs = None
        else:
            if is_not_valid_string(row["pocket"]):
                pocket = None
            else:
                pocket = row["pocket"]
            klifs = KLIFS(
                gene_name=row["gene_name"],
                name=row["name"],
                full_name=row["full_name"],
                group=convert_to_group(row["group"], bool_list=False),
                family=convert_to_family(row["family"], bool_list=False),
                iuphar=row["iuphar"],
                kinase_id=row["kinase_ID"],
                pocket_seq=pocket
            )

        # create Pfam model
        if row["domain_name"] not in LIST_PFAM_KD:
            pfam = None
        else:
            pfam = Pfam(
                domain_name=row["domain_name"],
                start=row["start"],
                end=row["end"],
                protein_length=row["protein_length"],
                pfam_accession=row["pfam_accession"],
                in_alphafold=row["in_alphafold"]
            )

        # create KinaseInfo model
        kinase_info = KinaseInfo(
            hgnc_name=row["HGNC Name"],
            uniprot_id=row["index"],
            KinHub=kinhub,
            UniProt=uniprot,
            KLIFS=klifs,
            Pfam=pfam
        )

        dict_kinase_models[row["HGNC Name"]] = kinase_info

    return dict_kinase_models