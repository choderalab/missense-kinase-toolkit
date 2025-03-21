import logging
import os
from itertools import chain

import pandas as pd
from mkt.databases import klifs
from mkt.databases.aligners import ClustalOmegaAligner
from mkt.databases.io_utils import get_repo_root
from mkt.databases.kincore import align_kincore2uniprot, extract_pk_fasta_info_as_dict
from mkt.schema.constants import LIST_PFAM_KD
from mkt.schema.kinase_schema import (
    KLIFS,
    Family,
    Group,
    KinaseInfo,
    KinCore,
    KinHub,
    Pfam,
    UniProt,
)
from pydantic import BaseModel, ValidationError, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)


class KinaseInfoGenerator(KinaseInfo):
    """Pydantic model for kinase information."""

    bool_offset: bool = True

    # https://docs.pydantic.dev/latest/examples/custom_validators/#validating-nested-model-fields
    @model_validator(mode="after")
    def change_wrong_klifs_pocket_seq(self) -> Self:
        """KLIFS pocket has some errors compared to UniProt sequence - fix this via validation."""
        # https://klifs.net/details.php?structure_id=9122 "NFM" > "HFM" but "H" present in canonical UniProt seq
        if self.hgnc_name == "ADCK3":
            self.klifs.pocket_seq = "RPFAAASIGQVHLVAMKIQDYQREAACARKFRFYVPEIVDEVLTTELVSGFPLDQAEGLELFEFHFMQTDPNWSNFFYLLDFGAT"
        # https://klifs.net/details.php?structure_id=15054 shows "FLL" only change and region I flanks g.l
        if self.hgnc_name == "LRRK2":
            self.klifs.pocket_seq = "FLLGDGSFGSVYRVAVKIFLLRQELVVLCHLHPSLISLLAAMLVMELASKGSLDRLLQQYLHSAMIIYRDLKPHNVLLIADYGIA"
        # https://klifs.net/details.php?structure_id=9709 just a misalignment vs. UniProt[130:196] aligns matches structure seq
        if self.hgnc_name == "CAMKK1":
            self.klifs.pocket_seq = "QSEIGKGAYGVVRHYAMKVERVYQEIAILKKLHVNVVKLIENLYLVFDLRKGPVMEVPCEYLHCQKIVHRDIKPSNLLKIADFGV"
        return self

    # https://stackoverflow.com/questions/68082983/validating-a-nested-model-in-pydantic
    # skip if other validation errors occur in nested models first
    @model_validator(mode="after")
    def validate_uniprot_length(self) -> Self:
        """Validate canonical UniProt sequence length matches Pfam length if Pfam not None."""
        if self.pfam is not None:
            if len(self.uniprot.canonical_seq) != self.pfam.protein_length:
                raise ValidationError(
                    "UniProt sequence length does not match Pfam protein length."
                )
        return self

    @model_validator(mode="after")
    def generate_klifs2uniprot_dict(self) -> Self:
        """Generate dictionary mapping KLIFS to UniProt indices."""

        if self.kincore is not None:
            kd_idx = (self.kincore.start - 1, self.kincore.end - 1)
        else:
            kd_idx = (None, None)

        if self.klifs is not None and self.klifs.pocket_seq is not None:
            temp_obj = klifs.KLIFSPocket(
                uniprotSeq=self.uniprot.canonical_seq,
                klifsSeq=self.klifs.pocket_seq,
                idx_kd=kd_idx,
                offset_bool=self.bool_offset,
            )

            if temp_obj.list_align is not None:
                self.KLIFS2UniProtIdx = temp_obj.KLIFS2UniProtIdx
                self.KLIFS2UniProtSeq = temp_obj.KLIFS2UniProtSeq

        return self


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

    # concatenate_source_dataframe could return None
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

        # TODO: include all KLIFS entries rather than just those in KinHub
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
        kinase_info = KinaseInfoGenerator(
            hgnc_name=name_hgnc,
            uniprot_id=id_uniprot,
            kinhub=kinhub_model,
            uniprot=uniprot_model,
            klifs=klifs_model,
            pfam=pfam_model,
            kincore=kincore_model,
        )

        dict_kinase_models[name_hgnc] = kinase_info

    # TODO: For entries in DICT_KINCORE that are not in df, add to dict_kinase_models

    return dict_kinase_models


def get_sequence_max_with_exception(list_in: list[int | None]) -> int:
    """Get maximum sequence length from dictionary of dictionaries.

    Parameters
    ----------
    dict_in : dict[str, dict[str, str | None]]
        Dictionary of dictionaries.

    Returns
    -------
    int
        Maximum sequence length.
    """
    try:
        return max(list_in)
    except ValueError:
        return 0


def replace_none_with_max_len(dict_in):
    dict_max_len = {
        key1: get_sequence_max_with_exception(
            [len(val2) for val2 in val1.values() if val2 is not None]
        )
        for key1, val1 in dict_in.items()
    }

    for region, length in dict_max_len.items():
        for hgnc, seq in dict_in[region].items():
            if seq is None:
                dict_in[region][hgnc] = "-" * length

    return dict_in


def align_inter_intra_region(
    dict_in: dict[str, KinaseInfo],
) -> dict[str, dict[str, str]]:
    """Align inter and intra region sequences.

    Parameters
    ----------
    dict_in : dict[str, KinaseInfo]
        Dictionary of kinase information models

    Returns
    -------
    dict[str, dict[str, str]]
        Dictionary of aligned inter and intra region
    """

    list_inter_intra = klifs.LIST_INTER_REGIONS + klifs.LIST_INTRA_REGIONS

    dict_align = {
        region: {hgnc: None for hgnc in dict_in.keys()} for region in list_inter_intra
    }

    for region in list_inter_intra:
        list_hgnc, list_seq = [], []
        for hgnc, kinase_info in dict_in.items():
            try:
                seq = kinase_info.KLIFS2UniProtSeq[region]
            except TypeError:
                seq = None
            if seq is not None:
                list_hgnc.append(hgnc)
                list_seq.append(seq)
        if len(list_seq) > 2:
            aligner_temp = ClustalOmegaAligner(list_seq)
            dict_align[region].update(
                dict(zip(list_hgnc, aligner_temp.list_alignments))
            )
        else:
            # hinge:linker - {'ATR': 'N', 'CAMKK1': 'L'}
            # αE:VI - {'MKNK1': 'DKVSLCHLGWSAMAPSGLTAAPTSLGSSDPPTSASQVAGTT'}
            dict_align[region].update(dict(zip(list_hgnc, list_seq)))

    replace_none_with_max_len(dict_align)

    return dict_align


def reverse_order_dict_of_dict(
    dict_in: dict[str, dict[str, str | int | None]],
) -> dict[str, dict[str, str | int | None]]:
    """Reverse order of dictionary of dictionaries.

    Parameters
    ----------
    dict_in : dict[str, dict[str, str | int | None]]
        Dictionary of dictionaries

    Returns
    -------
    dict_out : dict[str, dict[str, str | int | None]]
        Dictionary of dictionaries with reversed order

    """
    dict_out = {
        key1: {key2: dict_in[key2][key1] for key2 in dict_in.keys()}
        for key1 in set(chain(*[list(j.keys()) for j in dict_in.values()]))
    }
    return dict_out
