import logging
import os
from itertools import chain
from typing import Any, Callable

import pandas as pd
from mkt.databases import hgnc, klifs, pfam, scrapers, uniprot
from mkt.databases.aligners import ClustalOmegaAligner
from mkt.databases.config import set_request_cache
from mkt.databases.io_utils import get_repo_root
from mkt.databases.kincore import align_kincore2uniprot, harmonize_kincore_fasta_cif
from mkt.databases.utils import rgetattr, rsetattr
from mkt.schema.constants import LIST_PFAM_KD
from mkt.schema.kinase_schema import (
    KLIFS,
    Family,
    Group,
    KinaseInfo,
    KinaseInfoKinaseDomain,
    KinaseInfoUniProt,
    KinCore,
    KinHub,
    Pfam,
    UniProt,
)
from pydantic import BaseModel, ValidationError, model_validator
from tqdm import tqdm
from typing_extensions import Self

logger = logging.getLogger(__name__)


class KinaseInfoUniProtGenerator(KinaseInfoUniProt):
    """Pydantic model for to generate KinaseInfoUniProt (kinase info at the level of the UniProt ID)."""

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
    def validate_phosphosites2canonicalseq(self) -> Self:
        """Validate phosphosites match canonical sequence."""
        from mkt.databases.colors import map_aa_to_single_letter_code
        from mkt.databases.utils import return_bool_at_index

        if self.uniprot.phospho_sites is not None:
            list_str_inter = [
                i.replace("Phospho", "").split("; ")[0]
                for i in self.uniprot.phospho_description
            ]
            list_str_out = [map_aa_to_single_letter_code(i) for i in list_str_inter]

            dict_phospho_map = dict(zip(self.uniprot.phospho_sites, list_str_out))

            list_bool = [
                self.uniprot.canonical_seq[k - 1] == v
                for k, v in dict_phospho_map.items()
            ]

            if not all(list_bool):

                LIST_ATTR = [
                    "uniprot.phospho_sites",
                    "uniprot.phospho_evidence",
                    "uniprot.phospho_description",
                ]

                dict_phosphosite = {
                    "replace": dict.fromkeys(LIST_ATTR),
                    "wrong": dict.fromkeys(LIST_ATTR),
                }

                for attr in LIST_ATTR:
                    # phospho_sites, evidence, and description lists that match cannonical sequence
                    dict_phosphosite["replace"][attr] = return_bool_at_index(
                        rgetattr(self, attr), list_bool, True
                    )
                    # phospho_sites, evidence, and description lists that do not match cannonical sequence
                    dict_phosphosite["wrong"][attr] = return_bool_at_index(
                        rgetattr(self, attr), list_bool, False
                    )

                # phospho_sites, evidence, and description lists that match cannonical sequence
                # list_site_replace = return_bool_at_index(self.uniprot.phospho_sites, list_bool)
                # list_evid_replace = return_bool_at_index(self.uniprot.phospho_evidence, list_bool)
                # list_desc_replace = return_bool_at_index(self.uniprot.phospho_description, list_bool)

                # phospho_sites, evidence, and description lists that do not match cannonical sequence
                # list_site_wrong = return_bool_at_index(self.uniprot.phospho_sites, list_bool, False)
                # list_evid_wrong = return_bool_at_index(self.uniprot.phospho_evidence, list_bool, False)
                # list_desc_wrong = return_bool_at_index(self.uniprot.phospho_description, list_bool, False)

                list_actual = [
                    self.uniprot.canonical_seq[k - 1] for k in dict_phospho_map.keys()
                ]
                list_actual_wrong = self.return_bool_at_index(
                    list_actual, list_bool, False
                )
                logger.warning(
                    f"{self.hgnc_name}/{self.uniprot_id} has canonical sequence to phosphosite mismatches:\n"
                    f"Actual:\n{list_actual_wrong}\n"
                    "From UniProt:\n"
                    f"{dict_phosphosite['wrong']['uniprot.phospho_description']}"
                    f"{dict_phosphosite['wrong']['uniprot.phospho_sites']}\n"
                    f"{dict_phosphosite['wrong']['uniprot.phospho_evidence']}\n"
                    f"Mismatched residues will be removed from phosphosite list.\n"
                )

                for attr in LIST_ATTR:
                    # replace phospho_sites, evidence, and description lists that do not match cannonical sequence
                    rsetattr(self, attr, dict_phosphosite["replace"][attr])

                # self.uniprot.phospho_sites = list_site_replace
                # self.uniprot.phospho_evidence = list_evid_replace
                # self.uniprot.phospho_description = list_desc_replace

        return self


class KinaseInfoKinaseDomainGenerator(KinaseInfoKinaseDomain):
    """Pydantic model for to generate KinaseInfoKinaseDomain (kinase info at the level of the kinase domain)."""

    # https://docs.pydantic.dev/latest/examples/custom_validators/#validating-nested-model-fields
    @model_validator(mode="after")
    def change_wrong_klifs_pocket_seq(self) -> Self:
        """KLIFS pocket has some errors compared to UniProt sequence - fix this via validation."""

        uniprot_id = self.uniprot_id.split("_")[0]

        # https://klifs.net/details.php?structure_id=9122 "NFM" > "HFM" but "H" present in canonical UniProt seq
        if uniprot_id == "Q8NI60":
            self.klifs.pocket_seq = "RPFAAASIGQVHLVAMKIQDYQREAACARKFRFYVPEIVDEVLTTELVSGFPLDQAEGLELFEFHFMQTDPNWSNFFYLLDFGAT"

        # https://klifs.net/details.php?structure_id=15054 shows "FLL" only change and region I flanks g.l
        if uniprot_id == "Q5S007":
            self.klifs.pocket_seq = "FLLGDGSFGSVYRVAVKIFLLRQELVVLCHLHPSLISLLAAMLVMELASKGSLDRLLQQYLHSAMIIYRDLKPHNVLLIADYGIA"

        # https://klifs.net/details.php?structure_id=9709 just a misalignment vs. UniProt[130:196] aligns matches structure seq
        if uniprot_id == "Q8N5S9":
            self.klifs.pocket_seq = "QSEIGKGAYGVVRHYAMKVERVYQEIAILKKLHVNVVKLIENLYLVFDLRKGPVMEVPCEYLHCQKIVHRDIKPSNLLKIADFGV"

        return self

    @model_validator(mode="after")
    def generate_kincore_fasta2cif_alignment(self) -> Self:
        """Generate dictionary mapping KinCore FASTA to CIF indices."""
        if self.kincore is not None:

            # all non-None entries will have fastas
            fasta = self.kincore.fasta.seq

            if self.kincore.cif is not None:

                key_seq = "_entity_poly.pdbx_seq_one_letter_code"
                cif = self.kincore.cif.cif[key_seq][0].replace("\n", "")

                # KinCoreFASTA2CIF
                dict_temp = align_kincore2uniprot(fasta, cif)
                self.kincore.start = dict_temp["start"]
                self.kincore.end = dict_temp["end"]
                self.kincore.mismatch = dict_temp["mismatch"]

        return self


class KinaseInfoGenerator(KinaseInfo):
    """Pydantic model for kinase information."""

    bool_offset: bool = True

    @classmethod
    def standardize_offset(cls, idx_in: int) -> int:
        """Standardize offset where necessary.

        Parameters
        ----------
        idx_in : int
            Index to standardize.

        Returns
        -------
        int
            Standardized index.
        """
        if not cls.bool_offset:
            return idx_in - 1
        else:
            return idx_in

    @model_validator(mode="after")
    def generate_kincore2uniprot_alignment(self) -> Self:
        """Generate dictionary mapping KinCore to UniProt indices."""
        if self.kincore is not None:

            # all non-None entries will have fastas
            fasta = self.fasta.seq

            # KinCoreFASTA2UniProt
            dict_temp = align_kincore2uniprot(fasta, self.uniprot.canonical_seq)
            self.kincore.fasta.start = self.standardize_offset(dict_temp["start"])
            self.kincore.fasta.end = self.standardize_offset(dict_temp["end"])
            self.kincore.fasta.mismatch = dict_temp["mismatch"]

            if self.kincore.cif is not None:

                key_seq = "_entity_poly.pdbx_seq_one_letter_code"
                cif = self.kincore.cif.cif[key_seq][0].replace("\n", "")

                # KinCoreCIF2UniProt
                dict_temp = align_kincore2uniprot(cif, self.uniprot.canonical_seq)
                self.kincore.cif.start = self.standardize_offset(dict_temp["start"])
                self.kincore.cif.end = self.standardize_offset(dict_temp["end"])
                self.kincore.cif.mismatch = dict_temp["mismatch"]

        return self

    @model_validator(mode="after")
    def generate_klifs2uniprot_dict(self) -> Self:
        """Generate dictionary mapping KLIFS to UniProt indices."""

        if self.kincore is not None:
            kd_idx = (
                min(
                    [
                        i
                        for i in [self.kincore.fasta.start, self.kincore.cif.start]
                        if i is not None
                    ]
                )
                - 1,
                max(
                    [
                        i
                        for i in [self.kincore.fasta.end, self.kincore.cif.end]
                        if i is not None
                    ]
                )
                - 1,
            )
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


def is_not_valid_string(str_input: str) -> bool:
    if pd.isna(str_input) or str_input == "" or isinstance(str_input, float | int):
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


def return_none_if_not_valid_string(str_input: str) -> str | None:
    """Return None if string is not valid.

    Parameters
    ----------
    str_input : str
        String to check.

    Returns
    -------
    str | None
        String if valid, otherwise None.
    """
    if is_not_valid_string(str_input):
        return None
    else:
        return str_input


DICT_COL2OBJ_ORIG = {
    "kinhub": {
        "object": KinHub,
        "uniprot_id": "UniprotID",
        "keys": {
            "list_obj": [
                "hgnc_name",
                "kinase_name",
                "manning_name",
                "xname",
                "group",
                "family",
            ],
            "list_col": [
                "HGNC Name",
                "Kinase Name",
                "Manning Name",
                "xName",
                "Group",
                "Family",
            ],
            "list_fnc": [
                return_none_if_not_valid_string,
                return_none_if_not_valid_string,
                lambda x: x,
                lambda x: x,
                lambda x: convert_to_group(x, bool_list=False),
                lambda x: convert_to_family(x, bool_list=False),
            ],
        },
    },
    "klifs": {
        "object": KLIFS,
        "uniprot_id": "uniprot",
        "keys": {
            "list_obj": [
                "gene_name",
                "name",
                "full_name",
                "group",
                "family",
                "iuphar",
                "kinase_id",
                "pocket_seq",
            ],
            "list_col": [
                "gene_name",
                "name",
                "full_name",
                "group",
                "family",
                "iuphar",
                "kinase_ID",
                "pocket",
            ],
            "list_fnc": [
                lambda x: x,
                lambda x: x,
                lambda x: x,
                lambda x: convert_to_group(x, bool_list=False),
                lambda x: convert_to_family(x, bool_list=False),
                lambda x: x,
                lambda x: x,
                return_none_if_not_valid_string,
            ],
        },
    },
    "pfam": {
        "object": Pfam,
        "uniprot_id": "uniprot",
        "keys": {
            "list_obj": [
                "domain_name",
                "start",
                "end",
                "protein_length",
                "pfam_accession",
                "in_alphafold",
            ],
            "list_col": [
                "name",
                "start",
                "end",
                "protein_length",
                "pfam_accession",
                "in_alphafold",
            ],
            "list_fnc": [
                lambda x: x,
                lambda x: x,
                lambda x: x,
                lambda x: x,
                lambda x: x,
                lambda x: x,
            ],
        },
    },
}
"""dict[str, dict[str, Callable | str | dict[str, list[str, Callable]]]]: Dictionary of columns to object mapping."""


def process_keys_dict(
    dict_in: dict[str, list[str, Callable]]
) -> dict[str, dict[str, str | Callable]]:
    """Process keys dictionary to convert to list of functions.

    Parameters
    ----------
    dict_in : dict[str, list[str, function]]
        Dictionary of keys to process.

    Returns
    -------
    dict[str, dict[str, str | function]]
        Dictionary of keys with list of functions.
    """
    list_val = [
        {"column": col, "function": fnc}
        for col, fnc in zip(dict_in["list_col"], dict_in["list_fnc"])
    ]

    dict_out = dict(zip(dict_in["list_obj"], list_val))

    return dict_out


DICT_COL2OBJ_REV = {
    k1: {k2: (process_keys_dict(v2) if k2 == "keys" else v2) for k2, v2 in v1.items()}
    for k1, v1 in DICT_COL2OBJ_ORIG.items()
}
"""dict[str, dict[str, Callable | str | dict[str, dict[str, str | Callable]]]: Dictionary of columns to object mapping."""


def convert_df2dictobj(
    df: pd.DataFrame,
    str_obj: str,
) -> dict[str, Any] | None:
    """Convert dataframe to dictionary of objects.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to convert.
    str_obj : str
        Object to convert to - needs to match key in DICT_COL2OBJ_REV.

    Returns
    -------
    dict[str, Any]
        Dictionary of objects where key is UniProt ID and value is object.

    """
    dict_obj = DICT_COL2OBJ_REV[str_obj]

    obj = dict_obj["object"]
    dict_key = dict_obj["keys"]
    try:
        list_key = df[dict_obj["uniprot_id"]].to_list()
    except Exception as e:
        logger.error(f"Key error {e} for {str_obj}")
        return None

    # keep only kinase domains from Pfam
    if str_obj == "pfam":
        df = df.loc[df["name"].isin(LIST_PFAM_KD), :].reset_index(drop=True)

    list_cols = [
        df[v["column"]].apply(v["function"]).tolist() for v in dict_key.values()
    ]
    set_count = {len(i) for i in list_cols}

    if len(set_count) != 1:
        logger.error("One or more columns have no values.")
        return None

    dict_out = {i.upper(): [] for i in list_key}  # Pfam not capitalized
    for key, entry in zip(list_key, zip(*list_cols)):
        obj_temp = obj.model_validate(dict(zip(dict_key.keys(), entry)))
        dict_out[key.upper()].append(obj_temp)

    return dict_out


DICT_MERGE_MULTIMAP = {
    "general": {
        "kinhub": "xname",
        "klifs": "gene_name",
        "kincore": "fasta.hgnc",
    },
    "manual": {
        "P23458": [
            ["JAK1", "JAK1", "JAK1"],
            ["JAK1_b", "JAK1-b", None],
        ],
        "Q15772": [
            ["SPEG", "SPEG", "SPEG1"],
            ["SPEG_b", "SPEG-b", "SPEG2"],
        ],
        "Q9UK32": [
            ["RSK4", "RPS6KA6", "RPS6KA61"],
            ["RSK4_b", "RPS6KA6-b", "RPS6KA62"],
        ],
        "P29597": [
            ["TYK2", "TYK2", "TYK2"],
            ["TYK2_b", "TYK2-b", None],
        ],
        "Q15349": [
            ["RSK3", "RPS6KA2", "RPS6KA21"],
            ["RSK3_b", "RPS6KA2-b", "RPS6KA22"],
        ],
        "Q15418": [
            ["RSK1", "RPS6KA1", "RPS6KA11"],
            ["RSK1_b", "RPS6KA1-b", "RPS6KA12"],
        ],
        "Q5VST9": [
            ["Obscn", "OBSCN", "OBSCN1"],
            ["Obscn_b", "OBSCN-b", "OBSCN2"],
        ],
        "O60674": [
            ["JAK2", "JAK2", "JAK2"],
            ["JAK2_b", "JAK2-b", None],
        ],
        "Q9P2K8": [
            ["GCN2", "EIF2AK4", "EIF2AK4"],
            ["GCN2_b", "EIF2AK4-b", None],
        ],
        "P51812": [
            ["RSK2", "RPS6KA3", "RPS6KA31"],
            ["RSK2_b", "RPS6KA3-b", "RPS6KA32"],
        ],
        "O75676": [
            ["MSK2", "RPS6KA4", "RPS6KA41"],
            ["MSK2_b", "RPS6KA4-b", "RPS6KA42"],
        ],
        "O75582": [
            ["MSK1", "RPS6KA5", "RPS6KA51"],
            ["MSK1_b", "RPS6KA5-b", "RPS6KA52"],
        ],
        "Q8IWB6": [
            ["SgK307", "TEX14", "TEX14"],
            ["SgK424", None, None],
        ],
        "P52333": [
            ["JAK3", "JAK3", "JAK3"],
            ["JAK3_b", "JAK3-b", None],
        ],
    },
}
"""dict[str, dict[str, str] | list[list[str]]]: Dictionary where keys are UniProt IDs with multi-mapping.
    For the "general" sub-dictionary each key is the key of the dict_obj and the value is the attr on which to collapse.
    For the "manual" sub-dict values are lists where entries are for KinHub, KLIFS, and KinCore, respectively.
"""


DICT_MERGE_MULTIMAP_REV = {
    k1: (
        {
            k2: [
                dict(zip(DICT_MERGE_MULTIMAP["general"].keys(), entry)) for entry in v2
            ]
            for k2, v2 in v1.items()
        }
        if k1 == "manual"
        else v1
    )
    for k1, v1 in DICT_MERGE_MULTIMAP.items()
}
"""dict[str, dict[str, str] | list[list[str]]]: DICT_MERGE_MULTIMAP in more accessible format."""


def find_alternative_hgnc(
    id_uniprot: str,
    kinhub_dict: dict[str, Any],
    klifs_dict: dict[str, Any],
    kincore_dict: dict[str, Any],
    kinhub_attr: str = ["hgnc_name", "xname"],
    klifs_attr: str = ["gene_name"],
    kincore_attr: str = ["fasta.hgnc"],
) -> str | list[str] | None:
    """Find alternative HGNC names for a given UniProt ID.

    Parameters
    ----------
    id_uniprot : str
        UniProt ID to search for.
    kinhub_dict : dict[str, Any]
        KinHub dictionary.
    klifs_dict : dict[str, Any]
        KLIFS dictionary.
    kincore_dict : dict[str, Any]
        KinCore dictionary.
    kinhub_attr : list[str], optional
        List of attributes to access in KinHub dictionary.
    klifs_attr : list[str], optional
        List of attributes to access in KLIFS dictionary.
    kincore_attr : list[str], optional
        AttribuList of attributeste to access in KinCore dictionary.

    Returns
    -------
    str | list[str] | None
        String, list of strings of alternative HGNC names if found, else None.
    """
    list_dict = [kinhub_dict, klifs_dict, kincore_dict]
    list_attr = [kinhub_attr, klifs_attr, kincore_attr]

    list_out = []
    for dict_in, attr_list in zip(list_dict, list_attr):
        for attr in attr_list:
            try:
                entry = dict_in[id_uniprot]
                if isinstance(entry, list):
                    temp = [rgetattr(i, attr) for i in entry]
                    if all(val is None for val in temp):
                        continue
                    if isinstance(temp[0], set):
                        temp = list(chain(*temp))
                    list_out.extend(temp)
                    break  # Stop trying other attributes if a valid value is found
                else:
                    temp = rgetattr(entry, attr)
                    if temp is None:
                        continue
                    list_out.append(temp)
                    break  # Stop trying other attributes if a valid value is found
            except Exception:
                logger.warning(
                    f"Attribute {attr} not found in dictionary for UniProt ID {id_uniprot}..."
                )
                pass

    if len(list_out) == 0:
        return None
    elif len(list_out) == 1:
        return list_out[0]
    else:
        return list_out


def generate_dict_obj_from_api_or_scraper() -> dict[str, pd.DataFrame]:
    """Generate dataframes for KinHub, KLIFS, and Pfam databases.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing processed dataframes.
    """
    # set up request cache
    set_request_cache(os.path.join(get_repo_root(), "requests_cache.sqlite"))

    # kinhub
    df_kinhub = scrapers.kinhub()
    dict_kinhub = convert_df2dictobj(df_kinhub, "kinhub")

    # klifs
    klifs_kinase_info = klifs.KinaseInfo()
    df_klifs = pd.DataFrame(klifs_kinase_info.get_kinase_info())
    dict_klifs = convert_df2dictobj(df_klifs, "klifs")

    # kincore
    dict_kincore = harmonize_kincore_fasta_cif()

    set_uniprot = set(
        list(dict_kinhub.keys()) + list(dict_klifs.keys()) + list(dict_kincore.keys())
    )

    # collect HGNC, UniProt, and Pfam data from API
    dict_hgnc, dict_uniprot, dict_pfam = {}, {}, {}
    for uniprot_id in tqdm(set_uniprot, desc="Querying UniProt, HGNC, and Pfam..."):
        # HGNC
        obj_temp = hgnc.HGNC(uniprot_id)
        obj_temp.maybe_get_symbol_from_hgnc_search(
            custom_field="uniprot_ids", custom_term=uniprot_id
        )
        hgnc_name = obj_temp.hgnc
        dict_hgnc[uniprot_id] = hgnc_name

        # UniProt
        fasta = uniprot.UniProtFASTA(uniprot_id)
        json = uniprot.UniProtJSON(uniprot_id)
        dict_temp = {
            "header": fasta._header,
            "canonical_seq": fasta._sequence,
        } | json.dict_mod_res
        obj_temp = UniProt.model_validate(dict_temp)
        dict_uniprot[uniprot_id] = obj_temp

        # Pfam
        df_pfam = pfam.Pfam(uniprot_id)._pfam
        if df_pfam is not None:
            dict_temp = convert_df2dictobj(df_pfam, "pfam")
            if dict_temp is None:
                logger.warning(f"{uniprot_id} has no Pfam entry...")
            elif len(dict_temp[uniprot_id]) == 0:
                logger.warning(f"{uniprot_id} has no Pfam annotated KD...")
            else:
                try:
                    assert len(dict_temp[uniprot_id]) == 1
                except AssertionError:
                    logger.warning(
                        f"{uniprot_id} has multiple Pfam KD entries. Defaulting to first..."
                    )
                dict_pfam[uniprot_id] = dict_temp[uniprot_id][0]

    # B5MCJ9, A0A0B4J2F2, and Q6IBK5 are missing HGNC names
    # alternative option (not for A0A0B4J2F2 which is no longer in SwissProt):
    # uniprot.UniProtJSON(uniprot_id)._json["genes"][0]["geneName"]["value"]
    list_hgnc_missing = [k for k, v in dict_hgnc.items() if k == v]
    list_hgnc_new = [
        find_alternative_hgnc(i, dict_kinhub, dict_klifs, dict_kincore)
        for i in list_hgnc_missing
    ]
    dict_replace = dict(zip(list_hgnc_missing, list_hgnc_new))
    dict_hgnc.update(dict_replace)

    try:
        assert len(dict_uniprot) == len(set_uniprot)
    except AssertionError:
        logger.warning(
            f"UniProt dictionary has {len(dict_uniprot)} entries but {len(set_uniprot)} unique UniProt IDs."
        )

    dict_out = {
        "kinhub": dict_kinhub,
        "klifs": dict_klifs,
        "kincore": dict_kincore,
        "hgnc": dict_hgnc,
        "uniprot": dict_uniprot,
        "pfam": dict_pfam,
    }

    logger.info("Retrieved the following...")
    for idx, (k, v) in enumerate(dict_out.items()):
        if idx != len(dict_out) - 1:
            logger.info(f"\t{k}: {len(v)} entries")
        else:
            logger.info(f"\t{k}: {len(v)} entries\n")

    return dict_out


def combine_kinaseinfo_uniprot(
    dict_in: dict[str, dict[str, Any]],
) -> dict[str, KinaseInfoUniProtGenerator]:
    """Generate KinaseInfoUniProtGenerator from dictionary generated by generate_dict_obj_from_api_or_scraper.

    Parameters
    ----------
    dict_in : dict[str, str]
        Dictionary of dictionary of mkt.kinase_schema objects.
        Keys needed here are "hgnc", "uniprot", and "pfam".

    Returns
    -------
    dict[str, KinaseInfoUniProtGenerator]
        Dictionary of KinaseInfoUniProtGenerator objects.

    """
    try:
        dict_hgnc = dict_in["hgnc"]
        dict_uniprot = dict_in["uniprot"]
        dict_pfam = dict_in["pfam"]
    except KeyError as e:
        logger.error(f"Key error {e} in combine_kinaseinfo_uniprot.")

    set_uniprot = set(
        list(dict_hgnc.keys()) + list(dict_uniprot.keys()) + list(dict_pfam.keys())
    )

    dict_kinaseinfo_uniprot = {}
    for uniprot_id in set_uniprot:
        str_hgnc = dict_hgnc[uniprot_id]
        obj_uniprot = dict_uniprot[uniprot_id]

        try:
            obj_pfam = dict_pfam[uniprot_id]
        except KeyError:
            obj_pfam = None

        obj_temp = KinaseInfoUniProtGenerator(
            hgnc_name=str_hgnc,
            uniprot_id=uniprot_id,
            uniprot=obj_uniprot,
            pfam=obj_pfam,
        )

        dict_kinaseinfo_uniprot[uniprot_id] = obj_temp

    return dict_kinaseinfo_uniprot


def combine_kinaseinfo_kd(
    dict_in: dict[str, dict[str, Any]],
) -> dict[str, KinaseInfoKinaseDomainGenerator]:
    """Generate KinaseInfoKinaseDomainGenerator from dictionary generated by generate_dict_obj_from_api_or_scraper.

    Parameters
    ----------
    dict_in : dict[str, str]
        Dictionary of dictionary of mkt.kinase_schema objects.
        Keys needed here are "hgnc", "uniprot", and "pfam".

    Returns
    -------
    dict[str, KinaseInfoKinaseDomainGenerator]
        Dictionary of KinaseInfoKinaseDomainGenerator objects.

    """
    dict_general = DICT_MERGE_MULTIMAP_REV["general"]
    dict_manual = DICT_MERGE_MULTIMAP_REV["manual"]

    try:
        set_uniprot = set(chain(*[dict_in[k].keys() for k in dict_general.keys()]))
    except KeyError as e:
        logger.error(f"Key error {e} in combine_kinaseinfo_kd.")

    dict_kinaseinfo_kd = {}
    for uniprot_id in set_uniprot:

        dict_temp = {i: None for i in ["uniprot_id"] + list(dict_general.keys())}

        # if mult-kinase domain entry
        if uniprot_id in dict_manual:

            dict_man_uniprot = dict_manual[uniprot_id]

            for i, l in enumerate(dict_man_uniprot):

                uniprot_id_temp = uniprot_id + "_" + str(i + 1)
                dict_temp["uniprot_id"] = uniprot_id_temp

                for k, v in l.items():
                    if v is None:
                        dict_temp[k] = None
                    else:
                        # kincore fasta.hgnc is a set
                        if k == "kincore":
                            dict_temp[k] = [
                                i
                                for i in dict_in[k][uniprot_id]
                                if v in rgetattr(i, dict_general[k])
                            ][0]
                        # kinhub xname and klifs gene_name are strings
                        else:
                            dict_temp[k] = [
                                i
                                for i in dict_in[k][uniprot_id]
                                if rgetattr(i, dict_general[k]) == v
                            ][0]

                obj_temp = KinaseInfoKinaseDomainGenerator.model_validate(dict_temp)
                dict_kinaseinfo_kd[obj_temp.uniprot_id] = obj_temp
                dict_temp = {
                    i: None for i in ["uniprot_id"] + list(dict_general.keys())
                }

        # if single kinase domain entry
        else:
            dict_temp["uniprot_id"] = uniprot_id
            for k in dict_general.keys():
                try:
                    dict_temp[k] = dict_in[k][uniprot_id][0]
                except KeyError or IndexError:
                    logger.warning(f"Key error: {uniprot_id} missing {k} entry.")
                    dict_temp[k] = None

            obj_temp = KinaseInfoKinaseDomainGenerator.model_validate(dict_temp)
            dict_kinaseinfo_kd[obj_temp.uniprot_id] = obj_temp

    return dict_kinaseinfo_kd


PATH_DATA_DIR = os.path.join(get_repo_root(), "data")


DICT_INPUT_FILES = {
    "kinhub": {
        "filename": os.path.join(PATH_DATA_DIR, "kinhub.csv"),
        "col_merge": "UniprotID",
        "df": None,
    },
    "uniprot": {
        "filename": os.path.join(PATH_DATA_DIR, "kinhub_uniprot.csv"),
        "col_merge": "uniprot_id",
        "df": None,
    },
    "klifs": {
        "filename": os.path.join(PATH_DATA_DIR, "kinhub_klifs.csv"),
        "col_merge": "uniprot",
        "df": None,
    },
    "pfam": {
        "filename": os.path.join(PATH_DATA_DIR, "kinhub_pfam.csv"),
        "col_merge": "uniprot",
        "df": None,
    },
    "kincore": {
        "filename": os.path.join(PATH_DATA_DIR, "kinhub_kincore.csv"),
        "col_merge": "uniprot",
        "df": None,
    },
}


LIST_COL_PFAM_INCLUDE = [
    "name",
    "start",
    "end",
    "protein_length",
    "pfam_accession",
    "in_alphafold",
]


def concatenate_source_dataframe(
    kinhub_df: pd.DataFrame | None = None,
    uniprot_df: pd.DataFrame | None = None,
    klifs_df: pd.DataFrame | None = None,
    pfam_df: pd.DataFrame | None = None,
    kincore_df: pd.DataFrame | None = None,
    col_kinhub_merge: str | None = None,
    col_uniprot_merge: str | None = None,
    col_klifs_merge: str | None = None,
    col_pfam_merge: str | None = None,
    col_pfam_include: list[str] | None = None,
    list_domains_include: list[str] | None = None,
    col_kincore_merge: str | None = None,
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
    kincore_df : pd.DataFrame | None, optional
        KinCore dataframe, by default None and will be loaded from "data" dir.
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
    col_kincore_merge : str | None, optional
        Column to merge KinCore dataframe, by default None.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe.
    """
    from copy import deepcopy

    DICT_INPUT_FILES_REV = deepcopy(DICT_INPUT_FILES)

    # order needs to match DICT_INPUT_FILES_REV keys
    list_df = [
        kinhub_df,
        uniprot_df,
        klifs_df,
        pfam_df,
        kincore_df,
    ]

    # Pfam columns to include in the final dataframe
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

    # load dataframes if not provided from "data" sub-directory
    for key, df in zip(DICT_INPUT_FILES_REV.keys(), list_df):
        if df is None:
            df_temp = check_if_file_exists_then_load_dataframe(
                DICT_INPUT_FILES_REV[key]["filename"]
            )
        else:
            df_temp = df.copy()
        if key == "pfam":
            # filter Pfam dataframe for KD domains and columns to include
            df_temp = df_temp.loc[
                df_temp["name"].isin(list_domains_include),
                col_pfam_include,
            ].reset_index(drop=True)
            # rename "name" column in Pfam so doesn't conflict with KLIFS name
            try:
                df_temp.rename(columns={"name": "domain_name"}, inplace=True)
            except KeyError:
                pass
        DICT_INPUT_FILES_REV[key]["df"] = df_temp
    list_df = [i["df"] for i in DICT_INPUT_FILES_REV.values()]

    if any([True if i is None else False for i in list_df]):
        list_df_shape = [i.shape if i is not None else None for i in list_df]
        logger.error(f"One or more dataframes are None\n{list_df_shape}")
        return None

    # order needs to match DICT_INPUT_FILES_REV keys
    list_col = [
        col_kinhub_merge,
        col_uniprot_merge,
        col_klifs_merge,
        col_pfam_merge,
        col_kincore_merge,
    ]
    for key, col in zip(DICT_INPUT_FILES_REV.keys(), list_col):
        if col is not None:
            DICT_INPUT_FILES_REV[key]["col_merge"] = col
    list_col = [i["col_merge"] for i in DICT_INPUT_FILES_REV.values()]

    # set indices to merge columns
    for df, col in zip(list_df, list_col):
        df.set_index(col, inplace=True)

    # concat dataframes
    df_merge = pd.concat(list_df, join="outer", axis=1).reset_index()

    return df_merge


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
    DICT_KINCORE = extract_pk_fasta_info_as_dict()  # noqa F821

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
        for hgnc_name, seq in dict_in[region].items():
            if seq is None:
                dict_in[region][hgnc_name] = "-" * length

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
        region: {hgnc_name: None for hgnc_name in dict_in.keys()}
        for region in list_inter_intra
    }

    for region in list_inter_intra:
        list_hgnc, list_seq = [], []
        for hgnc_name, kinase_info in dict_in.items():
            try:
                seq = kinase_info.KLIFS2UniProtSeq[region]
            except TypeError:
                seq = None
            if seq is not None:
                list_hgnc.append(hgnc_name)
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
