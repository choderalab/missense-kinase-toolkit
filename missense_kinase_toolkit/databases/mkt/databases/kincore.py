"""Parsing and harmonization of KinCoRe FASTA and CIF structure files, aligned to UniProt.

Reads KinCoRe FASTA and CIF files, extracts kinase-domain metadata, aligns KinCoRe
sequences to UniProt, and harmonizes the FASTA- and CIF-derived records.
"""

import io
import logging
import os
import re
import tarfile
import zipfile
from collections import Counter
from itertools import chain

from Bio import SeqIO

# from biotite.structure.io.pdbx import CIFFile
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from mkt.databases.aligners import Kincore2UniProtAligner
from mkt.databases.io_utils import DataSource
from mkt.databases.utils import (
    flatten_iterables_in_iterable,
    split_on_first_only,
    try_except_split_concat_str,
)
from mkt.schema.io_utils import get_repo_root, untar_files_in_memory
from mkt.schema.kinase_schema import (
    KinCoRe,
    KinCoReCIF,
    KinCoReFASTA,
    KinCoReSeqSource,
    KinCoReStructureSource,
)
from mkt.schema.utils import TQDM_BAR_FORMAT
from tqdm import tqdm

logger = logging.getLogger(__name__)


PATH_DATA = os.path.join(get_repo_root(), "data")
PATH_ORIG_CIF = os.path.join(
    PATH_DATA, "Kincore_AlphaFold2_ActiveHumanCatalyticKinases_v2.tar.gz"
)
PATH_ALIGN_CIF = os.path.join(
    PATH_DATA, "Kincore_AlphaFold2_ActiveHumanCatalyticKinases_v2_aligned.tar.gz"
)

KINCORE_CIF_URL = (
    "https://dunbrack.fccc.edu/kincore/static/downloads/af2activemodels/"
    "AF2_Active_Models_v2.zip"
)
"""str: Dunbrack KinCoRe AF2 active-model v2 archive (one active-state CIF per kinase domain)."""
PATH_CIF_ZIP = os.path.join(PATH_DATA, "AF2_Active_Models_v2.zip")
"""str: Local (gitignored) cache path for the downloaded v2 CIF archive."""

KINCORE_FASTA_URL = (
    "https://dunbrack.fccc.edu/kincore/static/downloads/af2activemodels/"
    "kinasedomainfasta.tar.gz"
)
"""str: Dunbrack KinCoRe kinase-domain FASTA archive matching the v2 active models."""
PATH_FASTA_TAR = os.path.join(PATH_DATA, "kinasedomainfasta.tar.gz")
"""str: Local (gitignored) cache path for the downloaded kinase-domain FASTA archive."""
PATH_FASTA_COMBINED = os.path.join(PATH_DATA, "kinasedomainfasta_combined.fasta")
"""str: Local (gitignored) concatenation of the per-kinase kinase-domain FASTA files."""


# --- KinCoRe source provenance ---
# citations for the Dunbrack KinCoRe resources (sequence + structure) used below
CITATION_GIZZIO = "Gizzio et al., 2026 (10.1042/BCJ20260137)"
CITATION_FAEZOV = "Faezov & Dunbrack, 2023 (bioRxiv)"
CITATION_MODI = "Modi & Dunbrack, 2019 (PNAS)"

# per-source metadata (archive/file name, version tier, citation, download URL) resolved into a
# Provenance record; version tiers follow priority order (v1 = current/highest priority)
DICT_SEQ_SOURCE = {
    KinCoReSeqSource.GIZZIO_2026: DataSource(
        name="kinasedomainfasta.tar.gz",
        path=PATH_FASTA_TAR,
        url=KINCORE_FASTA_URL,
        version="v1",
        citation=CITATION_GIZZIO,
    ),
    KinCoReSeqSource.FAEZOV_2023: DataSource(
        name="AF2-active.fasta",
        path=os.path.join(PATH_DATA, "AF2-active.fasta"),
        version="v2",
        citation=CITATION_FAEZOV,
    ),
    KinCoReSeqSource.MODI_2019: DataSource(
        name="Human-PK.fasta",
        path=os.path.join(PATH_DATA, "Human-PK.fasta"),
        version="v3",
        citation=CITATION_MODI,
    ),
}
"""dict[KinCoReSeqSource, DataSource]: KinCoRe kinase-domain sequence sources (priority order)."""

DICT_STRUCTURE_SOURCE = {
    KinCoReStructureSource.GIZZIO_2026: DataSource(
        name="AF2_Active_Models_v2.zip",
        path=PATH_CIF_ZIP,
        url=KINCORE_CIF_URL,
        version="v1",
        citation=CITATION_GIZZIO,
    ),
    KinCoReStructureSource.FAEZOV_2023: DataSource(
        name="Kincore_AlphaFold2_ActiveHumanCatalyticKinases",
        path=PATH_ORIG_CIF,
        version="v2",
        citation=CITATION_FAEZOV,
    ),
}
"""dict[KinCoReStructureSource, DataSource]: KinCoRe active-state structure sources (priority order)."""


def return_fasta_contents(path_filename=str) -> SeqIO.FastaIO.FastaIterator:
    return SeqIO.parse(open(path_filename), "fasta")


LIST_FASTA_KEYS1 = [
    "seq",
    "group",
    "hgnc1",
    "swissprot",
    "hgnc2",
    "uniprot",
    "start_md",
    "end_md",
    "length_md",
    "start_af2",
    "end_af2",
    "length_af2",
    "length_uniprot",
]
"""list[str]: List of FASTA keys for the AF2-active KinCoRe FASTA header."""


LIST_FASTA_KEYS2 = [
    "seq",
    "group",
    "hgnc1",
    "start_md",
    "end_md",
    "swissprot",
    "hgnc2",
    "uniprot",
]
"""list[str]: List of FASTA keys for the Human-PK (Modi-Dunbrack) KinCoRe FASTA header."""


DICT_KINCORE_PARAMS = {
    "gizzio": {
        "filename": None,  # downloaded on demand (kinasedomainfasta.tar.gz)
        "LIST_FASTA_KEYS": LIST_FASTA_KEYS1,
        "bool_af2": True,
        "seq_source": KinCoReSeqSource.GIZZIO_2026,
    },
    "faezov": {
        "filename": "AF2-active.fasta",
        "LIST_FASTA_KEYS": LIST_FASTA_KEYS1,
        "bool_af2": True,
        "seq_source": KinCoReSeqSource.FAEZOV_2023,
    },
    "modi": {
        "filename": "Human-PK.fasta",
        "LIST_FASTA_KEYS": LIST_FASTA_KEYS2,
        "bool_af2": False,
        "seq_source": KinCoReSeqSource.MODI_2019,
    },
}
"""dict[str, dict]: KinCoRe FASTA source tiers (gizzio -> faezov -> modi, priority order)."""


DICT_GROUP_KINCORE = {
    "AGC": "AGC",
    "CAMK": "CAMK",
    "CK1": "CK1",
    "CMGC": "CMGC",
    "NEK": "NEK",
    "OTHER": "Other",
    "RGC": "RGC",  # this is only in Modi-Dunbrack dataset, not AF2
    "STE": "STE",
    "TKL": "TKL",
    "TYR": "TK",
}
"""dict[str, str]: Dictionary of KinCoRe groups to map to mkt.schema.kinase_schema.Group."""


def update_original_cif_with_new_coords(
    str_orig: str,
    str_updated: str,
    str_filepath: str,
) -> None:
    """Update original CIF file with new coordinates post-alignment.

    Parameters
    ----------
    str_orig : str
        Path to original tar.gz CIF file
    str_updated : str
        Path to updated tar.gz CIF file
    str_filepath : str
        Path to save updated CIF tar.gz file

    Returns
    -------
    None
        None

    """
    _, dict_previous = untar_files_in_memory(PATH_ORIG_CIF)
    _, dict_current = untar_files_in_memory(PATH_ALIGN_CIF)

    dict_previous = {
        k.split("/")[1]: v for k, v in dict_previous.items() if k.endswith(".cif")
    }


def parse_fasta_description(
    str_description: str,
    bool_af2: bool = True,
) -> dict[str, str | int]:
    """Parse fasta description to extract metadata.

    Parameters
    ----------
    str_description : str
        Description from fasta file

    Returns
    -------
    dict[str, str]
        Dictionary of metadata
    """
    if bool_af2:
        # remove extra spaces only present in AF2-active headers
        str_description = " ".join(str_description.split())

    temp = str_description.split(" ")

    for char in ["/", "-"]:
        temp = list(chain(*[i.split(char) for i in temp]))

    temp = [
        split_on_first_only(i, "_") if idx == 0 else i for idx, i in enumerate(temp)
    ]
    temp = flatten_iterables_in_iterable(temp)

    return temp


def _resolve_kincore_kinasedomain_fasta() -> str:
    """Download the v2 kinase-domain FASTA archive on demand and return a combined FASTA path.

    Mirrors the CIF archive's local-else-fetch pattern: the ~140 KB tar of per-kinase FASTA
    files is streamed to a gitignored file under ``data/`` and concatenated once into a single
    FASTA so the existing parser can read it.

    Returns
    -------
    str
        Path to the combined kinase-domain FASTA.
    """
    if not os.path.exists(PATH_FASTA_COMBINED):
        path_tar = DICT_SEQ_SOURCE[KinCoReSeqSource.GIZZIO_2026].resolve()
        with tarfile.open(path_tar) as tf, open(PATH_FASTA_COMBINED, "wb") as out:
            for member in tf.getmembers():
                # skip macOS AppleDouble sidecars (._*), whose names also end in .fasta
                if (
                    member.isfile()
                    and member.name.endswith(".fasta")
                    and not os.path.basename(member.name).startswith("._")
                ):
                    content = tf.extractfile(member).read()
                    out.write(content)
                    # separate records: per-kinase files may lack a trailing newline
                    if not content.endswith(b"\n"):
                        out.write(b"\n")
    return PATH_FASTA_COMBINED


def extract_pk_fasta_info_as_list(
    study: str,
) -> list[KinCoReFASTA]:
    """Parse a KinCoRe FASTA source tier into KinCoReFASTA objects with source provenance.

    Parameters
    ----------
    study : str
        FASTA source tier: "gizzio" (kinasedomainfasta, downloaded on demand), "faezov"
        (AF2-active.fasta), or "modi" (Human-PK.fasta).

    Returns
    -------
    list[KinCoReFASTA]
        List of KinCoReFASTA objects, each tagged with its ``source`` Provenance.
    """
    try:
        dict_temp = DICT_KINCORE_PARAMS[study]
    except KeyError:
        logger.error(
            f"Study {study} not recognized; must be one of {list(DICT_KINCORE_PARAMS)}"
        )
        return None

    list_fasta_keys = dict_temp["LIST_FASTA_KEYS"]
    bool_af2 = dict_temp["bool_af2"]
    seq_source = dict_temp["seq_source"]

    if dict_temp["filename"] is None:
        str_path_filename = _resolve_kincore_kinasedomain_fasta()
    else:
        str_path_filename = os.path.join(get_repo_root(), "data", dict_temp["filename"])
    if not os.path.exists(str_path_filename):
        logger.error(f"File {str_path_filename} does not exist")

    provenance = DICT_SEQ_SOURCE[seq_source].provenance(str_path_filename)

    fasta_sequences = return_fasta_contents(str_path_filename)
    list_out = [
        dict(
            zip(
                list_fasta_keys,
                [str(fasta.seq)] + parse_fasta_description(fasta.description, bool_af2),
            )
        )
        for fasta in fasta_sequences
    ]

    for i in list_out:
        for k, v in i.items():
            if "start" in k or "end" in k or "length" in k:
                i[k] = int(v)
        i["group"] = DICT_GROUP_KINCORE[i["group"]]
        i["hgnc"] = {i["hgnc1"], i["hgnc2"]}
        i.pop("hgnc1")
        i.pop("hgnc2")
        i["source"] = provenance

    return [KinCoReFASTA.model_validate(i) for i in list_out]


LIST_CIF_KEYS = [
    "cif",
    "group",
    "hgnc",
    "model_confidence",
    "species",
    "state",
    "dfg_conf",
    "dihedral",
    "snc",
    "af_id",
]
"""list[str]: Metadata keys parsed from an AF2_Active_Models_v2 CIF filename
(``group_hgnc_confidence_species_state_dfg_dihedral_snc_afid.cif``), with ``cif``
prepended for the parsed mmCIF dict. ``species``/``state`` are constant (HUMAN/Active)
and dropped before building :class:`KinCoReCIF`."""


def _resolve_kincore_cif_zip() -> str:
    """Return the local v2 CIF archive path, downloading it on demand if absent.

    Mirrors the local-else-fetch pattern in :mod:`mkt.databases.oncotree`; the ~90 MB
    archive is streamed to a gitignored file under ``data/`` so it is fetched at most
    once rather than cached in the requests store.

    Returns
    -------
    str
        Path to the local ``AF2_Active_Models_v2.zip``.
    """
    return DICT_STRUCTURE_SOURCE[KinCoReStructureSource.GIZZIO_2026].resolve()


def extract_pk_cif_files_as_list() -> list[KinCoReCIF]:
    """Extract all CIF files from the KinCoRe AF2_Active_Models_v2 archive.

    Returns
    -------
    list[KinCoReCIF]
        List of KinCoReCIF objects (one active-state model per kinase domain).
    """
    path_zip = _resolve_kincore_cif_zip()
    provenance = DICT_STRUCTURE_SOURCE[KinCoReStructureSource.GIZZIO_2026].provenance(
        path_zip
    )

    list_out = []
    with zipfile.ZipFile(path_zip) as zf:
        # the archive holds the descriptive per-domain models at the top level plus a
        # redundant flat copy of the same structures under a nested <author>/ subdir
        # (e.g. awar04/AF-P08631-K3.cif); keep only the descriptive top-level CIFs (one
        # nesting level below the archive root), skipping macOS AppleDouble (._) sidecars
        list_name = [
            n
            for n in zf.namelist()
            if n.endswith(".cif")
            and n.count("/") == 1
            and not os.path.basename(n).startswith("._")
        ]
        for name in tqdm(
            list_name,
            desc="Extracting and processing CIF files...",
            bar_format=TQDM_BAR_FORMAT,
        ):
            list_token = os.path.basename(name)[:-4].split("_")
            cif = MMCIF2Dict(io.StringIO(zf.read(name).decode("utf-8")))
            dict_temp = dict(zip(LIST_CIF_KEYS, [cif] + list_token))
            dict_temp["group"] = DICT_GROUP_KINCORE[dict_temp["group"]]
            dict_temp["model_confidence"] = float(dict_temp["model_confidence"])
            # species/state are constant (HUMAN/Active) and not KinCoReCIF fields
            dict_temp.pop("species")
            dict_temp.pop("state")
            dict_temp["source"] = provenance
            list_out.append(dict_temp)

    return [KinCoReCIF.model_validate(v) for v in list_out]


def align_kincore2uniprot(
    str_kincore: str,
    str_uniprot: str,
) -> dict[str, dict[str, str | int | list[int] | None]]:
    """Align KinCoRe Human-PK.fasta to canonical Uniprot sequences.

    Parameters
    ----------
    str_kicore : str
        KinCoRe sequence
    str_uniprot : str
        Uniprot sequence

    Returns
    -------
    dict[str, dict[str, str | None]]
        Dictionary of {start : int | None, end : int, mismatch : list[int]}
    """

    dict_out = dict.fromkeys(["seq", "start", "end", "mismatch"])
    dict_out["seq"] = str_kincore

    aligner = Kincore2UniProtAligner()
    alignments = aligner.align(str_kincore, str_uniprot)

    # if multiple alignments, return None
    if len(alignments) != 1:
        logger.warning(f"Multiple alignments found for {str_kincore} and {str_uniprot}")
        return dict_out

    alignment = alignments[0]

    # if alignment does not include full sequence, None
    if alignment.sequences[0] != alignment[0, :]:
        logger.warning(
            "Alignment does not include full sequence "
            f"for {str_kincore} and {str_uniprot}"
        )
        pass

    start = int(alignment.aligned[1][0][0])
    dict_out["start"] = start + 1

    end = int(alignment.aligned[1][0][1])
    dict_out["end"] = end

    # if mismatch, provide idx of mismatch in KinCoRe sequence
    str_align = "".join(
        [
            i.split(" ")[-1]
            for idx, i in enumerate(str(alignment).split("\n"))
            if (idx + 1) % 2 == 0
        ]
    )
    str_align = re.sub(r"[a-zA-Z0-9]", "", str_align)
    if "." in str_align:
        dict_out["mismatch"] = [idx for idx, i in enumerate(str_align) if i == "."]

    return dict_out


def harmonize_kincore_fasta_cif():
    """Harmonize KinCoRe FASTA/CIF sources into KinCoRe objects with per-entry provenance.

    Builds the AF2 sequence tier hierarchically -- the latest (Gizzio) kinase-domain FASTA,
    falling back to Faezov for kinases it dropped (e.g. SGK3, whose v2 CIF still exists) -- then
    matches each to its active-state CIF, and finally adds Modi-only kinases (no active-state
    structure) with ``cif=None``. Every FASTA/CIF record carries its ``source`` Provenance.

    Returns
    -------
    dict[str, list[KinCoRe]]
        Dictionary of {uniprot : list[KinCoRe]}
    """
    list_gizzio_fasta = extract_pk_fasta_info_as_list("gizzio")
    list_faezov_fasta = extract_pk_fasta_info_as_list("faezov")
    list_md_fasta = extract_pk_fasta_info_as_list("modi")
    list_kincore_cif = extract_pk_cif_files_as_list()

    # AF2 sequence tier: prefer Gizzio, fall back to Faezov for uniprots Gizzio dropped
    set_gizzio_uniprot = {i.uniprot for i in list_gizzio_fasta}
    list_af2_fasta = list_gizzio_fasta + [
        i for i in list_faezov_fasta if i.uniprot not in set_gizzio_uniprot
    ]

    dict_kincore = {}

    # process AF2-active dataset
    list_af2_uniprot = [i.uniprot for i in list_af2_fasta]
    list_cif_hgnc_split = [
        try_except_split_concat_str(i.hgnc, idx1=0, idx2=1) for i in list_kincore_cif
    ]
    # multi-kinase domain (AF2)
    list_multi = [
        item for item, count in Counter(list_af2_uniprot).items() if count > 1
    ]
    for uniprot in list_multi:
        fastas = [i for i in list_af2_fasta if i.uniprot == uniprot]
        list_temp = []
        for fasta in fastas:
            hgnc_fasta = max(fasta.hgnc, key=len)
            idx = list_cif_hgnc_split.index(hgnc_fasta)
            cif = list_kincore_cif[idx]
            list_temp.append(KinCoRe(fasta=fasta, cif=cif))
        dict_kincore[uniprot] = list_temp
    # single kinase domain (AF2)
    for uniprot in list_af2_uniprot:
        fasta = [i for i in list_af2_fasta if i.uniprot == uniprot]
        # don't re-incorporate multi-mapping
        if len(fasta) == 1:
            hgnc = fasta[0].hgnc  # use whole set for CILK1/ILK
            try:
                idx = [idx for idx, i in enumerate(list_cif_hgnc_split) if i in hgnc][0]
                cif = list_kincore_cif[idx]
                temp = KinCoRe(fasta=fasta[0], cif=cif)
            except IndexError:
                temp = KinCoRe(fasta=fasta[0], cif=None)
            dict_kincore[uniprot] = [temp]

    # process Modi-Dunbrack dataset
    list_md_only_uniprot = [
        i.uniprot for i in list_md_fasta if i.uniprot not in list_af2_uniprot
    ]
    # MD genes only - there are no multi-KD
    for uniprot in list_md_only_uniprot:
        fasta = [i for i in list_md_fasta if i.uniprot == uniprot]
        if len(fasta) == 1:
            temp = KinCoRe(fasta=fasta[0], cif=None)
        else:
            logger.warning(
                f"{uniprot} has multipe FASTA entries in Modi-Dunbrack dataset\n{fasta}\n"
            )
        dict_kincore[uniprot] = [temp]

    return dict_kincore
