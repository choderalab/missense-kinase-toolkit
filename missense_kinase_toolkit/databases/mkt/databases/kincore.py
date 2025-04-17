# import glob
import logging
import os
import re

# import shutil
import tempfile
from collections import Counter
from io import StringIO
from itertools import chain

from Bio import SeqIO

# from biotite.structure.io.pdbx import CIFFile
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO
from mkt.databases.aligners import Kincore2UniProtAligner
from mkt.databases.colors import map_aa_to_single_letter_code
from mkt.databases.io_utils import create_tar_without_metadata, get_repo_root
from mkt.databases.utils import (
    flatten_iterables_in_iterable,
    split_on_first_only,
    try_except_split_concat_str,
)
from mkt.schema.io_utils import untar_files_in_memory

# from mkt.schema.io_utils import extract_tarfiles, untar_files_in_memory
from mkt.schema.kinase_schema import KinCore, KinCoreCIF, KinCoreFASTA
from tqdm import tqdm

logger = logging.getLogger(__name__)


PATH_DATA = os.path.join(get_repo_root(), "data")
"""str: Path to the data directory."""
PATH_ORIG_CIF = os.path.join(
    PATH_DATA, "Kincore_AlphaFold2_ActiveHumanCatalyticKinases_v2.tar.gz"
)
"""str: Path to the original CIF file."""
PATH_ALIGN_CIF = os.path.join(
    PATH_DATA, "Kincore_AlphaFold2_ActiveHumanCatalyticKinases_v2_aligned.tar.gz"
)
"""str: Path to the aligned CIF file."""


# LIST_CIF_DICT_UPDATE = [
#     "_atom_site.Cartn_x",
#     "_atom_site.Cartn_y",
#     "_atom_site.Cartn_z",
#     "_atom_site.label_atom_id",
#     "_atom_site.label_seq_id",
#     "_atom_site.pdbx_formal_charge",
#     "_atom_site.type_symbol",
# ]
# """list[str]: List of CIF keys to update with aligned values."""


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
    "source_file",
]
"""list[str]: List of FASTA keys for KinCore FASTA file."""


LIST_FASTA_KEYS2 = [
    "seq",
    "group",
    "hgnc1",
    "start_md",
    "end_md",
    "swissprot",
    "hgnc2",
    "uniprot",
    "source_file",
]
"""list[str]: List of FASTA keys for KinCore FASTA file."""


DICT_KINCORE_PARAMS = {
    "af2": {
        "filename": "AF2-active.fasta",
        "LIST_FASTA_KEYS": LIST_FASTA_KEYS1,
        "bool_af2": True,
        "study": "Faezov-Dunbrack_2023",
    },
    "md": {
        "filename": "Human-PK.fasta",
        "LIST_FASTA_KEYS": LIST_FASTA_KEYS2,
        "bool_af2": False,
        "study": "Modi-Dunbrack_2019",
    },
}
"""dict[str, dict[str, str | list[str]]]: Dictionary of KinCore parameters for FASTA files."""


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
"""dict[str, str]: Dictionary of KinCore groups to map to mkt.schema.kinase_schema.Group."""


def return_fasta_contents(path_filename=str) -> SeqIO.FastaIO.FastaIterator:
    return SeqIO.parse(open(path_filename), "fasta")


def cif_string_to_mmcif_dict(cif_string):
    """
    Convert a string representation of a CIF file to an MMCIF2Dict object.

    Parameters:
    -----------
    cif_string : str
        String representation of the CIF file content

    Returns:
    --------
    dict
        Dictionary representation of the CIF file (MMCIF2Dict)

    """
    # Create a temporary file with the CIF content
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".cif", delete=True
    ) as temp_file:
        # Write the CIF string to the temporary file
        temp_file.write(cif_string)
        temp_file.flush()  # Ensure all data is written to disk

        # Rewind to the beginning of the file
        temp_file.seek(0)

        # Parse the temporary file into an MMCIF2Dict object
        mmcif_dict = MMCIF2Dict(temp_file.name)

    # The temporary file is automatically deleted when the context manager exits
    return mmcif_dict


def process_tar_gz_cif_files(cif_files_dict):
    """
    Process a dictionary of CIF files extracted from a tar.gz archive.

    Parameters:
    -----------
    cif_files_dict : dict
        Dictionary with filenames as keys and CIF file contents as string values

    Returns:
    --------
    dict
        Dictionary with filenames as keys and MMCIF2Dict objects as values

    """
    result = {}
    for filename, cif_string in tqdm(
        cif_files_dict.items(), desc="Converting extracted CIF strings to MMCIF2Dict..."
    ):
        # remove path if included in filename
        if "/" in filename:
            filename = filename.split("/")[1]
        if filename.endswith("cif"):
            result[filename] = cif_string_to_mmcif_dict(cif_string)
    return result


def save_mmcif_dicts_to_tar_gz(mmcif_dict_objects, output_tar_path):
    """
    Save a dictionary of MMCIF2Dict objects to a tar.gz file.

    Parameters:
    -----------
    mmcif_dict_objects : dict
        Dictionary with filenames as keys and MMCIF2Dict objects as values
    output_tar_path : str
        Path where the output tar.gz file will be saved

    Returns:
    --------
    None

    """
    # Create a temporary directory to store the CIF files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write each MMCIF2Dict object to a CIF file in the temporary directory
        for filename, mmcif_dict in mmcif_dict_objects.items():
            # Ensure the filename has .cif extension
            if not filename.endswith(".cif"):
                filename = f"{filename}.cif"

            file_path = os.path.join(temp_dir, filename)

            # Convert MMCIF2Dict to string and write to file
            try:
                mmcif_io = MMCIFIO()
                mmcif_io.set_dict(mmcif_dict)
                temp_string = StringIO()
                mmcif_io.save(temp_string)
                # cif_string = mmcif_dict_to_cif_string(mmcif_dict)
                with open(file_path, "w") as f:
                    f.write(temp_string.getvalue())
            except Exception as e:
                logger.error(f"Error writing {filename}: {str(e)}")
                continue

        # Create a tar.gz file from the temporary directory
        create_tar_without_metadata(temp_dir, output_tar_path)

    logger.info(f"Created {output_tar_path} with {len(mmcif_dict_objects)} CIF files")


def update_original_cif_with_new_coords(
    str_filepath: str,
    path_orig: str | None = None,
    path_updated: str | None = None,
) -> None:
    """Update original CIF file with new coordinates post-alignment.

    Parameters
    ----------
    str_filepath : str
        Path to save updated CIF tar.gz file;
        if not prepended with directory, will save to PATH_DATA
    str_orig : str
        Path to original tar.gz CIF file
    str_updated : str
        Path to updated tar.gz CIF file

    Returns
    -------
    None
        None

    """
    if path_orig is None:
        path_orig = PATH_ORIG_CIF
    if path_updated is None:
        path_updated = PATH_ALIGN_CIF

    _, dict_previous = untar_files_in_memory(path_orig)
    _, dict_current = untar_files_in_memory(path_updated)

    dict_previous_mmcif = process_tar_gz_cif_files(dict_previous)
    dict_current_mmcif = process_tar_gz_cif_files(dict_current)

    dict_harmonized = {}
    for k1, v1 in dict_previous_mmcif.items():
        dict_temp = {}
        for k2, v2 in v1.items():
            if k2 not in ["data_", "_entry.id"] and k2 in dict_current_mmcif[k1]:
                dict_temp[k2] = dict_current_mmcif[k1][k2]
            else:
                dict_temp[k2] = v2
        dict_harmonized[k1] = dict_temp

    if "/" not in str_filepath:
        str_filepath = os.path.join(PATH_DATA, str_filepath)
    else:
        if not os.path.exists(os.path.dirname(str_filepath)):
            logger.info(
                f"Creating directory {os.path.dirname(str_filepath)} for output file..."
            )
            os.makedirs(os.path.dirname(str_filepath))

    save_mmcif_dicts_to_tar_gz(dict_harmonized, str_filepath)


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


def extract_pk_fasta_info_as_list(
    study: str,
) -> list[KinCoreFASTA]:
    """Parse KinCore Human-PK.fasta file to extract information for KinaseInfo object.

    Parameters
    ----------
    study : str
        Study FASTA to use; options are "af2" (Faezov-Dunbrack, 2023) or "md" (Modi-Dunbrack, 2019)

    Returns
    -------
    list[KinCoreFASTA]
        List of KinCoreFASTA objects
    """
    try:
        dict_temp = DICT_KINCORE_PARAMS[study]
        str_filename = dict_temp["filename"]
        list_fasta_keys = dict_temp["LIST_FASTA_KEYS"]
        bool_af2 = dict_temp["bool_af2"]
        study = dict_temp["study"]
    except KeyError:
        logger.error(f"Study {study} not recognized; must be 'af2' or 'md'")
        return None

    str_path_filename = os.path.join(PATH_DATA, str_filename)
    if not os.path.exists(str_path_filename):
        logger.error(f"File {str_path_filename} does not exist")

    fasta_sequences = return_fasta_contents(str_path_filename)
    list_out = [
        dict(
            zip(
                list_fasta_keys,
                [str(fasta.seq)]
                + parse_fasta_description(fasta.description, bool_af2)
                + [study],
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

    list_out = [KinCoreFASTA.model_validate(i) for i in list_out]

    return list_out


LIST_CIF_KEYS = [
    "cif",
    "group",
    "hgnc",
    "min_aloop_pLDDT",
    "template_source",
    "msa_size",
    "msa_source",
    "model_no",
]
"""list[str]: List of CIF keys for KinCore CIF file."""


def extract_pk_cif_files_as_list(bool_align: False) -> list[KinCoreCIF]:
    """Extract all cif files from KinCore directory.

    Parameters
    ----------
    bool_align: bool
        Whether to use original or aligned CIF files.s

    Returns
    -------
    list[KinCoreCIF]
        List of KinCoreCIF objects
    """
    if bool_align:
        path_use = PATH_ALIGN_CIF
    else:
        path_use = PATH_ORIG_CIF
    if not os.path.exists(path_use):
        logger.error(
            f"KinCore tar.gz file not found in {path_use}..."
            "File can be downloaded from: http://dunbrack.fccc.edu/kincore/static/downloads/af2activemodels/Kincore_AlphaFold2_ActiveHumanCatalyticKinases_v2.tar.gz"
        )

    _, dict_untar = untar_files_in_memory(PATH_ORIG_CIF)
    dict_untar_cif = process_tar_gz_cif_files(dict_untar)

    # extract_tarfiles(PATH_ALIGN_CIF, PATH_DATA)

    # list_file = glob.glob(os.path.join(PATH_DATA, "*", "*.cif"))

    list_out = []
    for filename, dict_mmcif in tqdm(
        dict_untar_cif.items(), desc="Extracting and processing CIF files..."
    ):
        # use the filename to extract metadata as dict
        filename = filename.split("/")[-1]
        # filename = os.path.basename(file)
        if filename == "TYR_LMTK2_38.37_tea2MSA_AF2tholog_model1.cif":
            # TODO: confirm this is the correct filename with Dunbrack lab
            filename = "TYR_LMTK2_38.37_activeAF2_2MSA_ortholog_model1.cif"
        list_filename = filename.replace(".cif", "").replace("__", "_").split("_")
        # cif_file = CIFFile.read(file)
        # dict_mmcif = MMCIF2Dict(file)
        if "_entity_poly.pdbx_seq_one_letter_code" not in dict_mmcif:
            str_seq = ""
            for atom_id, comp_id in zip(
                dict_mmcif["_atom_site.label_atom_id"],
                dict_mmcif["_atom_site.label_comp_id"],
            ):
                if atom_id == "CA":
                    str_seq += map_aa_to_single_letter_code(comp_id)
            dict_mmcif["_entity_poly.pdbx_seq_one_letter_code"] = str_seq
        dict_temp = dict(
            zip(
                LIST_CIF_KEYS,
                # [cif_file.serialize()] + list_filename
                [dict_mmcif] + list_filename,
            )
        )
        list_out.append(dict_temp)

    for v in list_out:
        v["group"] = DICT_GROUP_KINCORE[v["group"]]
        v["min_aloop_pLDDT"] = float(v["min_aloop_pLDDT"])
        v["msa_size"] = int(v["msa_size"].replace("MSA", ""))
        v["model_no"] = int(v["model_no"].replace("model", ""))

    list_out = [KinCoreCIF.model_validate(v) for v in list_out]

    # remove unzipped directory and all contents
    # paths_remove = {os.path.dirname(i) for i in list_file}
    # [shutil.rmtree(i) for i in paths_remove if os.path.isdir(i)]

    return list_out


def align_kincore2uniprot(
    str_kincore: str,
    str_uniprot: str,
) -> dict[str, dict[str, str | int | list[int] | None]]:
    """Align KinCore Human-PK.fasta to canonical Uniprot sequences.

    Parameters
    ----------
    str_kicore : str
        KinCore sequence
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

    # if mismatch, provide idx of mismatch in KinCore sequence
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


def harmonize_kincore_fasta_cif(bool_aligned=False):
    """Harmonize KinCore FASTA/CIF files for af2/md and generate KinCore objects.

    Returns
    -------
    dict[str, list[KinCore]]
        Dictionary of {uniprot : list[KinCore]}
    """
    list_af2_fasta = extract_pk_fasta_info_as_list("af2")
    list_md_fasta = extract_pk_fasta_info_as_list("md")
    list_kincore_cif = extract_pk_cif_files_as_list()

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
            list_temp.append(KinCore(fasta=fasta, cif=cif))
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
                temp = KinCore(fasta=fasta[0], cif=cif)
            except IndexError:
                temp = KinCore(fasta=fasta[0], cif=None)
            dict_kincore[uniprot] = [temp]

    # process Modi-Dunbrack dataset
    list_md_only_uniprot = [
        i.uniprot for i in list_md_fasta if i.uniprot not in list_af2_uniprot
    ]
    # MD genes only - there are no multi-KD
    for uniprot in list_md_only_uniprot:
        fasta = [i for i in list_md_fasta if i.uniprot == uniprot]
        if len(fasta) == 1:
            temp = KinCore(fasta=fasta[0], cif=None)
        else:
            logger.warning(
                f"{uniprot} has multipe FASTA entries in Modi-Dunbrack dataset\n{fasta}\n"
            )
        dict_kincore[uniprot] = [temp]

    return dict_kincore
