import logging
import os
import tarfile
from io import BytesIO

from mkt.schema.io_utils import get_repo_root

logger = logging.getLogger(__name__)


PATH_DATA = os.path.join(get_repo_root(), "data")


def get_parser():
    """Parse arguments for the alignment script.

    Returns
    -------
    parser : argparse.ArgumentParser
        Argument parser for the alignment script.

    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Align multiple CIF files to a reference PDB structure"
    )

    parser.add_argument(
        "--referencePDB",
        default=os.path.join(PATH_DATA, "1gag_template.pdb"),
        help="Reference PDB file to which all CIF files will be aligned",
    )

    parser.add_argument(
        "--inpuTar",
        default=os.path.join(
            PATH_DATA, "Kincore_AlphaFold2_ActiveHumanCatalyticKinases_v2.tar.gz"
        ),
        help="Tar gzipped directory containing CIF files",
    )

    parser.add_argument(
        "--outputTar",
        default=os.path.join(
            PATH_DATA,
            "Kincore_AlphaFold2_ActiveHumanCatalyticKinases_v2_ALIGNED.tar.gz",
        ),
        help="Directory to save aligned CIF files as tar gzipped files",
    )

    parser.add_argument(
        "--molecSelection",
        default="all",
        help="PyMOL selection for alignment (default: `all`, use `name CA` for C-alpha atoms)",
    )

    parser.add_argument(
        "--methodAlign",
        default="align",
        choices=["align", "super", "cealign"],
        help="Alignment method to use (default: align)",
    )

    return parser


def untar_files_in_memory(
    str_path: str,
    bool_extract: bool = True,
    list_ids: list[str] | None = None,
) -> dict[str, str]:
    """Untar files exclusively in memory.

    Parameters
    ----------
    str_path : str
        Path to the tar.gz file.

    Returns
    -------
    dict[str, str]
        Dictionary of file names and their contents as strings.

    """
    with open(str_path, "rb") as f:
        tar_data = f.read()

    list_entries, dict_bytes = [], {}
    with BytesIO(tar_data) as tar_buffer, tarfile.open(
        fileobj=tar_buffer, mode="r"
    ) as tar:
        for member in tar.getmembers():
            filename = os.path.basename(member.name)
            # make sure entry is file
            cond1 = member.isfile()
            # ignore MacOS AppleDouble files
            cond2 = "._" not in filename
            # use list_ids, if provided; if None also True
            cond3 = list_ids is None or filename.split(".")[0] in list_ids
            if cond1 and cond2 and cond3:
                list_entries.append(filename.split(".")[0])
                if bool_extract:
                    with tar.extractfile(member) as f:
                        dict_bytes[member.name] = f.read()

    if bool_extract:
        # decode bytes to string
        dict_bytes = {k: v.decode("utf-8") for k, v in dict_bytes.items()}

    return list_entries, dict_bytes


def create_tar_without_metadata(
    path_source: str,
    filename_tar: str,
) -> None:
    """Create a tar file without metadata.

    Parameters
    ----------
    path_source : str
        Path to the source directory to be tarred
    filename_tar : str
        Path and filename to the save the tar file

    Returns
    -------
    None

    """
    # check if the source directory exists
    if not os.path.exists(path_source):
        logging.error(f"Source directory {path_source} does not exist.")
    # check if the source directory is a directory
    if not os.path.isdir(path_source):
        logging.error(f"Source path {path_source} is not a directory.")
    # check if the output tar file already exists
    if os.path.exists(filename_tar):
        logging.error(f"Output tar file {filename_tar} already exists.")

    with tarfile.open(filename_tar, "w:gz") as tar:
        for root, _, files in os.walk(path_source):
            for file in files:
                file_path = os.path.join(root, file)
                if not file.startswith("._"):
                    tar.add(file_path, arcname=os.path.relpath(file_path, path_source))
