import glob
import json
import logging
import os
import shutil
import tarfile
from io import BytesIO
from typing import Any, Optional

import pkg_resources
import toml
import yaml
from mkt.schema import kinase_schema
from pydantic import BaseModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


DICT_FUNCS = {
    "json": {
        "serialize": json.dumps,
        "kwargs_serialize": {"default": list, "indent": 4},
        "deserialize_file": json.load,
        "deserialize_str": json.loads,
        "kwargs_deserialize": {},
    },
    "yaml": {
        "serialize": yaml.safe_dump,
        "kwargs_serialize": {"sort_keys": False},
        "deserialize_file": yaml.safe_load,
        "deserialize_str": yaml.safe_load,
        "kwargs_deserialize": {},
    },
    "toml": {
        "serialize": toml.dumps,
        "kwargs_serialize": {},
        "deserialize_file": toml.load,
        "deserialize_str": toml.loads,
        "kwargs_deserialize": {},
    },
}
"""dict[str, dict[str, Callable]]: Dictionary of serialization and deserialization functions supported."""


def extract_tarfiles(path_from, path_to):
    """Extract tar.gz files.

    Parameters
    ----------
    path_from : str
        Path to the tar.gz file
    path_to : str
        Pth to extract the files to

    Returns
    -------
    None
        None

    """
    import tarfile

    try:
        with tarfile.open(path_from, "r:gz") as tar:
            tar.extractall(path_to)
    except Exception as e:
        logger.error(f"Exception {e}")


def return_filenotfound_error_if_empty_or_missing(
    str_path_in: str,
) -> FileNotFoundError | None:
    """Return FileNotFoundError for the given path.

    Parameters
    ----------
    str_path_in : str
        Path that was not found.

    Returns
    -------
    FileNotFoundError | None
        FileNotFoundError for the given path. If the path is not empty, return None.
    """
    if os.path.exists(str_path_in) and str_path_in.endswith(".tar.gz"):
        logger.info(
            f"File {str_path_in} exists as a tar.gz directory. Will extract in memory..."
        )
        return None
    elif not os.path.exists(str_path_in) or len(os.listdir(str_path_in)) == 0:
        return FileNotFoundError
    elif os.path.exists(str_path_in) and str_path_in.endswith(".tar.gz"):
        logger.info(
            f"File {str_path_in} exists as a tar.gz directory. Will extract in memory..."
        )
        return None
    else:
        return None


def untar_if_neeeded(str_filename: str) -> str:
    """Unzip the file if it is a zip file.

    Parameters
    ----------
    str_filename : str
        Path to the file.

    Returns
    -------
    str
        Path to the unzipped file or original file if not .tar.gz.
    """
    if str_filename.endswith(".tar.gz"):

        str_path_extract = os.path.dirname(str_filename)
        extract_tarfiles(str_filename, str_path_extract)
        str_filename = str_filename.replace(".tar.gz", "")

    return str_filename


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

    list_entries, dict_btyes = [], {}
    with BytesIO(tar_data) as tar_buffer, tarfile.open(
        fileobj=tar_buffer, mode="r"
    ) as tar:
        for member in tar.getmembers():
            filename = os.path.basename(member.name)
            # make sure entry is file
            cond1 = member.isfile()
            # ignore MacOS AppleDouble files
            cond2 = "._" not in filename
            # use list_ids, if provided
            cond3 = list_ids is None or filename.split(".")[0] in list_ids
            if cond1 and cond2 and cond3:
                list_entries.append(filename.split(".")[0])
                if bool_extract:
                    with tar.extractfile(member) as f:
                        dict_btyes[member.name] = f.read()

    if bool_extract:
        # decode bytes to string
        dict_btyes = {k: v.decode("utf-8") for k, v in dict_btyes.items()}

    return list_entries, dict_btyes


def return_str_path_from_pkg_data(
    str_path: str | None = None,
    pkg_name: str | None = None,
    pkg_resource: str | None = None,
) -> str:
    """Return the path to the package data directory or of a user-provided directory.

    Parameters
    ----------
    str_path : str | None, optional
        Path to the KinaseInfo directory, by default None.
    pkg_name : str | None, optional
        Package name, by default None and will use mkt.schema.
    pkg_resource : str | None, optional
        Package resource, by default None and will use KinaseInfo.

    Returns
    -------
    str
        Path to the package data or user-provided directory.
    """
    if pkg_name is None:
        pkg_name = "mkt.schema"
    if pkg_resource is None:
        pkg_resource = "KinaseInfo.tar.gz"

    if str_path is None:
        try:
            str_path = pkg_resources.resource_filename(pkg_name, pkg_resource)
            # str_path = untar_if_neeeded(str_path)
            return_filenotfound_error_if_empty_or_missing(str_path)
        except Exception as e:
            logger.error(
                f"Could not find {pkg_resource} directory within {pkg_name}: {e}"
                f"\nPlease provide a path to the {pkg_resource} directory."
            )
    else:
        if not os.path.exists(str_path):
            os.makedirs(str_path)
    return str_path


def clean_files_and_delete_directory(list_files: list[str]) -> None:
    """Remove unzipped files.

    Parameters
    ----------
    list_files : list[str]
        List of files to remove.

    Returns
    -------
    None
        None

    """
    try:
        paths_remove = {os.path.dirname(i) for i in list_files}
        [shutil.rmtree(i) for i in paths_remove if os.path.isdir(i)]
        logger.info(f"Removed unzipped files: {[i for i in paths_remove]}.")
    except Exception as e:
        logger.error(f"Exception {e}")
        logger.info(f"Could not remove unzipped files: {list_files}.")


# adapted from https://axeldonath.com/scipy-2023-pydantic-tutorial/notebooks-rendered/4-serialisation-and-deserialisation.html
def serialize_kinase_dict(
    kinase_dict: dict[str, BaseModel],
    suffix: str = "json",
    serialization_kwargs: Optional[dict[str, Any]] = None,
    str_path: str | None = None,
):
    """Serialize KinaseInfo object to files.

    Parameters
    ----------
    kinase_dict : dict[str, BaseModel]
        Dictionary of KinaseInfo objects.
    suffix : str
        Serialization types supported: json, yaml, toml.
    serialization_kwargs : dict[str, Any], optional
        Additional keyword arguments for serialization function, by default None;
            (e.g., {"indent": 2} for json.dumps, {"sort_keys": False} for yaml.safe_dump).
    str_path: str | None = None
        Path to save the serialized file, by default None will use package data or Github repo data.
    """
    if suffix not in DICT_FUNCS:
        logger.error(
            f"Serialization type ({suffix}) not supported; must be json, yaml, or toml."
        )
        return None

    if os.name == "nt" and suffix == "toml":
        logger.info("TOML serialization is not supported on Windows.")
        return None

    if serialization_kwargs is None:
        serialization_kwargs = DICT_FUNCS[suffix]["kwargs_serialize"]

    str_path = return_str_path_from_pkg_data(str_path)

    for key, val in tqdm(kinase_dict.items(), desc="Serializing KinaseInfo objects..."):
        with open(f"{str_path}/{key}.{suffix}", "w") as outfile:
            val_serialized = DICT_FUNCS[suffix]["serialize"](
                val.model_dump(),
                **serialization_kwargs,
            )
            outfile.write(val_serialized)


def deserialize_kinase_dict(
    suffix: str = "json",
    deserialization_kwargs: Optional[dict[str, Any]] = None,
    str_path: str | None = None,
    bool_remove: bool = True,
    list_ids: list[str] | None = None,
) -> dict[str, BaseModel]:
    """Deserialize KinaseInfo object from files.

    Parameters
    ----------
    suffix : str
        Deserialization types supported: json, yaml, toml.
    deserialization_kwargs : dict[str, Any], optional
        Additional keyword arguments for deserialization function, by default None.
    str_path : str | None, optional
        Path from which to load files, by default None.
    bool_remove : bool, optional
        If True, remove the files after deserialization, by default True.
    list_ids : list[str] | None, optional
        List of IDs to filter the files if reading from memory, by default None.

    Returns
    -------
    dict[str, KinaseInfo]
        Dictionary of KinaseInfo objects.
    """
    if suffix not in DICT_FUNCS:
        logger.error(
            f"Serialization type ({suffix}) not supported; must be json, yaml, or toml."
        )
        return None

    if str_path is None and suffix != "json":
        logger.error("Only json deserialization is supported without providing a path.")
        return None

    if deserialization_kwargs is None:
        deserialization_kwargs = DICT_FUNCS[suffix]["kwargs_deserialize"]

    str_path = return_str_path_from_pkg_data(str_path)

    dict_import = {}
    if str_path.endswith(".tar.gz"):
        dict_str = untar_files_in_memory(str_path, list_ids=list_ids)[1]
        for val in tqdm(
            dict_str.values(), desc="Deserializing KinaseInfo objects in memory..."
        ):

            val_deserialized = DICT_FUNCS[suffix]["deserialize_str"](
                val,
                **deserialization_kwargs,
            )

            kinase_obj = kinase_schema.KinaseInfo.model_validate(val_deserialized)
            dict_import[kinase_obj.hgnc_name] = kinase_obj
    else:
        list_file = glob.glob(os.path.join(str_path, f"*.{suffix}"))
        for file in tqdm(
            list_file, desc="Deserializing KinaseInfo objects from files..."
        ):
            with open(file) as openfile:

                val_deserialized = DICT_FUNCS[suffix]["deserialize_file"](
                    openfile,
                    **deserialization_kwargs,
                )

                kinase_obj = kinase_schema.KinaseInfo.model_validate(val_deserialized)
                dict_import[kinase_obj.hgnc_name] = kinase_obj

        if bool_remove:
            clean_files_and_delete_directory(list_file)

    dict_import = {key: dict_import[key] for key in sorted(dict_import.keys())}

    return dict_import
