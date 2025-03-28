import glob
import json
import logging
import os
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
        "deserialize": json.load,
    },
    "yaml": {
        "serialize": yaml.safe_dump,
        "deserialize": yaml.safe_load,
    },
    "toml": {
        "serialize": toml.dumps,
        "deserialize": toml.loads,
    },
}
"""dict[str, dict[str, Callable]]: Dictionary of serialization and deserialization functions supported."""


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
    if not os.path.exists(str_path_in) or len(os.listdir(str_path_in)) == 0:
        return FileNotFoundError
    else:
        return None


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
        pkg_resource = "KinaseInfo"

    if str_path is None:
        try:
            str_path = pkg_resources.resource_filename(pkg_name, pkg_resource)
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
        if str_path is None and suffix == "json":
            serialization_kwargs = {"indent": 4}
        else:
            serialization_kwargs = {}

    str_path = return_str_path_from_pkg_data(str_path)

    for key, val in tqdm(kinase_dict.items()):
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
        deserialization_kwargs = {}

    str_path = return_str_path_from_pkg_data(str_path)
    list_file = glob.glob(os.path.join(str_path, f"*.{suffix}"))

    dict_import = {}
    for file in tqdm(list_file):
        with open(file) as openfile:
            # toml files are read as strings
            if suffix == "toml":
                openfile = openfile.read()

            val_deserialized = DICT_FUNCS[suffix]["deserialize"](
                openfile,
                **deserialization_kwargs,
            )

            kinase_obj = kinase_schema.KinaseInfo.model_validate(val_deserialized)
            dict_import[kinase_obj.hgnc_name] = kinase_obj

    dict_import = {key: dict_import[key] for key in sorted(dict_import.keys())}

    return dict_import
