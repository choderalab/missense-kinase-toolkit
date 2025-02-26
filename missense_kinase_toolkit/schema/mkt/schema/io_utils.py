import logging
from typing import Any, Callable, Optional
from os import path
import glob
import git
import pkg_resources
from tqdm import tqdm
import json
import toml
import yaml
from pydantic import BaseModel

from mkt.schema import kinase_schema

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

def get_repo_root() -> str | None:
    """Get the root directory of the repository.

    Returns
    -------
    str
        Path to the root directory of the repository.
    """
    try:
        repo = git.Repo(".", search_parent_directories=True)
        return repo.working_tree_dir
    except git.InvalidGitRepositoryError:
        return None

def return_str_path(str_path: str | None = None) -> str:
    """Return the path to the KinaseInfo directory.

    Parameters
    ----------
    str_path : str | None, optional
        Path to the KinaseInfo directory, by default None.

    Returns
    -------
    str
        Path to the KinaseInfo directory.
    """
    if str_path is None:
        # first look in package resources
        try:
            str_path = pkg_resources.resource_filename(
                "mkt.schema", "KinaseInfo"
            )
        except Exception as e:
            logger.warning(
                f"Could not find KinaseInfo directory within package; {e}"
                f"\nPlease provide a path to the KinaseInfo directory."
            )
            try:
                # otherwise look in root of github repo
                str_path = path.join(
                    get_repo_root(),
                    "missense_kinase_toolkit/mkt/schema/KinaseInfo",
                )
            except Exception as e:
                logger.error(
                    f"Could not find KinaseInfo directory in the repository; {e}"
                    f"\nPlease provide a path to the KinaseInfo directory."
                )
    else:
        if not path.exists(str_path):
            path.os.makedirs(str_path)

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
        return

    str_path = return_str_path(str_path)

    if serialization_kwargs is None:
        serialization_kwargs = {}

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
        return

    if str_path is None and suffix != "json":
        logger.error(
            "Only json deserialization is supported without providing a path."
        )
        return

    str_path = return_str_path(str_path)
    list_file = glob.glob(path.join(str_path, f"*.{suffix}"))

    if deserialization_kwargs is None:
        deserialization_kwargs = {}

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
