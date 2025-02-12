from os import path
import glob
import json
from typing import Any, Callable, Optional
import pkg_resources
from pydantic import BaseModel 
import json
import yaml
import toml
import git
import logging

from mkt.schema import kinase_schema

logger = logging.getLogger(__name__)


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

def load_package_if_present(package_name: str):
    """Load a package if it is present.

    Parameters
    ----------
    package_name : str
        Name of the package to load.

    Returns
    -------
    Any
        The loaded package or None if the package is not found.
    """
    import importlib

    try:
        return importlib.import_module(package_name)
    except ImportError:
        return None

def return_str_path(
    str_path: str | None = None
) -> str:
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
        try:
            str_path = pkg_resources.resource_filename(
                "mkt.schema", 
                "KinaseInfo/*.json"
            )
        except Exception as e:
            str_path = path.join(
                get_repo_root(),
                "missense_kinase_toolkit/mkt/schema/KinaseInfo/*.json"
            )
    else:
        str_path = str_path

    return str_path

def return_file_suffix(
    serialization_function: Callable[[Any], str]
) -> str | None:
    if serialization_function == json.dumps:
        suffix = ".json"
    elif serialization_function == yaml.safe_dump:
        suffix = ".yaml"
    elif serialization_function == toml.dumps:
        suffix = ".toml"
    # else:
    #     package = load_package_if_present("toml")
    #     if package is not None:
    #         suffix = ".toml"
    #     else:
    #         logger.error(f"Serialization function not supported"
    #                     f"; must be json.dumps, yaml.safe_dump, or toml.dumps.")
    #         return
    else:
        logger.error(f"Serialization function not supported"
                    f"; must be json.dumps, yaml.safe_dump, or toml.dumps.")
        return
    return suffix

# adapted from https://axeldonath.com/scipy-2023-pydantic-tutorial/notebooks-rendered/4-serialisation-and-deserialisation.html
#TODO make yaml.safe_dump and toml.dumps compatible
def serialize_kinase_dict(
    kinase_dict: dict[str, BaseModel],
    serialization_function: Callable[[Any], str],
    serialization_kwargs: Optional[dict[str, Any]] = None,
    str_path: str | None = None,
):
    """Serialize KinaseInfo object to files.

    Parameters
    ----------
    kinase_dict : dict[str, BaseModel]
        Dictionary of KinaseInfo objects.
    serialization_function : Callable[[Any], str]
        Serialization function (e.g., json.dumps, yaml.safe_dump, toml.dumps).
    serialization_kwargs : dict[str, Any], optional
        Additional keyword arguments for serialization function, by default None;
            (e.g., {"indent": 2} for json.dumps, {"sort_keys": False} for yaml.safe_dump).
    str_path: str | None = None
        Path to save the serialized file, by default None.
    """
    # otherwise will have /*.json in the path
    str_path = return_str_path(str_path).replace("/*.json", "")

    if serialization_kwargs is None:
        serialization_kwargs = {}

    suffix = return_file_suffix(serialization_function)
    
    for key, val in kinase_dict.items():
        with open(f"{str_path}/{key}{suffix}", "w") as outfile:
           val_serialized =  serialization_function(
                val,
                **serialization_kwargs,
            )
           outfile.write(val_serialized)

def deserialize_kinase_dict(
    deserialization_function: Callable[[Any], str] = json.load,
    deserialization_kwargs: Optional[dict[str, Any]] = None,
    str_path: str | None = None,
) -> dict[str, BaseModel]:
    """Deserialize KinaseInfo object from files.

    Parameters
    ----------
    deserialization_function : Callable[[Any], str]
        Deserialization function (e.g., json.loads, yaml.safe_load, toml.loads).
    deserialization_kwargs : dict[str, Any], optional
        Additional keyword arguments for deserialization function, by default None.
    str_path : str | None, optional
        Path from which to load json files, by default None.

    Returns
    -------
    dict[str, KinaseInfo]
        Dictionary of KinaseInfo objects.
    """
    str_path = return_str_path(str_path)
    list_file = glob.glob(str_path)

    if deserialization_kwargs is None:
        deserialization_kwargs = {}

    dict_import = {}
    for file in list_file:
        with open(file) as openfile:
            val_deserialized = deserialization_function(
                openfile,
                **deserialization_kwargs,
            )
            kinase_obj = kinase_schema.KinaseInfo.parse_raw(val_deserialized)
            dict_import[kinase_obj.hgnc_name] = kinase_obj

    dict_import = {key: dict_import[key] for key in sorted(dict_import.keys())}

    return dict_import
