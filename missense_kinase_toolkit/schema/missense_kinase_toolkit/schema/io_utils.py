import git
from typing import Any, Callable, Optional
import json
import glob

from missense_kinase_toolkit.schema import kinase_schema


def get_repo_root():
    try:
        repo = git.Repo(".", search_parent_directories=True)
        return repo.working_tree_dir
    except git.InvalidGitRepositoryError:
        return None


def serialize_kinase_dict(
    kinase_dict: dict[str, kinase_schema.KinaseInfo],
    serialization_function: Callable[[Any], str],
    serialization_kwargs: Optional[dict[str, Any]] = None,
    str_path: str | None = None,
) -> None:
    """Serialize KinaseInfo object to files.

    Parameters
    ----------
    kinase_dict : dict[str, KinaseInfo]
        Dictionary of KinaseInfo objects.
    serialization_function : Callable[[Any], str]
        Serialization function (e.g., json.dumps, yaml.safe_dump, toml.dumps).
    serialization_kwargs : dict[str, Any], optional
        Additional keyword arguments for serialization function, by default None;
            (e.g., {"indent": 2} for json.dumps, {"sort_keys": False} for yaml.safe_dump).
    str_path : str | None, optional
        Path to save json files, by default None.
    """
    if serialization_kwargs is None:
        serialization_kwargs = {}

    if str_path is None:
        str_path = get_repo_root() + "/data/KinaseInfo"
    else:
        str_path = str_path

    for key, val in kinase_dict.items():
        with open(f"{str_path}/{key}.json", "w") as outfile:   
            val_serialized = serialization_function(
                val, 
                **serialization_kwargs,
            )
            #TODO: Fix this
            json.dump(val_serialized, outfile)


def deserialize_kinase_dict(
    deserialization_function: Callable[[Any], str],
    deserialization_kwargs: Optional[dict[str, Any]] = None,
    str_path: str | None = None,
) -> dict[str, kinase_schema.KinaseInfo]:
    """Deserialize KinaseInfo object from files.

    Parameters
    ----------
    deserialization_function : Callable[[Any], str]
        Deserialization function (e.g., json.loads, yaml.safe_load, toml.loads).
    deserialization_kwargs : dict[str, Any], optional
        Additional keyword arguments for deserialization function, by default None.
    str_path : str | None, optional
        Path to save json files, by default None.

    Returns
    -------
    dict[str, KinaseInfo]
        Dictionary of KinaseInfo objects.
    """
    if deserialization_kwargs is None:
        deserialization_kwargs = {}

    if str_path is None:
        str_path = get_repo_root() + "/data/KinaseInfo/*"
    else:
        str_path = str_path

    list_file = glob.glob(str_path)

    dict_import = {}
    for file in list_file:
        with open(file, "r") as openfile:
            val_deserialized = deserialization_function(
                openfile,
                **deserialization_kwargs,
            )
            kinase_obj = kinase_schema.KinaseInfo.parse_raw(val_deserialized)
            dict_import[kinase_obj.hgnc_name] = kinase_obj

    dict_import = {key: dict_import[key] for key in sorted(dict_import.keys())}

    return dict_import