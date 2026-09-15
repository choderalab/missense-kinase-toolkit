"""Serialize and deserialize :class:`KinaseInfo` objects to/from json/yaml/toml and tar archives.

Provides :func:`serialize_kinase_dict` and :func:`deserialize_kinase_dict` for
round-tripping per-kinase files, plus helpers for reading packaged ``.tar.gz`` data
in memory. TOML serialization is skipped on Windows.
"""

import glob
import json
import logging
import os
import shutil
import tarfile
from datetime import datetime, timezone
from importlib import resources
from io import BytesIO
from typing import Any, Optional

import git
import toml
import yaml
from mkt.schema import kinase_schema
from mkt.schema.config import get_output_dir
from mkt.schema.utils import TQDM_BAR_FORMAT, return_manifest_tallies
from pydantic import BaseModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


_deserialization_cache = {}

STR_MANIFEST_FILENAME = "manifest.json"
"""str: Filename of the build manifest stored alongside the per-kinase files."""


class Manifest(BaseModel):
    """Build record of a :class:`KinaseInfo` archive, checked against the loaded dict."""

    manifest_version: int = 1
    """Manifest schema version, by default 1."""
    generated_at: datetime
    """UTC build timestamp."""
    git: dict[str, str | bool | None] = {}
    """Build checkout ``sha`` and ``dirty`` flag, by default empty."""
    packages: dict[str, str] = {}
    """Package name -> version used for the build, by default empty."""
    n_entries: int
    """Number of kinase entries."""
    counts: dict[str, int]
    """Dotted sub-model path -> number of non-None entries."""
    source_versions: dict[str, dict[str, int]] = {}
    """Dotted sub-model path -> ``Provenance.version`` tally, by default empty."""

    @classmethod
    def from_kinase_dict(
        cls,
        dict_kinase: dict[str, BaseModel],
        list_paths: list[str] | None = None,
        **kwargs: Any,
    ) -> "Manifest":
        """Build a manifest from a kinase dictionary.

        Parameters
        ----------
        dict_kinase : dict[str, KinaseInfo]
            Kinase dictionary being archived.
        list_paths : list[str] | None, optional
            Paths to tally, by default None (see :func:`return_manifest_tallies`).
        **kwargs : Any
            Additional fields (``git``, ``packages``, ``generated_at``); ``generated_at``
            defaults to now (UTC).

        Returns
        -------
        Manifest
            Manifest with counts and source versions computed from ``dict_kinase``.
        """
        kwargs.setdefault("generated_at", datetime.now(timezone.utc))
        counts, source_versions = return_manifest_tallies(dict_kinase, list_paths)
        return cls(
            n_entries=len(dict_kinase),
            counts=counts,
            source_versions=source_versions,
            **kwargs,
        )

    def return_mismatches(self, dict_kinase: dict[str, BaseModel]) -> list[str]:
        """Return expected-vs-actual differences against a loaded kinase dictionary.

        Parameters
        ----------
        dict_kinase : dict[str, KinaseInfo]
            Loaded kinase dictionary.

        Returns
        -------
        list[str]
            One line per mismatched quantity; empty if consistent.
        """
        # tally only this manifest's paths so newer schema fields don't count as drift
        actual = Manifest.from_kinase_dict(
            dict_kinase, list(self.counts), generated_at=self.generated_at
        )
        list_diff = []
        if actual.n_entries != self.n_entries:
            list_diff.append(
                f"n_entries: expected {self.n_entries}, got {actual.n_entries}"
            )
        for field in ("counts", "source_versions"):
            dict_expected, dict_actual = getattr(self, field), getattr(actual, field)
            list_diff.extend(
                f"{field}[{key}]: expected {val}, got {dict_actual.get(key)}"
                for key, val in dict_expected.items()
                if dict_actual.get(key) != val
            )
        return list_diff

    def return_summary(self, int_bar_width: int = 20) -> str:
        """Return a tree-indented text summary of the manifest.

        Parameters
        ----------
        int_bar_width : int, optional
            Width of the coverage bar in characters, by default 20.

        Returns
        -------
        str
            Header (build time, git, packages, entries) plus one row per path with count,
            percent of entries, a coverage bar, and any source-version tally.
        """
        # order each path directly after its parent (e.g. klifs.pocket_seq after klifs)
        dict_order = {path: idx for idx, path in enumerate(self.counts)}
        list_paths = sorted(
            self.counts,
            key=lambda path: tuple(
                dict_order.get(".".join(path.split(".")[: i + 1]), len(dict_order))
                for i in range(path.count(".") + 1)
            ),
        )
        dict_label = {
            path: "  " * path.count(".") + path.rsplit(".", 1)[-1]
            for path in list_paths
        }
        int_label = max(len(label) for label in dict_label.values())

        str_git = self.git.get("sha", "n/a")[:12]
        if self.git.get("dirty"):
            str_git += " (dirty)"
        list_packages = [f"{name} {ver}" for name, ver in self.packages.items()]

        list_lines = [
            f"KinaseInfo manifest v{self.manifest_version}",
            f"  generated  {self.generated_at:%Y-%m-%d %H:%M:%S %Z}".rstrip(),
            f"  git        {str_git}",
            f"  packages   {list_packages[0] if list_packages else 'n/a'}",
            *(f"             {pkg}" for pkg in list_packages[1:]),
            f"  entries    {self.n_entries:,}",
            "",
            f"  {'field':<{int_label}}  {'n':>5}  {'%':>6}",
            "  " + "─" * (int_label + 17 + int_bar_width),
        ]
        for path in list_paths:
            int_n = self.counts[path]
            float_frac = int_n / self.n_entries if self.n_entries else 0.0
            int_fill = round(float_frac * int_bar_width)
            str_bar = "█" * int_fill + "░" * (int_bar_width - int_fill)
            str_versions = " · ".join(
                f"{ver} {n:,}" for ver, n in self.source_versions.get(path, {}).items()
            )
            list_lines.append(
                f"  {dict_label[path]:<{int_label}}  {int_n:>5,}  "
                f"{float_frac:>6.1%}  {str_bar}  {str_versions}".rstrip()
            )
        return "\n".join(list_lines)


def check_kinase_dict_manifest(
    dict_kinase: dict[str, BaseModel],
    manifest: Manifest | None,
    str_path: str,
) -> None:
    """Raise if a loaded kinase dictionary disagrees with its manifest; warn if absent.

    Parameters
    ----------
    dict_kinase : dict[str, KinaseInfo]
        Loaded kinase dictionary.
    manifest : Manifest | None
        Manifest read from the archive, or None if missing.
    str_path : str
        Archive path, for messages.

    Returns
    -------
    None
    """
    if manifest is None:
        logger.warning(
            f"No {STR_MANIFEST_FILENAME} in {str_path}; skipping integrity check."
        )
        return

    list_diff = manifest.return_mismatches(dict_kinase)
    if list_diff:
        raise ValueError(
            f"{str_path} does not match its {STR_MANIFEST_FILENAME} "
            f"(generated_at {manifest.generated_at.isoformat()}):\n"
            + "\n".join(list_diff)
        )


def load_manifest(str_path: str) -> Manifest | None:
    """Read the build manifest from a KinaseInfo ``.tar.gz`` or directory.

    Parameters
    ----------
    str_path : str
        Path to the archive or per-kinase directory.

    Returns
    -------
    Manifest | None
        The parsed manifest, or None if absent.
    """
    if str_path.endswith(".tar.gz"):
        # an empty list_ids skips every kinase entry but still reads the manifest
        str_manifest = _untar_in_memory(str_path, list_ids=[])[2]
    else:
        path_manifest = os.path.join(str_path, STR_MANIFEST_FILENAME)
        if not os.path.exists(path_manifest):
            return None
        with open(path_manifest) as openfile:
            str_manifest = openfile.read()

    if str_manifest is None:
        return None
    return Manifest.model_validate_json(str_manifest)


def print_manifest_summary(str_path: str | None = None) -> None:
    """Print the manifest summary of a KinaseInfo archive or directory.

    Parameters
    ----------
    str_path : str | None, optional
        Path to the archive or directory, by default None (the packaged tar).

    Returns
    -------
    None
    """
    if str_path is None:
        str_path = return_str_path_from_pkg_data()
    manifest = load_manifest(str_path)
    if manifest is None:
        logger.warning(f"No {STR_MANIFEST_FILENAME} in {str_path}.")
        return
    print(manifest.return_summary())


def get_repo_root():
    """Get the root of the git repository.

    Returns
    -------
    str
        Path to the root of the git repository; if not found, return current directory
    """
    try:
        repo = git.Repo(".", search_parent_directories=True)
        return repo.working_tree_dir
    except git.InvalidGitRepositoryError:
        logger.info("Not a git repository; using current directory as root...")
        return "."


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


def raise_if_missing_or_empty(str_path_in: str) -> None:
    """Raise if a path does not exist or is an empty directory.

    Parameters
    ----------
    str_path_in : str
        Path to a file (e.g. ``KinaseInfo.tar.gz``) or directory.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If ``str_path_in`` does not exist or is a directory with no entries.
    """
    if not os.path.exists(str_path_in):
        raise FileNotFoundError(f"{str_path_in} does not exist.")
    if os.path.isdir(str_path_in) and not os.listdir(str_path_in):
        raise FileNotFoundError(f"{str_path_in} is an empty directory.")


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
    bool_extract : bool, optional
        If True, extract the files to memory, by default True.
    list_ids : list[str] | None, optional
        List of IDs to filter the files if reading from memory, by default None.
        If None, all files will be extracted.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        Entry IDs and a dictionary of file names to contents; the build manifest
        (:data:`STR_MANIFEST_FILENAME`) is excluded from both.

    """
    return _untar_in_memory(str_path, bool_extract=bool_extract, list_ids=list_ids)[:2]


def _untar_in_memory(
    str_path: str,
    bool_extract: bool = True,
    list_ids: list[str] | None = None,
) -> tuple[list[str], dict[str, str], str | None]:
    """Untar files in memory, returning the build manifest separately.

    Parameters
    ----------
    str_path : str
        Path to the tar.gz file.
    bool_extract : bool, optional
        If True, extract the files to memory, by default True.
    list_ids : list[str] | None, optional
        List of IDs to filter the files, by default None (all files).

    Returns
    -------
    tuple[list[str], dict[str, str], str | None]
        Entry IDs, file names to contents, and the manifest contents (None if absent
        or not extracted).
    """
    with open(str_path, "rb") as f:
        tar_data = f.read()

    list_entries, dict_bytes, str_manifest = [], {}, None
    with BytesIO(tar_data) as tar_buffer, tarfile.open(
        fileobj=tar_buffer, mode="r"
    ) as tar:
        for member in tar.getmembers():
            filename = os.path.basename(member.name)
            # make sure entry is file; ignore MacOS AppleDouble files
            if not member.isfile() or "._" in filename:
                continue
            if filename == STR_MANIFEST_FILENAME:
                if bool_extract:
                    with tar.extractfile(member) as f:
                        str_manifest = f.read().decode("utf-8")
                continue
            # use list_ids, if provided
            if list_ids is None or filename.split(".")[0] in list_ids:
                list_entries.append(filename.split(".")[0])
                if bool_extract:
                    with tar.extractfile(member) as f:
                        dict_bytes[member.name] = f.read()

    if bool_extract:
        # decode bytes to string
        dict_bytes = {k: v.decode("utf-8") for k, v in dict_bytes.items()}

    return list_entries, dict_bytes, str_manifest


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

    Raises
    ------
    FileNotFoundError
        If ``str_path`` is None and the packaged resource is missing or empty.
    """
    if pkg_name is None:
        pkg_name = "mkt.schema"
    if pkg_resource is None:
        pkg_resource = "KinaseInfo.tar.gz"

    if str_path is None:
        str_path = os.path.join(resources.files(pkg_name), pkg_resource)
        raise_if_missing_or_empty(str_path)
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


def _raise_if_stem_mismatch(str_filename: str, kinase_obj: BaseModel) -> None:
    """Raise if a serialized file's stem differs from the ``hgnc_name`` it holds.

    ``list_ids`` loads match filename stems, so the stem must equal ``hgnc_name``.

    Parameters
    ----------
    str_filename : str
        Path or tar member name of the serialized file.
    kinase_obj : KinaseInfo
        The object deserialized from that file.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the filename stem is not ``kinase_obj.hgnc_name``.
    """
    str_stem = os.path.basename(str_filename).split(".")[0]
    if str_stem != kinase_obj.hgnc_name:
        raise ValueError(
            f"file {str_filename!r} holds hgnc_name {kinase_obj.hgnc_name!r}; "
            "filenames must match hgnc_name."
        )


# adapted from https://axeldonath.com/scipy-2023-pydantic-tutorial/notebooks-rendered/4-serialisation-and-deserialisation.html
def serialize_kinase_dict(
    kinase_dict: dict[str, BaseModel],
    suffix: str = "json",
    serialization_kwargs: Optional[dict[str, Any]] = None,
    str_path: str | None = None,
) -> None:
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

    Raises
    ------
    ValueError
        If any key differs from its object's ``hgnc_name`` (files are named by the key).
    """
    list_mismatch = [
        (key, val.hgnc_name)
        for key, val in kinase_dict.items()
        if getattr(val, "hgnc_name", key) != key
    ]
    if list_mismatch:
        raise ValueError(
            "dict keys must equal hgnc_name (files are named by key); "
            f"(key, hgnc_name) mismatches: {list_mismatch}"
        )

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

    for key, val in tqdm(
        kinase_dict.items(),
        desc="Serializing KinaseInfo objects...",
        bar_format=TQDM_BAR_FORMAT,
    ):
        with open(f"{str_path}/{key}.{suffix}", "w") as outfile:
            # TOML tables require string keys; mode="json" stringifies non-string dict keys
            # (e.g. int-keyed maps), which pydantic coerces back on deserialize
            model_dump = (
                val.model_dump(mode="json") if suffix == "toml" else val.model_dump()
            )
            val_serialized = DICT_FUNCS[suffix]["serialize"](
                model_dump,
                **serialization_kwargs,
            )
            outfile.write(val_serialized)


def deserialize_kinase_dict(
    suffix: str = "json",
    deserialization_kwargs: Optional[dict[str, Any]] = None,
    str_path: str | None = None,
    bool_remove: bool = True,
    list_ids: list[str] | None = None,
    str_name: str | None = None,
    bool_verbose: bool = True,
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
    str_name : str | None, optional
        Name of a variable in the global scope that contains the kinase dictionary to prevent reloading, by default None.

    Returns
    -------
    dict[str, KinaseInfo]
        Dictionary of KinaseInfo objects.
    """
    if str_name is not None and str_name in _deserialization_cache:
        if bool_verbose:
            logger.info(f"Loading KinaseInfo object from variable {str_name}...")
        return _deserialization_cache[str_name]

    # callers that only need the modules importable (e.g. a Sphinx docs build or
    # a CLI printing usage text) can set MKT_SKIP_KINASE_DICT to skip the
    # expensive deserialization; cache the empty result so repeat calls are cheap
    if os.environ.get("MKT_SKIP_KINASE_DICT"):
        if bool_verbose:
            logger.info(
                "MKT_SKIP_KINASE_DICT set; skipping KinaseInfo deserialization."
            )
        if str_name is not None:
            _deserialization_cache[str_name] = {}
        return {}

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

    dict_import, manifest = {}, None
    if str_path.endswith(".tar.gz"):
        _, dict_str, str_manifest = _untar_in_memory(str_path, list_ids=list_ids)
        if str_manifest is not None:
            manifest = Manifest.model_validate_json(str_manifest)
        for str_member, val in tqdm(
            dict_str.items(),
            desc="Deserializing KinaseInfo objects in memory...",
            bar_format=TQDM_BAR_FORMAT,
        ):

            val_deserialized = DICT_FUNCS[suffix]["deserialize_str"](
                val,
                **deserialization_kwargs,
            )

            kinase_obj = kinase_schema.KinaseInfo.model_validate(val_deserialized)
            _raise_if_stem_mismatch(str_member, kinase_obj)
            dict_import[kinase_obj.hgnc_name] = kinase_obj
    else:
        list_file = [
            file
            for file in glob.glob(os.path.join(str_path, f"*.{suffix}"))
            if os.path.basename(file) != STR_MANIFEST_FILENAME
        ]
        manifest = load_manifest(str_path)
        for file in tqdm(
            list_file,
            desc="Deserializing KinaseInfo objects from files...",
            bar_format=TQDM_BAR_FORMAT,
        ):
            with open(file) as openfile:

                val_deserialized = DICT_FUNCS[suffix]["deserialize_file"](
                    openfile,
                    **deserialization_kwargs,
                )

                kinase_obj = kinase_schema.KinaseInfo.model_validate(val_deserialized)
                _raise_if_stem_mismatch(file, kinase_obj)
                dict_import[kinase_obj.hgnc_name] = kinase_obj

        if bool_remove:
            clean_files_and_delete_directory(list_file)

    dict_import = {key: dict_import[key] for key in sorted(dict_import.keys())}

    # subset loads can't match the manifest; directories are checked only if one exists
    if list_ids is None and (manifest is not None or str_path.endswith(".tar.gz")):
        check_kinase_dict_manifest(dict_import, manifest, str_path)

    if str_name is not None:
        _deserialization_cache[str_name] = dict_import

    return dict_import


STR_CONSERVATION_FILENAME = "KLIFSConservationData"
"""str: Basename (no suffix) of the persisted KLIFS conservation-data artifact."""

_conservation_cache = {}
"""dict: Module-level cache of loaded :class:`KLIFSConservationData` objects keyed by
``str_name`` (mirrors :data:`_deserialization_cache`)."""


def serialize_conservation_data(
    conservation_data: BaseModel,
    suffix: str = "json",
    str_path: str | None = None,
) -> str | None:
    """Serialize a :class:`KLIFSConservationData` artifact to a single file.

    Unlike :func:`serialize_kinase_dict` (one file per kinase), this writes the whole
    single-object artifact to ``{str_path}/KLIFSConservationData.{suffix}``, reusing the
    :data:`DICT_FUNCS` serialization registry.

    Parameters
    ----------
    conservation_data : BaseModel
        The :class:`mkt.schema.conservation_schema.KLIFSConservationData` object.
    suffix : str
        Serialization type supported: json, yaml, toml.
    str_path : str | None
        Directory to write into, by default None (the ``mkt.schema`` package directory).

    Returns
    -------
    str | None
        Path written, or None if the suffix is unsupported.
    """
    if suffix not in DICT_FUNCS:
        logger.error(
            f"Serialization type ({suffix}) not supported; must be json, yaml, or toml."
        )
        return None

    if os.name == "nt" and suffix == "toml":
        logger.info("TOML serialization is not supported on Windows.")
        return None

    if str_path is None:
        str_path = str(resources.files("mkt.schema"))
    os.makedirs(str_path, exist_ok=True)

    filepath = os.path.join(str_path, f"{STR_CONSERVATION_FILENAME}.{suffix}")
    with open(filepath, "w") as outfile:
        outfile.write(
            DICT_FUNCS[suffix]["serialize"](
                conservation_data.model_dump(),
                **DICT_FUNCS[suffix]["kwargs_serialize"],
            )
        )
    logger.info(f"Serialized KLIFSConservationData to {filepath}")
    return filepath


def load_conservation_data(
    suffix: str = "json",
    str_path: str | None = None,
    str_name: str | None = None,
):
    """Load a :class:`KLIFSConservationData` artifact from a single file.

    Parameters
    ----------
    suffix : str
        Deserialization type supported: json, yaml, toml.
    str_path : str | None
        Path to the artifact file, by default None (the packaged
        ``mkt/schema/KLIFSConservationData.json``).
    str_name : str | None
        If provided, cache the loaded object under this name in
        :data:`_conservation_cache` to prevent reloading.

    Returns
    -------
    KLIFSConservationData
        The deserialized conservation-data object.
    """
    if str_name is not None and str_name in _conservation_cache:
        logger.info(f"Loading KLIFSConservationData from variable {str_name}...")
        return _conservation_cache[str_name]

    if suffix not in DICT_FUNCS:
        logger.error(
            f"Serialization type ({suffix}) not supported; must be json, yaml, or toml."
        )
        return None

    from mkt.schema.conservation_schema import KLIFSConservationData

    if str_path is None:
        str_path = os.path.join(
            str(resources.files("mkt.schema")), f"{STR_CONSERVATION_FILENAME}.{suffix}"
        )

    with open(str_path) as openfile:
        val_deserialized = DICT_FUNCS[suffix]["deserialize_file"](
            openfile,
            **DICT_FUNCS[suffix]["kwargs_deserialize"],
        )
    obj = KLIFSConservationData.model_validate(val_deserialized)

    if str_name is not None:
        _conservation_cache[str_name] = obj
    return obj


def save_plot(
    fig,
    output_filename: str,
    plot_type: str = "Plot",
    bool_force_local: bool = True,
    bool_image_subdir=True,
    output_path: str | None = None,
    bool_svg: bool = True,
    bool_png: bool = True,
    bool_pdf: bool = False,
    **kwargs,
) -> None:
    """Save the current matplotlib figure in the requested vector/raster formats.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    output_filename : str
        Name of the output file to save the plot. Any extension is stripped; the
        format suffixes are appended per the ``bool_svg``/``bool_png``/``bool_pdf`` flags.
    plot_type : str
        Description of the plot type for logging purposes (e.g., "Dynamic range plot")
    bool_force_local : bool
        If True, forces saving to the local dir (repo root or cwd) regardless of env var; default is True.
    bool_image_subdir : bool
        If True, saves images to a subdirectory named "images" within the output path; default is True.
    output_path : str | None
        Optional path to save the plot. If None, saves to the current working directory.
    bool_svg : bool
        If True, write an SVG copy; default is True.
    bool_png : bool
        If True, write a PNG copy; default is True.
    bool_pdf : bool
        If True, write a PDF copy; default is False.
    **kwargs
        Additional keyword arguments to pass to plt.savefig (e.g., {"dpi": 300}). Default is empty dict.
    """
    import matplotlib.pyplot as plt

    # remove extension if provided
    output_filename = os.path.splitext(output_filename)[0]

    # get_repo_root() > output_path > get_output_dir()
    if bool_force_local:
        if output_path is not None:
            logger.info(
                "bool_force_local is True, so ignoring provided output_path "
                f"{output_path} and saving to local directory instead."
            )
        output_path = get_repo_root()
    elif output_path is None:
        output_path = get_output_dir()

    # set default savefig parameters
    savefig_params = {"bbox_inches": "tight"}
    # update with any user-provided kwargs
    savefig_params.update(kwargs)

    suffixes = [
        suffix
        for suffix, flag in (("svg", bool_svg), ("png", bool_png), ("pdf", bool_pdf))
        if flag
    ]
    for suffix in suffixes:
        if bool_image_subdir:
            file_path = os.path.join(
                output_path, "images", f"{output_filename}.{suffix}"
            )
        else:
            file_path = os.path.join(output_path, f"{output_filename}.{suffix}")

        fig.savefig(file_path, format=suffix, **savefig_params)
        logger.info(f"{plot_type} saved to {file_path}")

    plt.close(fig)
