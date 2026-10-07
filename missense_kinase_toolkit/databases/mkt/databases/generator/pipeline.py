"""Orchestration for the compositional KinaseInfo build pipeline.

The :class:`Pipeline` class runs ``generate_kinaseinfo_objects`` in one of three modes: a
full kinome regeneration; a partial per-source rebuild (``--only <source>``) that re-fetches
one base-build source and re-runs the dependent validators on the existing dict; and a
per-entry one-off update (``--kinase``) that rebuilds targeted entries and splices them back
into the existing ``KinaseInfo.tar.gz``. Each mode produces or updates the assembled dict and
then shares one finalize step (enrich -> serialize -> tar -> reports -> cleanup). The base
build (API fetch + harmonization) is a mandatory prerequisite; enrichment steps only mutate
additive optional fields on the assembled objects.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from inspect import isclass
from typing import Any, get_args

import git
from mkt.databases.generator import steps as build_steps
from mkt.databases.io_utils import create_tar_without_metadata
from mkt.databases.kinase_schema import (
    Source,
    apply_hgnc_fallback,
    combine_kinaseinfo,
    combine_kinaseinfo_kd,
    combine_kinaseinfo_uniprot,
    fetch_source,
    generate_dict_obj_from_api_or_scraper,
)
from mkt.schema.io_utils import (
    STR_MANIFEST_FILENAME,
    Manifest,
    deserialize_kinase_dict,
    get_repo_root,
    load_manifest,
    return_dir_entry_sha256,
    return_str_path_from_pkg_data,
    serialize_kinase_dict,
)
from mkt.schema.kinase_schema import Provenance, register_sources
from mkt.schema.utils import return_resolved_sources, rgetattr, split_domain_suffix
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def return_manifest_sources(dict_kinase: dict[str, Any]) -> dict[str, Provenance]:
    """Return the manifest ``sources`` table for every SHA-256-only source in the dict.

    Parameters
    ----------
    dict_kinase : dict[str, KinaseInfo]
        The dict being archived.

    Returns
    -------
    dict[str, Provenance]
        Source-file SHA-256 -> full Provenance, from the registered sources.

    Raises
    ------
    ValueError
        If a record's SHA-256-only source was never registered (the archive could not
        resolve it).
    """
    dict_sources, set_missing = return_resolved_sources(dict_kinase)
    if set_missing:
        raise ValueError(
            "record sources with no registered provenance: "
            + ", ".join(f"{sha[:12]}..." for sha in sorted(set_missing))
        )
    return dict(sorted(dict_sources.items()))


DEFAULT_PATH_OBJECTS = "missense_kinase_toolkit/schema/mkt/schema/KinaseInfo"
"""str: Default objects directory (relative to repo root), matching package-data layout."""

DEFAULT_PATH_REPORTS = "images"
"""str: Default reports directory (relative to repo root)."""

REPORTS_GROUP_SUBDIR = "dict_kinase"
"""str: Reports sub-directory grouping the whole-kinome ``KinaseInfo`` figures under
``{path_reports}/dict_kinase/<generated_at>/``, one folder per archive version."""

DATETIME_SUBDIR_FMT = "%Y.%m.%d.%H%M%S"
"""str: ``strftime`` format for the datetime-stamped reports subdirectory, applied to the
archive manifest's ``generated_at`` (UTC)."""

LIST_MANIFEST_PACKAGES = ["mkt-schema", "mkt-databases"]
"""list[str]: Packages whose versions are recorded in the archive manifest."""


@dataclass
class BuildContext:
    """Mutable state threaded through the build pipeline and its steps."""

    dict_kinaseinfo: dict[str, Any]
    """Assembled KinaseInfo objects keyed by ``hgnc_name`` (incl. ``_1``/``_2`` multi-kinase-domain suffixes)."""
    path_objects: str
    """Absolute path to the objects directory; the archive is written beside it, and
    entries are staged in a temporary directory, never here."""
    path_reports: str
    """Absolute path to the reports/figures directory."""
    path_tar: str
    """Absolute path to the ``KinaseInfo.tar.gz`` archive."""
    subset_hgnc: set[str] | None = None
    """If not None, the ``hgnc_name`` keys targeted by a subset (``--kinase``) build; enrichment steps iterate only these (reports still characterize the whole spliced dict), by default None."""
    force: bool = False
    """If True (``--recompute``), structure steps re-fetch/re-slice and recompute their derived properties (SASA, superposition) even when already present, by default False."""
    report_config: Any = None
    """Loaded :class:`KinaseInfoFiguresConfig` for the report steps (aesthetics from the ``kinaseinfo`` config namespace, or defaults), by default None."""


def run_base_build(
    subset_uniprot: set[str] | None = None,
) -> dict[str, Any]:
    """Fetch, harmonize, and assemble KinaseInfo objects (the base build).

    Parameters
    ----------
    subset_uniprot : set[str] | None, optional
        If provided, restrict the build to these UniProt IDs (one-off update),
        by default None (full kinome).

    Returns
    -------
    dict[str, Any]
        KinaseInfoGenerator objects keyed by ``hgnc_name``.
    """
    dict_obj = generate_dict_obj_from_api_or_scraper(subset_uniprot=subset_uniprot)
    dict_uniprot = combine_kinaseinfo_uniprot(dict_obj)
    dict_kd = combine_kinaseinfo_kd(dict_obj)
    return combine_kinaseinfo(dict_uniprot, dict_kd)


def _reconstruct_dict_obj(dict_kinase: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the raw per-source dict_obj from an assembled KinaseInfo dict.

    Groups multi-kinase-domain (``_1``/``_2``) entries back to their base UniProt so the
    combine_* functions see the same structure as a fresh base build; hgnc/uniprot/pfam are
    single-valued, kinhub/klifs/kincore are per-domain lists.

    Parameters
    ----------
    dict_kinase : dict[str, Any]
        Assembled KinaseInfo objects keyed by ``hgnc_name``.

    Returns
    -------
    dict[str, Any]
        Raw per-source dict keyed by the :class:`Source` values.
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for obj in dict_kinase.values():
        grouped[obj.uniprot_id.split("_")[0]].append(obj)

    dict_obj = {source.value: {} for source in Source}
    for base, list_obj in grouped.items():
        first = list_obj[0]
        dict_obj[Source.hgnc][base] = first.hgnc_name.split("_")[0]
        dict_obj[Source.uniprot][base] = first.uniprot
        # pfam may be dropped from one domain (drop_nonintersecting_pfam); keep any survivor
        pfam = next((o.pfam for o in list_obj if o.pfam is not None), None)
        if pfam is not None:
            dict_obj[Source.pfam][base] = pfam
        dict_obj[Source.kinhub][base] = [o.kinhub for o in list_obj]
        dict_obj[Source.klifs][base] = [o.klifs for o in list_obj]
        dict_obj[Source.kincore][base] = [o.kincore for o in list_obj]
    return dict_obj


def _return_model_cls(model_cls: type, str_field: str) -> type | None:
    """Return the Pydantic model class a (possibly optional) field holds, or None."""
    field = model_cls.model_fields.get(str_field)
    if field is None:
        return None
    for sub_cls in get_args(field.annotation) or (field.annotation,):
        if isclass(sub_cls) and issubclass(sub_cls, BaseModel):
            return sub_cls
    return None


def _carry_field(obj_new: Any, obj_old: Any, str_path: str) -> None:
    """Copy one dotted field from an existing entry onto its rebuilt counterpart.

    Missing parents on the rebuilt entry are created when their model has no required
    fields (e.g. a KinCoRe shell holding only ``msa``); otherwise the value is dropped
    with a warning, since it has nothing to attach to.

    Parameters
    ----------
    obj_new : KinaseInfo
        Rebuilt entry (mutated in place).
    obj_old : KinaseInfo
        Existing entry the value is copied from.
    str_path : str
        Dotted field path, e.g. ``"kincore.cif.sasa"``.
    """
    val = rgetattr(obj_old, str_path)
    if val is None:
        return
    *list_parents, str_leaf = str_path.split(".")
    parent = obj_new
    for str_name in list_parents:
        child = getattr(parent, str_name, None)
        if child is None:
            child_cls = _return_model_cls(type(parent), str_name)
            try:
                child = child_cls() if child_cls is not None else None
            except ValidationError:
                child = None
            if child is None:
                logger.warning(
                    f"{obj_new.hgnc_name}: no '{str_name}' to carry '{str_path}' onto; "
                    "dropping it."
                )
                return
            setattr(parent, str_name, child)
        parent = child
    setattr(parent, str_leaf, val)


def _clear_field(obj: Any, str_path: str) -> None:
    """Set one dotted field to None if its parent exists."""
    str_parent, _, str_leaf = str_path.rpartition(".")
    parent = rgetattr(obj, str_parent) if str_parent else obj
    if parent is not None:
        setattr(parent, str_leaf, None)


def merge_rebuilt_entries(
    dict_existing: dict[str, Any],
    dict_new: dict[str, Any],
    names: list[str],
    set_base_uniprot: set[str],
) -> set[str]:
    """Merge rebuilt entries into the existing dict, field by field.

    For each rebuilt entry, fields owned by steps that will re-run (``names``) are cleared so
    the step computes them fresh, unless the step checks its own inputs
    (``Component.checks_inputs``) and keeps or recomputes the carried value itself. Fields
    owned by every other step are carried over from the existing entry. Existing entries of
    a rebuilt UniProt that the rebuild no longer produces (e.g. a renamed or dropped domain)
    are removed.

    Parameters
    ----------
    dict_existing : dict[str, KinaseInfo]
        Existing dict keyed by ``hgnc_name`` (mutated in place).
    dict_new : dict[str, KinaseInfo]
        Rebuilt entries keyed by ``hgnc_name``.
    names : list[str]
        Enrichment steps that will run on the rebuilt entries.
    set_base_uniprot : set[str]
        Base UniProt IDs that were rebuilt.

    Returns
    -------
    set[str]
        ``hgnc_name`` keys of the rebuilt entries (the enrichment target set).
    """
    set_clear = {
        name for name in names if not build_steps.COMPONENTS[name].checks_inputs
    }
    for hgnc_name, obj_new in dict_new.items():
        obj_old = dict_existing.get(hgnc_name)
        for step, list_paths in build_steps.return_step_writes().items():
            for str_path in list_paths:
                if step in set_clear:
                    _clear_field(obj_new, str_path)
                elif obj_old is not None:
                    _carry_field(obj_new, obj_old, str_path)

    list_stale = [
        hgnc_name
        for hgnc_name, obj in dict_existing.items()
        if _strip_kd_suffix(obj.uniprot_id) in set_base_uniprot
        and hgnc_name not in dict_new
    ]
    if list_stale:
        logger.warning(
            f"removing entr(ies) no longer produced by the rebuild: {sorted(list_stale)}"
        )
    for hgnc_name in list_stale:
        del dict_existing[hgnc_name]

    dict_existing.update(dict_new)
    return set(dict_new)


def run_source_rebuild(
    sources: list[str],
    dict_existing: dict[str, Any],
    subset_uniprot: set[str] | None = None,
) -> dict[str, Any]:
    """Rebuild the dict refreshing only the named base-build source(s).

    Reconstructs the raw dict_obj from ``dict_existing``, replaces each named source with a
    fresh :func:`fetch_source`, and re-runs the combine_* pipeline so every cross-source
    validator (KinCoRe alignments, KLIFS2UniProt mapping) recomputes. The result holds only
    base-build fields; merge it with :func:`merge_rebuilt_entries`.

    Parameters
    ----------
    sources : list[str]
        Source names (subset of :class:`Source`) to refresh.
    dict_existing : dict[str, Any]
        The currently serialized KinaseInfo dict.
    subset_uniprot : set[str] | None, optional
        Base UniProt IDs to rebuild (``--kinase``), by default None (every entry).

    Returns
    -------
    dict[str, Any]
        The rebuilt KinaseInfo dict.
    """
    if subset_uniprot is not None:
        dict_existing = {
            k: v
            for k, v in dict_existing.items()
            if _strip_kd_suffix(v.uniprot_id) in subset_uniprot
        }
    dict_obj = _reconstruct_dict_obj(dict_existing)
    set_uniprot = set(dict_obj[Source.uniprot].keys())
    for source in sources:
        logger.info(
            f"refreshing source '{source}' for {len(set_uniprot)} UniProt IDs..."
        )
        dict_obj[source] = fetch_source(source, set_uniprot)
    if Source.hgnc in sources:
        apply_hgnc_fallback(dict_obj)
    dict_uniprot = combine_kinaseinfo_uniprot(dict_obj)
    dict_kd = combine_kinaseinfo_kd(dict_obj)
    return combine_kinaseinfo(dict_uniprot, dict_kd)


def _strip_kd_suffix(str_id: str) -> str:
    """Strip a trailing multi-kinase-domain suffix (e.g. ``_1``) from an id/name.

    Parameters
    ----------
    str_id : str
        HGNC name or UniProt ID, possibly suffixed with ``_<digits>``.

    Returns
    -------
    str
        The base id/name with any trailing ``_<digits>`` removed.
    """
    return split_domain_suffix(str_id)[0]


def _resolve_targets(
    list_kinase: list[str],
    dict_existing: dict[str, Any],
) -> tuple[set[str], set[str]]:
    """Map requested HGNC names to base UniProt IDs using the existing dict.

    Parameters
    ----------
    list_kinase : list[str]
        Requested HGNC names (``--kinase``); a base name matches all its ``_1``/``_2``
        variants.
    dict_existing : dict[str, Any]
        The currently serialized KinaseInfo dict keyed by ``hgnc_name``.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(subset_uniprot, set_unresolved)`` — resolved base UniProt IDs and the
        requested names absent from the existing dict.
    """
    map_hgnc2uniprot: dict[str, set[str]] = {}
    for hgnc_name, obj in dict_existing.items():
        base_hgnc = _strip_kd_suffix(hgnc_name)
        base_uniprot = _strip_kd_suffix(obj.uniprot_id)
        map_hgnc2uniprot.setdefault(base_hgnc, set()).add(base_uniprot)

    subset_uniprot, set_unresolved = set(), set()
    for kinase in list_kinase:
        base = _strip_kd_suffix(kinase)
        if base in map_hgnc2uniprot:
            subset_uniprot |= map_hgnc2uniprot[base]
        else:
            set_unresolved.add(kinase)
    return subset_uniprot, set_unresolved


def _resolve_new_kinases(set_hgnc: set[str]) -> tuple[set[str], set[str]]:
    """Resolve HGNC names absent from the existing dict to UniProt IDs via KinHub.

    Parameters
    ----------
    set_hgnc : set[str]
        HGNC names not found in the existing dict (candidate new kinases).

    Returns
    -------
    tuple[set[str], set[str]]
        ``(subset_uniprot, set_missing)`` — resolved UniProt IDs and names that could
        not be resolved from the KinHub kinome.
    """
    from mkt.databases import scrapers
    from mkt.databases.kinase_schema import convert_df2dictobj

    dict_kinhub = convert_df2dictobj(scrapers.kinhub(), "kinhub")

    map_hgnc2uniprot: dict[str, set[str]] = {}
    for uniprot_id, entry in dict_kinhub.items():
        list_entry = entry if isinstance(entry, list) else [entry]
        for obj in list_entry:
            for attr in ("hgnc_name", "xname"):
                val = getattr(obj, attr, None)
                if val is not None:
                    map_hgnc2uniprot.setdefault(val, set()).add(uniprot_id)

    subset_uniprot, set_missing = set(), set()
    for hgnc_name in set_hgnc:
        if hgnc_name in map_hgnc2uniprot:
            subset_uniprot |= map_hgnc2uniprot[hgnc_name]
        else:
            set_missing.add(hgnc_name)
    return subset_uniprot, set_missing


def _resolve_dir(path_repo: str, path_rel: str | None, default_rel: str) -> str:
    """Resolve and create an output directory relative to the repo root.

    Parameters
    ----------
    path_repo : str
        Repo root (or cwd if not a git repo).
    path_rel : str | None
        User-provided path relative to the repo root, or None for the default.
    default_rel : str
        Default path relative to the repo root.

    Returns
    -------
    str
        Absolute path to the (created) output directory.
    """
    path_out = os.path.join(
        path_repo, path_rel if path_rel is not None else default_rel
    )
    os.makedirs(path_out, exist_ok=True)
    if not os.path.isdir(path_out):
        raise NotADirectoryError(f"output path is not a directory: {path_out}")
    return path_out


def _return_git_info() -> dict[str, str | bool]:
    """Return the build checkout's commit SHA and dirty flag.

    Returns
    -------
    dict[str, str | bool]
        ``{"sha": ..., "dirty": ...}``, or empty outside a git checkout.
    """
    try:
        repo = git.Repo(get_repo_root(), search_parent_directories=True)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        logger.warning("not a git checkout; manifest records package versions only.")
        return {}
    bool_dirty = repo.is_dirty()
    if bool_dirty:
        logger.warning("building from a dirty tree; manifest git sha is ambiguous.")
    return {"sha": repo.head.commit.hexsha, "dirty": bool_dirty}


def _return_package_versions() -> dict[str, str]:
    """Return installed versions of :data:`LIST_MANIFEST_PACKAGES`.

    Returns
    -------
    dict[str, str]
        Package name -> version string.
    """
    return {name: version(name) for name in LIST_MANIFEST_PACKAGES}


@dataclass
class Pipeline:
    """Orchestrates the KinaseInfo build across its run modes.

    Holds the resolved output paths and exposes one method per mode (:meth:`full`,
    :meth:`update`, :meth:`source_rebuild`); each produces or updates the dict and hands off
    to the shared :meth:`_finalize` (enrich -> serialize -> tar -> reports -> cleanup).
    :meth:`run` dispatches to the right mode from the CLI arguments.
    """

    path_objects: str
    """Absolute path to the objects directory; the archive is written beside it, and
    entries are staged in a temporary directory, never here."""
    path_reports: str
    """Absolute path to the reports/figures directory."""
    path_tar: str
    """Absolute path to the ``KinaseInfo.tar.gz`` archive."""
    config_path: str | None = None
    """Shared study YAML supplying report aesthetics (``kinaseinfo`` namespace); when set, reports go to ``<output.subdir>/<config-stem>/kinaseinfo/`` instead of the datetime-stamped dir, by default None."""

    @classmethod
    def from_paths(
        cls,
        path_objects: str | None = None,
        path_reports: str | None = None,
        config_path: str | None = None,
    ) -> "Pipeline":
        """Build a Pipeline, resolving the objects/reports dirs and the tar path.

        Parameters
        ----------
        path_objects : str | None, optional
            Objects directory relative to the repo root, by default the package-data layout.
        path_reports : str | None, optional
            Reports directory relative to the repo root, by default ``images``.
        config_path : str | None, optional
            Shared study YAML for report aesthetics + output naming, by default None.

        Returns
        -------
        Pipeline
            A pipeline with resolved, created output directories.
        """
        from mkt.databases.config import set_request_cache

        path_repo = get_repo_root()
        # persist HTTP responses (incl. AlphaFold) so every mode -- not just the base
        # build -- reuses the SQLite cache rather than the per-run in-memory backend
        set_request_cache(os.path.join(path_repo, "requests_cache.sqlite"))
        # not created: entries are staged in a temp dir; only the tar's parent must exist
        path_objects = os.path.join(
            path_repo,
            path_objects if path_objects is not None else DEFAULT_PATH_OBJECTS,
        )
        path_reports = _resolve_dir(path_repo, path_reports, DEFAULT_PATH_REPORTS)
        path_tar = os.path.normpath(
            os.path.join(path_objects, "..", "KinaseInfo.tar.gz")
        )
        os.makedirs(os.path.dirname(path_tar), exist_ok=True)
        return cls(path_objects, path_reports, path_tar, config_path=config_path)

    def _load_existing(self) -> dict[str, Any]:
        """Deserialize the existing dict from the target archive or the packaged tar.

        Returns
        -------
        dict[str, Any]
            The existing KinaseInfo dict (empty if none is found).
        """
        if os.path.exists(self.path_tar):
            return deserialize_kinase_dict(str_path=self.path_tar)
        logger.info(
            f"no existing archive at {self.path_tar}; reading packaged KinaseInfo.tar.gz."
        )
        return deserialize_kinase_dict()

    def _serialize_and_tar(self, dict_kinaseinfo: dict[str, Any]) -> None:
        """Serialize the dict and its manifest to files and (re)build the tar archive.

        Parameters
        ----------
        dict_kinaseinfo : dict[str, Any]
            The KinaseInfo dict to serialize.

        Returns
        -------
        None
        """
        # stage in a fresh system temp dir: removed on success, error, or Ctrl-C, never
        # left in the repo/package, and never mixed with files from an earlier run
        with tempfile.TemporaryDirectory(prefix="KinaseInfo_") as path_staging:
            serialize_kinase_dict(dict_kinaseinfo, str_path=path_staging)
            # hash the files exactly as they will be tarred, so loads verify the bytes
            manifest = Manifest.from_kinase_dict(
                dict_kinaseinfo,
                git=_return_git_info(),
                packages=_return_package_versions(),
                entry_sha256=return_dir_entry_sha256(path_staging),
                sources=return_manifest_sources(dict_kinaseinfo),
            )
            path_manifest = os.path.join(path_staging, STR_MANIFEST_FILENAME)
            with open(path_manifest, "w") as outfile:
                outfile.write(manifest.model_dump_json(indent=4))
            # write beside the target, then swap atomically: a failed build keeps the
            # previous archive instead of deleting it first
            path_partial = f"{self.path_tar}.partial"
            try:
                create_tar_without_metadata(
                    path_source=path_staging, filename_tar=path_partial
                )
                os.replace(path_partial, self.path_tar)
            finally:
                if os.path.exists(path_partial):
                    os.remove(path_partial)
        logger.info(f"built {self.path_tar}\n{manifest.return_summary()}")

    def _dated_reports_dir(self) -> str:
        """Return (and create) the reports subdir for this archive version.

        Named by the manifest's ``generated_at`` (:data:`DATETIME_SUBDIR_FMT`) under
        :data:`REPORTS_GROUP_SUBDIR`, so each archive version gets exactly one folder that
        figures-only re-runs reuse.

        Returns
        -------
        str
            Absolute path ``{path_reports}/dict_kinase/{generated_at}`` (created if absent).

        Raises
        ------
        ArgumentError
            If the archive has no manifest (it has no version to name the folder by).
        """
        from mkt.databases.plot_config import ArgumentError

        manifest = load_manifest(self.path_tar)
        if manifest is None:
            raise ArgumentError(
                f"no {STR_MANIFEST_FILENAME} in {self.path_tar}; report folders are named by "
                "the manifest's generated_at, so rebuild the archive with --data."
            )
        path_dated = os.path.join(
            self.path_reports,
            REPORTS_GROUP_SUBDIR,
            manifest.generated_at.strftime(DATETIME_SUBDIR_FMT),
        )
        os.makedirs(path_dated, exist_ok=True)
        return path_dated

    def _reports_target(self) -> tuple[str, Any]:
        """Return the report output dir and the loaded report config.

        With a ``--config`` study YAML, figures go to a per-task subdir of the study dir
        (``<output.subdir>/<config-stem>/kinaseinfo/``) and use its ``kinaseinfo`` aesthetics;
        without one (a one-off CLI regen), they use the datetime-stamped
        ``dict_kinase/<generated_at>`` convention with default aesthetics.

        Returns
        -------
        tuple[str, KinaseInfoFiguresConfig]
            The (created) output directory and the report config.
        """
        from mkt.databases.plot_config import KinaseInfoFiguresConfig, load_task_config

        cfg = load_task_config(KinaseInfoFiguresConfig, self.config_path, "kinaseinfo")
        if self.config_path is None:
            return self._dated_reports_dir(), cfg
        config_name = os.path.splitext(os.path.basename(self.config_path))[0]
        path = os.path.join(
            get_repo_root(), cfg.output.subdir, config_name, "kinaseinfo"
        )
        os.makedirs(path, exist_ok=True)
        return path, cfg

    def _finalize(
        self,
        dict_kinaseinfo: dict[str, Any],
        names: list[str],
        subset_hgnc: set[str] | None,
        bool_figs: bool = True,
        force: bool = False,
        dict_step_subset: dict[str, set[str]] | None = None,
    ) -> None:
        """Run enrichment steps, serialize + tar, generate reports, and clean up.

        Shared tail of every mode: only the way the dict is produced differs.

        Parameters
        ----------
        dict_kinaseinfo : dict[str, Any]
            The assembled/updated dict to finalize.
        names : list[str]
            Enrichment step names to run.
        subset_hgnc : set[str] | None
            Targeted ``hgnc_name`` keys for a subset build (enrichment steps iterate only
            these); None for a full build (all entries).
        bool_figs : bool, optional
            Regenerate the report figures into the datetime-stamped reports subdir, by
            default True; ``--no-figs`` disables them.
        force : bool, optional
            Force structure steps to regenerate derived properties, by default False.
        dict_step_subset : dict[str, set[str]] | None, optional
            Per-step target keys overriding ``subset_hgnc`` (see
            :func:`~mkt.databases.generator.steps.run_steps`), by default None.

        Returns
        -------
        None
        """
        ctx = BuildContext(
            dict_kinaseinfo,
            self.path_objects,
            self.path_reports,
            self.path_tar,
            subset_hgnc=subset_hgnc,
            force=force,
        )
        build_steps.run_steps(names, ctx, dict_step_subset=dict_step_subset)
        self._serialize_and_tar(dict_kinaseinfo)
        if bool_figs:
            ctx.path_reports, ctx.report_config = self._reports_target()
            build_steps.run_reports(ctx)

    def figures(self) -> None:
        """Regenerate the report figures from the existing archive without rebuilding.

        Loads the currently serialized dict and renders the report steps into the reports
        subdir keyed by the existing archive's ``generated_at`` (reusing that directory), so
        figures can be refreshed without touching the data.

        Returns
        -------
        None
        """
        dict_existing = self._load_existing()
        if not dict_existing:
            logger.warning("no existing dict found; nothing to plot.")
            return
        path_reports, report_config = self._reports_target()
        ctx = BuildContext(
            dict_existing,
            self.path_objects,
            path_reports,
            self.path_tar,
            subset_hgnc=None,
            report_config=report_config,
        )
        build_steps.run_reports(ctx)

    def full(
        self, names: list[str], bool_figs: bool = True, force: bool = False
    ) -> None:
        """Full kinome regeneration: base build -> finalize.

        Parameters
        ----------
        names : list[str]
            Enrichment step names to run.
        bool_figs : bool, optional
            Regenerate report figures after the build, by default True.
        force : bool, optional
            Force structure steps to regenerate derived properties, by default False.

        Returns
        -------
        None
        """
        # register the previous archive's sources so unchanged files keep their query_date
        path_previous = (
            self.path_tar
            if os.path.exists(self.path_tar)
            else return_str_path_from_pkg_data()
        )
        manifest_previous = load_manifest(path_previous)
        if manifest_previous is not None:
            register_sources(manifest_previous.sources)
        dict_ki = run_base_build(subset_uniprot=None)
        self._finalize(
            dict_ki, names, subset_hgnc=None, bool_figs=bool_figs, force=force
        )

    def update(
        self,
        names: list[str],
        list_kinase: list[str],
        bool_figs: bool = True,
        force: bool = False,
    ) -> None:
        """One-off per-entry update: rebuild targeted kinases and splice into the archive.

        Parameters
        ----------
        names : list[str]
            Enrichment step names to run on the rebuilt entries.
        list_kinase : list[str]
            HGNC name(s) to rebuild; unknown names are resolved as new kinases or skipped.
        bool_figs : bool, optional
            Regenerate report figures after the splice, by default True.
        force : bool, optional
            Force structure steps to regenerate derived properties, by default False.

        Returns
        -------
        None
        """
        dict_full = self._load_existing()
        subset_uniprot, set_unresolved = _resolve_targets(list_kinase, dict_full)
        if set_unresolved:
            logger.warning(
                f"kinase(s) not in current set (attempting to add as new): "
                f"{sorted(set_unresolved)}"
            )
            new_uniprot, set_missing = _resolve_new_kinases(set_unresolved)
            subset_uniprot |= new_uniprot
            if set_missing:
                logger.warning(
                    f"could not resolve to UniProt IDs; skipping: {sorted(set_missing)}"
                )

        if not subset_uniprot:
            logger.warning(
                "no requested kinases resolved to UniProt IDs; nothing to update."
            )
            return

        dict_sub = run_base_build(subset_uniprot=subset_uniprot)
        if not dict_sub:
            logger.warning("base build produced no objects for the requested subset.")
            return

        subset_hgnc = merge_rebuilt_entries(dict_full, dict_sub, names, subset_uniprot)
        logger.info(
            f"spliced {len(dict_sub)} updated entr(ies) ({sorted(subset_hgnc)}) into "
            f"{len(dict_full)} total; re-archiving."
        )
        self._finalize(
            dict_full, names, subset_hgnc=subset_hgnc, bool_figs=bool_figs, force=force
        )

    def partial(
        self,
        sources: list[str],
        names: list[str],
        list_kinase: list[str] | None = None,
        bool_figs: bool = True,
        force: bool = False,
    ) -> None:
        """Partial update on the existing dict: refresh source(s) and/or run step(s).

        Loads the existing archive, optionally rebuilds the named base-build source(s), runs
        the enrichment steps (the requested ones plus everything downstream, resolved by the
        caller), and re-serializes. Rebuilt entries are merged field by field
        (:func:`merge_rebuilt_entries`), so fields of steps that don't re-run are kept. Falls
        back to a full regeneration when no existing dict is found.

        Parameters
        ----------
        sources : list[str]
            Base-build source names (:class:`Source`) to refresh; may be empty (steps only).
        names : list[str]
            Enrichment step names to run.
        list_kinase : list[str] | None, optional
            Restrict the update to these existing kinases (``--kinase``), by default None
            (every entry).
        bool_figs : bool, optional
            Regenerate report figures after the update, by default True.
        force : bool, optional
            Force structure steps to regenerate derived properties, by default False.

        Returns
        -------
        None
        """
        dict_existing = self._load_existing()
        if not dict_existing:
            logger.warning(
                "no existing dict found; falling back to a full regeneration "
                f"(requested {sorted(sources) + sorted(names)} built fresh)."
            )
            self.full(names, bool_figs=bool_figs, force=force)
            return

        subset_uniprot = None
        if list_kinase:
            subset_uniprot, set_unresolved = _resolve_targets(
                list_kinase, dict_existing
            )
            if set_unresolved:
                logger.warning(
                    f"--only applies to existing kinases; skipping {sorted(set_unresolved)} "
                    "(add new kinases with --kinase alone)."
                )
            if not subset_uniprot:
                logger.warning(
                    "no requested kinases are in the archive; nothing to do."
                )
                return

        if sources:
            dict_new = run_source_rebuild(sources, dict_existing, subset_uniprot)
            logger.info(
                f"source rebuild ({sorted(sources)}) produced {len(dict_new)} entries "
                f"(was {len(dict_existing)}); re-archiving."
            )
            set_base_uniprot = (
                subset_uniprot
                if subset_uniprot is not None
                else {
                    _strip_kd_suffix(obj.uniprot_id) for obj in dict_existing.values()
                }
            )
            # entries with no existing counterpart have nothing to carry over
            set_new = set(dict_new) - set(dict_existing)
            subset_hgnc = merge_rebuilt_entries(
                dict_existing, dict_new, names, set_base_uniprot
            )
        elif subset_uniprot is not None:
            set_new = set()
            subset_hgnc = {
                hgnc_name
                for hgnc_name, obj in dict_existing.items()
                if _strip_kd_suffix(obj.uniprot_id) in subset_uniprot
            }
        else:
            set_new = set()
            subset_hgnc = set(dict_existing)

        # existing entries get the requested + downstream steps; new entries get every step
        dict_step_subset = None
        if set_new:
            logger.info(
                f"new entr(ies) {sorted(set_new)} have no existing data; running every "
                "enrichment step on them."
            )
            set_names = set(names)
            dict_step_subset = {
                name: (subset_hgnc if name in set_names else set()) | set_new
                for name in build_steps.resolve_step_names()
            }
            names = list(dict_step_subset)
        self._finalize(
            dict_existing,
            names,
            subset_hgnc=subset_hgnc,
            bool_figs=bool_figs,
            force=force,
            dict_step_subset=dict_step_subset,
        )

    def run(
        self,
        only: list[str] | None = None,
        skip: list[str] | None = None,
        list_kinase: list[str] | None = None,
        bool_data: bool | None = None,
        bool_figs: bool = True,
        force: bool = False,
    ) -> None:
        """Dispatch to the run mode implied by the arguments.

        Parameters
        ----------
        only : list[str] | None, optional
            Components to rebuild on the existing dict: data sources
            (hgnc/uniprot/kinhub/klifs/pfam/kincore) and/or enrichment steps, plus every
            step downstream of them. Any ``only`` triggers a partial update; mutually
            exclusive with ``skip``.
        skip : list[str] | None, optional
            Skip these enrichment steps in a full regen; all other steps run.
        list_kinase : list[str] | None, optional
            HGNC name(s) to update; alone it rebuilds and splices those entries (adding new
            kinases), and with ``only`` it restricts the partial update to them. None (with
            no ``only``) runs a full regen.
        bool_data : bool | None, optional
            Build or update the archive; False draws figures from the existing archive. None
            defers to the config's ``kinaseinfo.data`` (True if unset), by default None.
        bool_figs : bool, optional
            Draw the report figures, by default True.
        force : bool, optional
            Recompute structure-derived properties (AlphaFold slice, SASA, superposition) even
            when already present, by default False.

        Returns
        -------
        None
        """
        from mkt.databases.plot_config import (
            ArgumentError,
            KinaseInfoFiguresConfig,
            load_task_config,
        )

        # validate --only/--skip once, before any work, against every valid component
        if only and skip:
            raise ArgumentError("--only and --skip are mutually exclusive.")
        list_sources = build_steps.return_component_names("source")
        list_steps = build_steps.return_component_names("step")
        list_components = list_sources + list_steps
        unknown_only = [name for name in only or [] if name not in list_components]
        if unknown_only:
            raise ArgumentError(
                f"unknown --only component(s) {unknown_only}; valid: {list_components}."
            )
        unknown_skip = [name for name in skip or [] if name not in list_steps]
        if unknown_skip:
            raise ArgumentError(
                f"unknown --skip component(s) {unknown_skip}; valid: {list_steps}."
            )

        cfg = load_task_config(KinaseInfoFiguresConfig, self.config_path, "kinaseinfo")
        if not cfg.resolve_data(bool_data, bool_figs):
            if only or skip or list_kinase:
                raise ArgumentError(
                    "--only/--skip/--kinase select data to rebuild, but data is off "
                    "(--no-data or kinaseinfo.data: false); pass --data."
                )
            self.figures()
            return

        sources = [name for name in only or [] if name in list_sources]
        # requested steps plus everything downstream of a requested source or step
        names = build_steps.resolve_step_names(only, skip)
        if skip:
            str_scope = (
                f"the --kinase entries ({', '.join(list_kinase)})"
                if list_kinase
                else "all entries"
            )
            build_steps.warn_skipped_steps(skip, names, str_scope)

        if only:
            self.partial(
                sources,
                names,
                list_kinase=list_kinase,
                bool_figs=bool_figs,
                force=force,
            )
        elif list_kinase:
            self.update(names, list_kinase, bool_figs=bool_figs, force=force)
        else:
            self.full(names, bool_figs=bool_figs, force=force)


def run(
    only: list[str] | None = None,
    skip: list[str] | None = None,
    list_kinase: list[str] | None = None,
    path_objects: str | None = None,
    path_reports: str | None = None,
    bool_data: bool | None = None,
    bool_figs: bool = True,
    force: bool = False,
    config_path: str | None = None,
) -> None:
    """Build a :class:`Pipeline` from the given paths and run it (CLI entry point).

    Parameters
    ----------
    only : list[str] | None, optional
        Components to rebuild (data sources and/or enrichment steps); mutually exclusive
        with ``skip``.
    skip : list[str] | None, optional
        Enrichment steps to skip in a full regen.
    list_kinase : list[str] | None, optional
        HGNC name(s) to update one-off; None (with no source) runs a full regen.
    path_objects : str | None, optional
        Objects directory relative to the repo root, by default the package-data layout.
    path_reports : str | None, optional
        Reports directory relative to the repo root, by default ``images``.
    bool_data : bool | None, optional
        Build or update the archive; None defers to the config's ``kinaseinfo.data``, by
        default None.
    bool_figs : bool, optional
        Draw the report figures, by default True.
    force : bool, optional
        Recompute structure-derived properties even when already present, by default False.
    config_path : str | None, optional
        Shared study YAML for report aesthetics + ``<config-stem>/kinaseinfo`` output naming,
        by default None (``dict_kinase/<generated_at>`` reports dir).

    Returns
    -------
    None
    """
    Pipeline.from_paths(path_objects, path_reports, config_path=config_path).run(
        only,
        skip,
        list_kinase,
        bool_data=bool_data,
        bool_figs=bool_figs,
        force=force,
    )
