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
import shutil
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import git
from mkt.databases.generator import steps as build_steps
from mkt.databases.io_utils import create_tar_without_metadata
from mkt.databases.kinase_schema import (
    Source,
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
    serialize_kinase_dict,
)
from mkt.schema.utils import split_domain_suffix

logger = logging.getLogger(__name__)


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
    """Absolute path to the per-kinase serialization directory."""
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
        if first.pfam is not None:
            dict_obj[Source.pfam][base] = first.pfam
        dict_obj[Source.kinhub][base] = [o.kinhub for o in list_obj]
        dict_obj[Source.klifs][base] = [o.klifs for o in list_obj]
        dict_obj[Source.kincore][base] = [o.kincore for o in list_obj]
    return dict_obj


def run_source_rebuild(
    sources: list[str],
    dict_existing: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the dict refreshing only the named base-build source(s).

    Reconstructs the raw dict_obj from ``dict_existing``, replaces each named source with a
    fresh :func:`fetch_source`, and re-runs the combine_* pipeline so every cross-source
    validator (KinCoRe alignments, KLIFS2UniProt mapping) recomputes.

    Parameters
    ----------
    sources : list[str]
        Source names (subset of :class:`Source`) to refresh.
    dict_existing : dict[str, Any]
        The currently serialized KinaseInfo dict.

    Returns
    -------
    dict[str, Any]
        The rebuilt KinaseInfo dict.
    """
    dict_obj = _reconstruct_dict_obj(dict_existing)
    set_uniprot = set(dict_obj[Source.uniprot].keys())
    for source in sources:
        logger.info(
            f"refreshing source '{source}' for {len(set_uniprot)} UniProt IDs..."
        )
        dict_obj[source] = fetch_source(source, set_uniprot)
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
    """Absolute path to the per-kinase serialization directory."""
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
        path_objects = _resolve_dir(path_repo, path_objects, DEFAULT_PATH_OBJECTS)
        path_reports = _resolve_dir(path_repo, path_reports, DEFAULT_PATH_REPORTS)
        path_tar = os.path.normpath(
            os.path.join(path_objects, "..", "KinaseInfo.tar.gz")
        )
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
        serialize_kinase_dict(dict_kinaseinfo, str_path=self.path_objects)
        manifest = Manifest.from_kinase_dict(
            dict_kinaseinfo,
            git=_return_git_info(),
            packages=_return_package_versions(),
        )
        path_manifest = os.path.join(self.path_objects, STR_MANIFEST_FILENAME)
        with open(path_manifest, "w") as outfile:
            outfile.write(manifest.model_dump_json(indent=4))
        if os.path.exists(self.path_tar):
            os.remove(self.path_tar)
        create_tar_without_metadata(
            path_source=self.path_objects, filename_tar=self.path_tar
        )
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
        build_steps.run_steps(names, ctx)
        self._serialize_and_tar(dict_kinaseinfo)
        if bool_figs:
            ctx.path_reports, ctx.report_config = self._reports_target()
            build_steps.run_reports(ctx)
        shutil.rmtree(self.path_objects)

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

        subset_hgnc = set(dict_sub)
        dict_full.update(dict_sub)
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
        bool_figs: bool = True,
        force: bool = False,
    ) -> None:
        """Partial update on the existing dict: refresh source(s) and/or run step(s).

        Loads the existing archive, optionally rebuilds the named base-build source(s), runs
        the named enrichment steps, and re-serializes. Falls back to a full regeneration when
        no existing dict is found.

        Parameters
        ----------
        sources : list[str]
            Base-build source names (:class:`Source`) to refresh; may be empty (steps only).
        names : list[str]
            Enrichment step names to run.
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

        if sources:
            dict_new = run_source_rebuild(sources, dict_existing)
            logger.info(
                f"source rebuild ({sorted(sources)}) produced {len(dict_new)} entries "
                f"(was {len(dict_existing)}); re-archiving."
            )
            dict_existing.update(dict_new)
            subset_hgnc = set(dict_new)
        else:
            subset_hgnc = set(dict_existing)
        self._finalize(
            dict_existing,
            names,
            subset_hgnc=subset_hgnc,
            bool_figs=bool_figs,
            force=force,
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
            (hgnc/uniprot/kinhub/klifs/pfam/kincore) and/or enrichment steps. Any ``only``
            triggers a partial update; mutually exclusive with ``skip``.
        skip : list[str] | None, optional
            Skip these enrichment steps in a full regen; all other steps run.
        list_kinase : list[str] | None, optional
            HGNC name(s) to update one-off; None (with no ``only``) runs a full regen.
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
        list_sources = [source.value for source in Source]
        list_steps = build_steps.resolve_step_names()
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
        only_steps = [name for name in only or [] if name in list_steps]
        names = build_steps.resolve_step_names(only_steps or None, skip)

        if only:
            self.partial(sources, names, bool_figs=bool_figs, force=force)
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
