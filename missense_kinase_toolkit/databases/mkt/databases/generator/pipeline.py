"""Orchestration for the compositional KinaseInfo build pipeline.

Implements the three run modes of ``generate_kinaseinfo_objects``: a full kinome
regeneration, a per-step run (``--only``/``--skip`` over the enrichment registry), and a
per-entry one-off update (``--kinase``) that rebuilds targeted entries and splices them
back into the existing ``KinaseInfo.tar.gz`` without a full rebuild. The base build (API
fetch + harmonization) is a mandatory prerequisite; enrichment steps only mutate additive
optional fields on the assembled objects.
"""

import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any

from mkt.databases.generator import steps as build_steps
from mkt.databases.io_utils import create_tar_without_metadata
from mkt.databases.kinase_schema import (
    combine_kinaseinfo,
    combine_kinaseinfo_kd,
    combine_kinaseinfo_uniprot,
    generate_dict_obj_from_api_or_scraper,
)
from mkt.schema.io_utils import (
    deserialize_kinase_dict,
    get_repo_root,
    serialize_kinase_dict,
)

logger = logging.getLogger(__name__)


DEFAULT_PATH_OBJECTS = "missense_kinase_toolkit/schema/mkt/schema/KinaseInfo"
"""str: Default objects directory (relative to repo root), matching package-data layout."""

DEFAULT_PATH_REPORTS = "images"
"""str: Default reports directory (relative to repo root)."""


@dataclass
class BuildContext:
    """Mutable state threaded through the build pipeline and its steps.

    Attributes
    ----------
    dict_kinaseinfo : dict[str, Any]
        Assembled KinaseInfo objects keyed by ``hgnc_name`` (incl. ``_1``/``_2``
        multi-kinase-domain suffixes).
    path_objects : str
        Absolute path to the per-kinase serialization directory.
    path_reports : str
        Absolute path to the reports/figures directory.
    path_tar : str
        Absolute path to the ``KinaseInfo.tar.gz`` archive.
    subset_hgnc : set[str] | None
        When not None, the ``hgnc_name`` keys targeted by a subset (``--kinase``) build;
        enrichment steps iterate only these and report steps are skipped.
    """

    dict_kinaseinfo: dict[str, Any]
    path_objects: str
    path_reports: str
    path_tar: str
    subset_hgnc: set[str] | None = None


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


def _strip_kd_suffix(str_id: str) -> str:
    """Strip a trailing multi-kinase-domain suffix (e.g. ``_1``) from an id/name.

    Parameters
    ----------
    str_id : str
        HGNC name or UniProt ID, possibly suffixed with ``_<digit>``.

    Returns
    -------
    str
        The base id/name with any trailing ``_<digit>`` removed.
    """
    parts = str_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return str_id


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


def _serialize_and_tar(ctx: BuildContext) -> None:
    """Serialize the KinaseInfo dict to per-kinase files and (re)build the tar archive.

    Parameters
    ----------
    ctx : BuildContext
        The build context holding the dict and output paths.
    """
    serialize_kinase_dict(ctx.dict_kinaseinfo, str_path=ctx.path_objects)
    if os.path.exists(ctx.path_tar):
        os.remove(ctx.path_tar)
    create_tar_without_metadata(
        path_source=ctx.path_objects,
        filename_tar=ctx.path_tar,
    )


def _run_full(
    names: list[str],
    path_objects: str,
    path_reports: str,
    path_tar: str,
) -> None:
    """Full kinome regeneration: base build -> enrichments -> serialize -> reports."""
    dict_ki = run_base_build(subset_uniprot=None)
    ctx = BuildContext(dict_ki, path_objects, path_reports, path_tar, subset_hgnc=None)
    build_steps._run_steps(names, ctx)
    _serialize_and_tar(ctx)
    build_steps._run_reports(ctx)
    shutil.rmtree(path_objects)


def _run_update(
    names: list[str],
    list_kinase: list[str],
    path_objects: str,
    path_reports: str,
    path_tar: str,
) -> None:
    """One-off per-entry update: rebuild targeted kinases and splice into the archive."""
    if os.path.exists(path_tar):
        dict_full = deserialize_kinase_dict(str_path=path_tar)
    else:
        logger.info(
            f"no existing archive at {path_tar}; reading packaged KinaseInfo.tar.gz."
        )
        dict_full = deserialize_kinase_dict()

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

    subset_hgnc = set(dict_sub.keys())
    ctx_sub = BuildContext(
        dict_sub, path_objects, path_reports, path_tar, subset_hgnc=subset_hgnc
    )
    build_steps._run_steps(names, ctx_sub)

    # splice the updated entries into the full dict and re-serialize/re-tar
    dict_full.update(dict_sub)
    logger.info(
        f"spliced {len(dict_sub)} updated entr(ies) ({sorted(subset_hgnc)}) into "
        f"{len(dict_full)} total; re-archiving."
    )
    ctx_full = BuildContext(
        dict_full, path_objects, path_reports, path_tar, subset_hgnc=subset_hgnc
    )
    _serialize_and_tar(ctx_full)
    shutil.rmtree(path_objects)


def run(
    only: list[str] | None = None,
    skip: list[str] | None = None,
    list_kinase: list[str] | None = None,
    path_objects: str | None = None,
    path_reports: str | None = None,
) -> None:
    """Run the KinaseInfo build pipeline in the mode implied by the arguments.

    Parameters
    ----------
    only : list[str] | None, optional
        Run only these enrichment steps; mutually exclusive with ``skip``.
    skip : list[str] | None, optional
        Skip these enrichment steps; all other default-on steps run.
    list_kinase : list[str] | None, optional
        HGNC name(s) to update one-off; when given, only these entries are rebuilt and
        spliced into the existing archive (reports skipped). None runs a full regen.
    path_objects : str | None, optional
        Objects directory relative to the repo root, by default the package-data layout.
    path_reports : str | None, optional
        Reports directory relative to the repo root, by default ``images``.
    """
    path_repo = get_repo_root()
    path_objects = _resolve_dir(path_repo, path_objects, DEFAULT_PATH_OBJECTS)
    path_reports = _resolve_dir(path_repo, path_reports, DEFAULT_PATH_REPORTS)
    path_tar = os.path.normpath(os.path.join(path_objects, "..", "KinaseInfo.tar.gz"))

    names = build_steps.resolve_step_names(only, skip)

    if list_kinase:
        _run_update(names, list_kinase, path_objects, path_reports, path_tar)
    else:
        _run_full(names, path_objects, path_reports, path_tar)
