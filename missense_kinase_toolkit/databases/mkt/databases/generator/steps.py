"""Enrichment- and report-step registries for the KinaseInfo build pipeline.

Defines the ordered registry of enrichment steps (each mutating additive optional
fields on the assembled :class:`KinaseInfo` objects in place), the terminal report
steps, and the selection/validation helpers that back the ``--only``/``--skip`` CLI
flags. Enrichment steps are added incrementally per workstream (alphafold, then rsasa,
activation_loop, alignment, exon). Heavy steps default off (see ``_DEFAULT_OFF``) and run
via ``--only``; every step must be idempotent (overwrite its field, never append) so
``--only <step>`` and ``--kinase`` splicing are safe to re-run.
"""

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from mkt.databases.generator.pipeline import BuildContext

logger = logging.getLogger(__name__)


def _iter_targets(ctx: "BuildContext"):
    """Yield ``(hgnc_name, KinaseInfo)`` for the targeted subset, or all entries.

    Parameters
    ----------
    ctx : BuildContext
        The build context; ``ctx.subset_hgnc`` (when not None) limits iteration to the
        targeted entries.

    Yields
    ------
    tuple[str, KinaseInfo]
        Each targeted ``(hgnc_name, object)`` pair.
    """
    if ctx.subset_hgnc is None:
        yield from ctx.dict_kinaseinfo.items()
    else:
        for hgnc_name in ctx.subset_hgnc:
            if hgnc_name in ctx.dict_kinaseinfo:
                yield hgnc_name, ctx.dict_kinaseinfo[hgnc_name]


def _enrich_alphafold(ctx: "BuildContext") -> None:
    """Store the KD-sliced AlphaFold structure on entries lacking a KinCore CIF.

    Per-entry failures are logged and skipped so one kinase never aborts the batch.

    Parameters
    ----------
    ctx : BuildContext
        The build context.

    Returns
    -------
    None
    """
    from mkt.databases.alphafold import enrich_with_alphafold

    for hgnc_name, obj_kinase in _iter_targets(ctx):
        try:
            enrich_with_alphafold(obj_kinase)
        except Exception as e:
            logger.error(
                f"alphafold enrichment failed for {hgnc_name}: {e}", exc_info=True
            )


def _enrich_sasa(ctx: "BuildContext") -> None:
    """Store KLIFS-pocket SASA/RSA computed over the adjudicated KD structure.

    Logs the SASA methodology once up front (also recorded per-entry on the stored
    :class:`SASA`); per-entry failures are logged and skipped so one kinase never aborts the
    batch.

    Parameters
    ----------
    ctx : BuildContext
        The build context.

    Returns
    -------
    None
    """
    from mkt.databases.sasa import (
        DEFAULT_SASA_CONFIG,
        MAX_ASA_REFERENCE,
        enrich_kinases_with_sasa,
    )

    cfg = DEFAULT_SASA_CONFIG
    logger.info(
        "SASA methodology: %s, probe_radius=%.2f A, n_points=%d, heavy-atom; "
        "RSA normalized by %s.",
        "Shrake-Rupley (Bio.PDB)" if cfg.bool_biopython else "dot_solvent (PyMOL)",
        cfg.probe_radius,
        cfg.n_points,
        MAX_ASA_REFERENCE,
    )

    # parallelize the CPU-bound per-residue SASA across all cores
    dict_targets = dict(_iter_targets(ctx))
    enrich_kinases_with_sasa(dict_targets, config=cfg, n_jobs=-1)


# ordered enrichment-step registry; each step takes a BuildContext and mutates additive
# optional fields on ctx.dict_kinaseinfo in place. steps run in this insertion order.
_ENRICH_STEPS: dict[str, Callable[["BuildContext"], None]] = {
    "alphafold": _enrich_alphafold,
    "sasa": _enrich_sasa,
}
"""dict[str, Callable]: Ordered enrichment-step registry (name -> step function)."""

_DEFAULT_OFF: set[str] = {"alphafold", "sasa"}
"""set[str]: Enrichment steps skipped in a full regen unless explicitly named via ``--only``
(alphafold fetches an AlphaFold structure per KinCore-less entry; sasa runs converged
Shrake-Rupley SASA over every structure -- both heavy, opt-in)."""

_STEP_DEPS: dict[str, set[str]] = {"alphafold": set(), "sasa": {"alphafold"}}
"""dict[str, set[str]]: Enrichment-step name -> prerequisite step names. alphafold reads the
base-build ``kincore`` field (always populated before steps run); sasa reads the adjudicated
structure, so the AlphaFold fallback should be materialized first for KinCore-less entries."""

_DEFAULT_STEPS: list[str] = [name for name in _ENRICH_STEPS if name not in _DEFAULT_OFF]
"""list[str]: Steps run when neither ``--only`` nor ``--skip`` is given (registry order)."""


def resolve_step_names(
    only: list[str] | None = None,
    skip: list[str] | None = None,
) -> list[str]:
    """Resolve the enrichment steps to run into registry order.

    Parameters
    ----------
    only : list[str] | None, optional
        Run only these steps (registry order); mutually exclusive with ``skip``.
    skip : list[str] | None, optional
        Skip these steps; all other default-on steps run.

    Returns
    -------
    list[str]
        Enrichment-step names to run, in registry order.

    Raises
    ------
    ValueError
        If both ``only`` and ``skip`` are given, or an unknown step is named.
    """
    if only and skip:
        raise ValueError("use --only or --skip, not both.")

    unknown = {
        name for name in (only or []) + (skip or []) if name not in _ENRICH_STEPS
    }
    if unknown:
        raise ValueError(
            f"unknown enrichment step(s): {sorted(unknown)}; "
            f"valid steps: {list(_ENRICH_STEPS)}."
        )

    if only:
        requested = set(only)
        return [name for name in _ENRICH_STEPS if name in requested]
    if skip:
        skipped = set(skip)
        return [name for name in _DEFAULT_STEPS if name not in skipped]
    return list(_DEFAULT_STEPS)


def _run_steps(names: list[str], ctx: "BuildContext") -> None:
    """Run the enabled enrichment steps sequentially in registry order.

    A failing step is logged and skipped rather than aborting the whole batch (step
    bodies additionally isolate per-kinase failures).

    Parameters
    ----------
    names : list[str]
        Enrichment-step names to run, in registry order.
    ctx : BuildContext
        The build context threaded through each step.
    """
    set_names = set(names)
    for name in names:
        missing = _STEP_DEPS.get(name, set()) - set_names
        if missing:
            logger.warning(
                f"enrichment step '{name}' requested without prerequisite step(s) "
                f"{sorted(missing)}; running anyway (they may already be materialized)."
            )
        logger.info(f"running enrichment step '{name}'...")
        try:
            _ENRICH_STEPS[name](ctx)
            logger.info(f"enrichment step '{name}' completed.")
        except Exception as e:
            logger.error(f"enrichment step '{name}' failed: {e}", exc_info=True)


def _report_upset(ctx: "BuildContext") -> None:
    """Generate the KinaseInfo data-source upset plot (preprint 2026 sizing)."""
    from mkt.databases.plot import plot_dict_kinase_upset
    from mkt.databases.plot_config import UpsetPlotConfig

    plot_dict_kinase_upset(
        ctx.dict_kinaseinfo, ctx.path_reports, cfg=UpsetPlotConfig.preprint_2026()
    )


def _report_region_gap_violin(ctx: "BuildContext") -> None:
    """Generate the inter-/intra-region gap violin plot (preprint 2026 sizing)."""
    from mkt.databases.plot import plot_region_gap_violin
    from mkt.databases.plot_config import RegionGapViolinConfig

    plot_region_gap_violin(
        ctx.dict_kinaseinfo, ctx.path_reports, cfg=RegionGapViolinConfig.preprint_2026()
    )


_REPORT_STEPS: dict[str, Callable[["BuildContext"], None]] = {
    "upset": _report_upset,
    "region_gap_violin": _report_region_gap_violin,
}
"""dict[str, Callable]: Terminal report steps, run only in full-regeneration mode."""


def _run_reports(ctx: "BuildContext") -> None:
    """Run the terminal report steps (skipped for subset/``--kinase`` builds).

    Parameters
    ----------
    ctx : BuildContext
        The build context; reports are skipped when ``ctx.subset_hgnc`` is not None
        because they characterize the whole kinome.
    """
    if ctx.subset_hgnc is not None:
        logger.info("subset build; skipping report steps.")
        return
    for name, fn in _REPORT_STEPS.items():
        logger.info(f"running report step '{name}'...")
        try:
            fn(ctx)
            logger.info(f"report step '{name}' completed.")
        except Exception as e:
            logger.error(f"report step '{name}' failed: {e}", exc_info=True)
