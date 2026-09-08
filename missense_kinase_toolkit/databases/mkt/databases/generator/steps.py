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


def _enrich_structure_derived(ctx: "BuildContext", only: str) -> None:
    """Compute the structure-derived properties (SASA + superposition) for one structure type.

    Shared body of the two structure-owning steps: the SASA over each ``only`` structure runs
    in a process pool (parallel across cores), then each such structure is superposed onto the
    shared 1GAG reference frame. Both honor ``ctx.force`` (recompute even when already present).

    Parameters
    ----------
    ctx : BuildContext
        The build context.
    only : str
        Structure type to enrich: ``"kincore"`` (the KinCoRe CIF) or ``"alphafold"``.

    Returns
    -------
    None
    """
    from mkt.databases.sasa import (
        DEFAULT_SASA_CONFIG,
        MAX_ASA_REFERENCE,
        enrich_kinases_with_sasa,
    )
    from mkt.databases.superpose import build_reference_frame, superpose_structure

    force = getattr(ctx, "force", False)

    cfg = DEFAULT_SASA_CONFIG
    logger.info(
        "SASA methodology (%s): %s, probe_radius=%.2f A, n_points=%d, heavy-atom; "
        "RSA normalized by %s.",
        only,
        "Shrake-Rupley (Bio.PDB)" if cfg.bool_biopython else "dot_solvent (PyMOL)",
        cfg.probe_radius,
        cfg.n_points,
        MAX_ASA_REFERENCE,
    )

    # parallelize the CPU-bound per-residue SASA across all cores
    dict_targets = dict(_iter_targets(ctx))
    enrich_kinases_with_sasa(
        dict_targets, config=cfg, n_jobs=-1, only=only, force=force
    )

    # superpose each structure of this type onto the shared reference frame (built once)
    frame = build_reference_frame(ctx.dict_kinaseinfo)
    for hgnc_name, obj_kinase in _iter_targets(ctx):
        try:
            model = (
                obj_kinase.kincore.cif
                if only == "kincore" and obj_kinase.kincore is not None
                else obj_kinase.alphafold if only == "alphafold" else None
            )
            superpose_structure(
                model, obj_kinase, frame, f"{hgnc_name}_{only}", force=force
            )
        except Exception as e:
            logger.error(
                f"superpose ({only}) failed for {hgnc_name}: {e}", exc_info=True
            )


def _enrich_kincore_cif(ctx: "BuildContext") -> None:
    """Compute the KinCoRe active-state CIF's derived properties (SASA + superposition).

    The CIF coordinates come from the base build (the ``kincore`` source); this step
    (re)generates the derived ``kincore.cif.sasa`` and ``kincore.cif.superposition`` that
    travel with that structure. Named ``kincore_cif`` to avoid colliding with the ``kincore``
    base-build source in ``--only``.

    Parameters
    ----------
    ctx : BuildContext
        The build context.

    Returns
    -------
    None
    """
    _enrich_structure_derived(ctx, only="kincore")


def _enrich_alphafold(ctx: "BuildContext") -> None:
    """Fetch the KD-sliced AlphaFold structure and compute its derived properties.

    Regenerates the AF structure (re-sliced on KD-bound changes, or forced via
    ``--force-regen``) and, alongside it, its ``alphafold.sasa`` and
    ``alphafold.superposition``. Per-entry failures are logged and skipped so one kinase never
    aborts the batch.

    Parameters
    ----------
    ctx : BuildContext
        The build context.

    Returns
    -------
    None
    """
    from mkt.databases.alphafold import enrich_with_alphafold

    force = getattr(ctx, "force", False)
    for hgnc_name, obj_kinase in _iter_targets(ctx):
        try:
            enrich_with_alphafold(obj_kinase, force=force)
        except Exception as e:
            logger.error(
                f"alphafold enrichment failed for {hgnc_name}: {e}", exc_info=True
            )

    _enrich_structure_derived(ctx, only="alphafold")


def _enrich_kincore_msa(ctx: "BuildContext") -> None:
    """Annotate entries with the Dunbrack structure-based MSA (activation-loop coordinates).

    Populates ``kincore.msa`` (a KinCoRe component): maps each domain's Human-PK alignment row
    to UniProt coordinates (creating an MSA-only KinCoRe shell where structure is absent);
    matched by ``hgnc_name`` with a UniProt-accession fallback. Batch failures are logged and
    skipped by the enricher.

    Parameters
    ----------
    ctx : BuildContext
        The build context.

    Returns
    -------
    None
    """
    from mkt.databases.msa import enrich_kinases_with_msa

    enrich_kinases_with_msa(dict(_iter_targets(ctx)))


# ordered enrichment-step registry; each step takes a BuildContext and mutates additive
# optional fields on ctx.dict_kinaseinfo in place. steps run in this insertion order.
# kincore_msa runs first so its KD bounds / MSA-only shells (and the MSA superposition tier)
# are available to the structure steps. each structure step owns its structure's derived
# properties (SASA + reference-frame superposition), so they are (re)generated alongside the
# structure itself. KinCoRe-component steps share the kincore_* prefix (fasta needs no step --
# it is fully populated in the base build).
_ENRICH_STEPS: dict[str, Callable[["BuildContext"], None]] = {
    "kincore_msa": _enrich_kincore_msa,
    "kincore_cif": _enrich_kincore_cif,
    "alphafold": _enrich_alphafold,
}
"""dict[str, Callable]: Ordered enrichment-step registry (name -> step function)."""

_DEFAULT_OFF: set[str] = {"kincore_msa", "kincore_cif", "alphafold"}
"""set[str]: Enrichment steps skipped in a full regen unless explicitly named via ``--only``
(kincore_msa downloads the Dunbrack alignment; kincore_cif computes SASA + reference-frame
superposition over the KinCoRe CIF; alphafold fetches an AlphaFold structure per entry and
computes its SASA + superposition -- all opt-in and CPU-heavy)."""

_STEP_DEPS: dict[str, set[str]] = {
    "kincore_msa": set(),
    "kincore_cif": set(),
    "alphafold": set(),
}
"""dict[str, set[str]]: Enrichment-step name -> prerequisite step names. All read base-build
fields (KLIFS mapping, adjudicated bounds) that are always populated before steps run; each
structure step owns its structure's derived properties, so there is no inter-step dependency.
The MSA superposition tier benefits from ``kincore_msa`` running first, but degrades to the
sequence tier if absent."""

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


def _report_sasa_concordance_scatter(ctx: "BuildContext") -> None:
    """Generate the KinCoRe-vs-AF2 per-region SASA/RSA concordance scatter."""
    from mkt.databases.plot import plot_sasa_concordance_scatter
    from mkt.databases.plot_config import SASAConcordanceScatterConfig

    plot_sasa_concordance_scatter(
        ctx.dict_kinaseinfo, ctx.path_reports, cfg=SASAConcordanceScatterConfig()
    )


def _report_sasa_concordance_delta(ctx: "BuildContext") -> None:
    """Generate the per-KLIFS-residue KinCoRe-minus-AF2 SASA/RSA delta boxplots."""
    from mkt.databases.plot import plot_sasa_concordance_delta
    from mkt.databases.plot_config import SASAConcordanceDeltaConfig

    plot_sasa_concordance_delta(
        ctx.dict_kinaseinfo, ctx.path_reports, cfg=SASAConcordanceDeltaConfig()
    )


_REPORT_STEPS: dict[str, Callable[["BuildContext"], None]] = {
    "upset": _report_upset,
    "region_gap_violin": _report_region_gap_violin,
    "sasa_concordance_scatter": _report_sasa_concordance_scatter,
    "sasa_concordance_delta": _report_sasa_concordance_delta,
}
"""dict[str, Callable]: Terminal report steps, run only in full-regeneration mode."""


def _run_reports(ctx: "BuildContext") -> None:
    """Run the terminal report steps over the whole assembled dict.

    Reports always characterize the full kinome: ``ctx.dict_kinaseinfo`` is the complete
    dict in every mode (subset builds splice back before finalizing), so reports run
    regardless of ``ctx.subset_hgnc``. Whether they run at all is gated upstream by the
    ``--no-figs`` flag in :meth:`Pipeline._finalize`.

    Parameters
    ----------
    ctx : BuildContext
        The build context; ``ctx.path_reports`` is the (datetime-stamped) output directory.
    """
    for name, fn in _REPORT_STEPS.items():
        logger.info(f"running report step '{name}'...")
        try:
            fn(ctx)
            logger.info(f"report step '{name}' completed.")
        except Exception as e:
            logger.error(f"report step '{name}' failed: {e}", exc_info=True)
