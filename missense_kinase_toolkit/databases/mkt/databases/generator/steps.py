"""Component and report-step registries for the KinaseInfo build pipeline.

:data:`COMPONENTS` lists every build component in run order: the base-build sources, then
the enrichment steps (each mutating its owned fields on the assembled :class:`KinaseInfo`
objects in place). Each declares what it reads and writes, which drives ``--only``
(requested components plus everything downstream), ``--skip`` warnings, partial-rebuild
merges, and the CLI help. Every step must be idempotent (overwrite its fields, never append)
so ``--only`` and ``--kinase`` re-runs are safe.
"""

import logging
from dataclasses import dataclass
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


def _enrich_kincore_structure_props(ctx: "BuildContext") -> None:
    """Compute the KinCoRe active-state CIF's derived properties (SASA + superposition).

    (Re)generates ``kincore.cif.sasa`` and ``kincore.cif.superposition``; the CIF itself is
    fetched by the ``kincore`` source (``--only kincore``).

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
    ``--recompute``) and, alongside it, its ``alphafold.sasa`` and
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


def _enrich_exon(ctx: "BuildContext") -> None:
    """Annotate entries with a per-residue exon map from GenomeNexus canonical transcripts.

    Populates ``exon`` (UniProt index -> exon number) for each entry, sharing a gene's transcript
    across its ``_1``/``_2`` domains. Reads only the UniProt canonical sequence, so it has no
    inter-step dependency.

    Parameters
    ----------
    ctx : BuildContext
        The build context.

    Returns
    -------
    None
    """
    from mkt.databases.genomenexus import enrich_kinases_with_exons

    enrich_kinases_with_exons(dict(_iter_targets(ctx)))


@dataclass(frozen=True)
class Component:
    """One build component: a base-build source or an enrichment step."""

    name: str
    """Name used by ``--only``/``--skip``."""
    kind: str
    """``"source"`` (fetched in the base build) or ``"step"`` (enrichment)."""
    help: str
    """One-line description for the CLI help."""
    reads: frozenset[str] = frozenset()
    """Sources and steps whose output it reads, by default empty."""
    writes: tuple[str, ...] = ()
    """Dotted KinaseInfo fields a step owns, by default empty. A partial rebuild carries
    these over when the step does not run, and clears them before it re-runs unless
    ``checks_inputs``."""
    run: Callable[["BuildContext"], None] | None = None
    """Step function; None for sources (fetched by ``fetch_source``)."""
    checks_inputs: bool = False
    """Step keeps a stored value whose recorded inputs are unchanged (``InputCheck``), so a
    re-run carries its fields over instead of clearing them, by default False."""


# run order: sources, then steps. kincore_msa runs first so its KD bounds (and the MSA
# superposition tier) are available to the structure steps; each structure step owns its
# structure's derived properties (SASA + superposition).
COMPONENTS: dict[str, Component] = {
    component.name: component
    for component in [
        Component("hgnc", "source", "HGNC gene symbols"),
        Component("uniprot", "source", "UniProt canonical sequences and phosphosites"),
        Component("kinhub", "source", "KinHub kinase classification"),
        Component("klifs", "source", "KLIFS pocket annotations"),
        Component("pfam", "source", "Pfam kinase-domain boundaries"),
        Component("kincore", "source", "KinCoRe kinase-domain FASTA and CIFs"),
        Component(
            "kincore_msa",
            "step",
            "Dunbrack MSA rows and kinase-domain bounds",
            # KLIFS2UniProtIdx (uniprot + klifs + kincore) picks the domain; hgnc fallback
            reads=frozenset({"hgnc", "uniprot", "klifs", "kincore"}),
            writes=("kincore.msa",),
            run=_enrich_kincore_msa,
        ),
        Component(
            "kincore_structure_props",
            "step",
            "SASA and superposition of the KinCoRe structure",
            # SASA keys on KLIFS2UniProtIdx; superposition falls back to the MSA tier
            reads=frozenset({"uniprot", "klifs", "kincore", "kincore_msa"}),
            writes=("kincore.cif.sasa", "kincore.cif.superposition"),
            run=_enrich_kincore_structure_props,
            checks_inputs=True,
        ),
        Component(
            "alphafold",
            "step",
            "KD-sliced AlphaFold structure, its SASA and superposition",
            # slices to adjudicated KD bounds: kincore cif > fasta > msa > pfam > KLIFS span
            reads=frozenset({"uniprot", "klifs", "kincore", "pfam", "kincore_msa"}),
            writes=("alphafold",),
            run=_enrich_alphafold,
            checks_inputs=True,
        ),
        Component(
            "exon",
            "step",
            "exon map from Genome Nexus",
            reads=frozenset({"hgnc", "uniprot"}),
            writes=("exon",),
            run=_enrich_exon,
        ),
    ]
}
"""dict[str, Component]: Every build component, in run order (name -> Component)."""


def return_component_names(kind: str | None = None) -> list[str]:
    """Return component names in run order, optionally of one kind (source/step)."""
    return [
        name
        for name, component in COMPONENTS.items()
        if kind is None or component.kind == kind
    ]


def return_step_writes() -> dict[str, tuple[str, ...]]:
    """Return each enrichment step's owned fields (step name -> dotted paths)."""
    return {name: COMPONENTS[name].writes for name in return_component_names("step")}


def warn_skipped_steps(skip: list[str], names: list[str], str_scope: str) -> None:
    """Warn which steps that will run read the output of a skipped step.

    Parameters
    ----------
    skip : list[str]
        Steps skipped with ``--skip``.
    names : list[str]
        Steps that will run.
    str_scope : str
        Which entries they run on, for the message (e.g. ``"all entries"``).
    """
    for str_skip in skip:
        list_downstream = [
            name for name in resolve_step_names(only=[str_skip]) if name in names
        ]
        if not list_downstream:
            continue
        list_fields = [
            path for name in list_downstream for path in COMPONENTS[name].writes
        ]
        logger.warning(
            f"--skip {str_skip}: {', '.join(list_downstream)} read its output and will "
            f"run on {str_scope} without a fresh {str_skip}, so these fields may regress: "
            f"{', '.join(list_fields)}. Rerun with --only {str_skip} to refresh them."
        )


def resolve_step_names(
    only: list[str] | None = None,
    skip: list[str] | None = None,
) -> list[str]:
    """Resolve the enrichment steps to run into registry order.

    ``only`` may name base-build sources and/or steps; the result is the requested steps
    plus every step downstream of a requested source or step (transitively, via each
    component's ``reads``), so nothing that depends on refreshed data is left stale. Names
    are validated by :meth:`mkt.databases.generator.pipeline.Pipeline.run`; unknown names
    are ignored here.

    Parameters
    ----------
    only : list[str] | None, optional
        Rebuild only these components and their downstream steps; takes precedence over
        ``skip``.
    skip : list[str] | None, optional
        Skip these steps; all other steps run.

    Returns
    -------
    list[str]
        Enrichment-step names to run, in registry order.
    """
    list_steps = return_component_names("step")
    if only:
        set_closure = set(only)
        bool_grew = True
        while bool_grew:
            set_downstream = {
                name
                for name in list_steps
                if name not in set_closure and COMPONENTS[name].reads & set_closure
            }
            set_closure |= set_downstream
            bool_grew = bool(set_downstream)
        return [name for name in list_steps if name in set_closure]
    skipped = set(skip or [])
    return [name for name in list_steps if name not in skipped]


def run_steps(
    names: list[str],
    ctx: "BuildContext",
    dict_step_subset: dict[str, set[str]] | None = None,
) -> None:
    """Run the enabled enrichment steps sequentially in registry order.

    A failing step is logged and skipped rather than aborting the whole batch (step
    bodies additionally isolate per-kinase failures).

    Parameters
    ----------
    names : list[str]
        Enrichment-step names to run, in registry order.
    ctx : BuildContext
        The build context threaded through each step.
    dict_step_subset : dict[str, set[str]] | None, optional
        Step name -> ``hgnc_name`` keys that step runs on, overriding ``ctx.subset_hgnc``
        per step, by default None (every step uses ``ctx.subset_hgnc``).
    """
    subset_default = ctx.subset_hgnc
    for name in names:
        if dict_step_subset is not None:
            ctx.subset_hgnc = dict_step_subset[name]
        logger.info(f"running enrichment step '{name}'...")
        try:
            COMPONENTS[name].run(ctx)
            logger.info(f"enrichment step '{name}' completed.")
        except Exception as e:
            logger.error(f"enrichment step '{name}' failed: {e}", exc_info=True)
    ctx.subset_hgnc = subset_default


def _report_cfg(ctx: "BuildContext"):
    """Return the report aesthetics config from the context (loaded defaults if unset)."""
    if ctx.report_config is not None:
        return ctx.report_config
    from mkt.databases.plot_config import KinaseInfoFiguresConfig, load_task_config

    return load_task_config(KinaseInfoFiguresConfig)


def _report_upset(ctx: "BuildContext") -> None:
    """Generate the KinaseInfo data-source upset plot."""
    from mkt.databases.plot import plot_dict_kinase_upset

    plot_dict_kinase_upset(
        ctx.dict_kinaseinfo, ctx.path_reports, cfg=_report_cfg(ctx).upset_plot
    )


def _report_region_gap_violin(ctx: "BuildContext") -> None:
    """Generate the inter-/intra-region gap violin plot."""
    from mkt.databases.plot import plot_region_gap_violin

    plot_region_gap_violin(
        ctx.dict_kinaseinfo, ctx.path_reports, cfg=_report_cfg(ctx).region_gap_violin
    )


def _report_sasa_concordance_scatter(ctx: "BuildContext") -> None:
    """Generate the KinCoRe-vs-AF2 per-region SASA/RSA concordance scatter."""
    from mkt.databases.plot import plot_sasa_concordance_scatter

    plot_sasa_concordance_scatter(
        ctx.dict_kinaseinfo,
        ctx.path_reports,
        cfg=_report_cfg(ctx).sasa_concordance_scatter,
    )


def _report_sasa_concordance_delta(ctx: "BuildContext") -> None:
    """Generate the per-KLIFS-residue KinCoRe-minus-AF2 SASA/RSA delta boxplots."""
    from mkt.databases.plot import plot_sasa_concordance_delta

    plot_sasa_concordance_delta(
        ctx.dict_kinaseinfo,
        ctx.path_reports,
        cfg=_report_cfg(ctx).sasa_concordance_delta,
    )


_REPORT_STEPS: dict[str, Callable[["BuildContext"], None]] = {
    "upset": _report_upset,
    "region_gap_violin": _report_region_gap_violin,
    "sasa_concordance_scatter": _report_sasa_concordance_scatter,
    "sasa_concordance_delta": _report_sasa_concordance_delta,
}
"""dict[str, Callable]: Terminal report steps, run only in full-regeneration mode."""


def run_reports(ctx: "BuildContext") -> None:
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
