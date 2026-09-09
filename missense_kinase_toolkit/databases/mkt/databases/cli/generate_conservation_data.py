#!/usr/bin/env python3
"""CLI to build and serialize the KLIFS hierarchical-conservation artifact and its figures.

Entry point (``generate_conservation_data``) that constructs a
:class:`mkt.databases.conservation.KLIFSHierarchicalConservation` engine over the shipped
``DICT_KINASE`` KLIFS panel, packages its distances + dendrogram + provenance as a
:class:`mkt.schema.conservation_schema.KLIFSConservationData`, and serializes it as
``mkt.schema`` package data. Mirroring ``generate_kinaseinfo_objects``, it also renders the
conservation figures from that artifact -- ``--figs-only`` re-renders them without rebuilding
the data, ``--no-figs`` builds the data alone. Figure aesthetics come from the ``conservation``
section of a shared study YAML (``--config``); each figure renders only when its config section
is present (the static tree renders by default when no config is given).
"""

import logging
import os
from pathlib import Path
from typing import Annotated, Callable, Optional

import typer
from mkt.databases.log_config import configure_logging
from mkt.databases.plot_config import ConservationFiguresConfig, load_task_config
from mkt.schema.io_utils import get_repo_root, serialize_conservation_data
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Generate the persisted KLIFS conservation-data artifact and its figures.",
    no_args_is_help=False,
)

TASK_KEY = "conservation"
"""str: Top-level namespace this CLI reads from the shared study YAML."""


def _fig_conservation_tree(output_dir: str, cfg: ConservationFiguresConfig) -> None:
    """Static KLIFS conservation-tree supplement (summary + top/bottom panels)."""
    from mkt.databases.conservation import (
        KLIFSConservationTreeFigure,
        load_conservation_renderer,
    )

    tree_cfg = cfg.conservation_tree
    load_conservation_renderer(
        KLIFSConservationTreeFigure,
        min_cluster_size=tree_cfg.min_cluster_size,
        font_size=tree_cfg.font_size,
        split_index=tree_cfg.split_index,
    ).plot_split(output_dir, formats=tuple(tree_cfg.formats))


def _fig_conservation_tree_explorer(
    output_dir: str, cfg: ConservationFiguresConfig
) -> None:
    """Interactive KLIFS conservation-tree Bokeh explorer (standalone HTML)."""
    from mkt.databases.conservation import (
        KLIFSTreeConservationApp,
        load_conservation_renderer,
    )

    app_cfg = cfg.conservation_tree_explorer
    load_conservation_renderer(
        KLIFSTreeConservationApp,
        min_cluster_size=app_cfg.min_cluster_size,
        logo_cutoff=app_cfg.logo_cutoff,
        name_trunc=app_cfg.name_trunc,
    ).save_app(output_dir, filename=app_cfg.filename)


def _fig_residue_dot(output_dir: str, cfg: ConservationFiguresConfig) -> None:
    """Static per-amino-acid KLIFS dot-plot figure (default cysteine)."""
    from mkt.databases.conservation import (
        KLIFSConservationTreeFigure,
        load_conservation_renderer,
    )

    dot_cfg = cfg.residue_dot
    load_conservation_renderer(
        KLIFSConservationTreeFigure,
        min_cluster_size=dot_cfg.min_cluster_size,
    ).plot_residue_dot(
        output_dir,
        aa=dot_cfg.amino_acid,
        formats=tuple(dot_cfg.formats),
        highlight_targets=dot_cfg.highlight_targets,
    )


def _fig_residue_dot_explorer(output_dir: str, cfg: ConservationFiguresConfig) -> None:
    """Interactive per-amino-acid KLIFS dot-plot Bokeh explorer (standalone HTML)."""
    from mkt.databases.conservation import (
        KLIFSResidueDotApp,
        load_conservation_renderer,
    )

    app_cfg = cfg.residue_dot_explorer
    load_conservation_renderer(
        KLIFSResidueDotApp,
        min_cluster_size=app_cfg.min_cluster_size,
        default_aa=app_cfg.default_aa,
    ).save_app(output_dir, filename=app_cfg.filename)


def _fig_clade_membership_table(
    output_dir: str, cfg: ConservationFiguresConfig
) -> None:
    """LaTeX table of named conservation clades and their member kinases."""
    from mkt.databases.plot import write_clade_membership_table
    from mkt.schema.io_utils import load_conservation_data

    table_cfg = cfg.clade_membership_table
    write_clade_membership_table(
        load_conservation_data(),
        str_group=table_cfg.str_group,
        str_filepath=os.path.join(output_dir, f"{table_cfg.filename}.tex"),
    )


# registry of figure steps keyed by config section, rendered in this order
_FIGURE_STEPS: dict[str, Callable[[str, ConservationFiguresConfig], None]] = {
    "conservation_tree": _fig_conservation_tree,
    "conservation_tree_explorer": _fig_conservation_tree_explorer,
    "residue_dot": _fig_residue_dot,
    "residue_dot_explorer": _fig_residue_dot_explorer,
    "clade_membership_table": _fig_clade_membership_table,
}

# rendered when no --config is given (the primary static supplement)
_DEFAULT_FIGURES = {"conservation_tree"}


def _run_figures(config_path: Optional[Path]) -> None:
    """Render the conservation figures whose config section is present (defaults otherwise)."""
    cfg = load_task_config(ConservationFiguresConfig, config_path, TASK_KEY)
    if config_path is not None:
        section = OmegaConf.load(config_path).get(TASK_KEY, {}) or {}
        sections = set(section.keys())
    else:
        sections = set(_DEFAULT_FIGURES)

    # figures go to a per-task subdir of the study dir: <output.subdir>/<config-stem>/<task>
    config_name = Path(config_path).stem if config_path is not None else "default"
    output_dir = os.path.join(get_repo_root(), cfg.output.subdir, config_name, TASK_KEY)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Rendering conservation figures into {output_dir}")

    for name, render in _FIGURE_STEPS.items():
        if name in sections:
            logger.info(f"Rendering {name}")
            render(output_dir, cfg)


@app.command()
def main(
    metric: Annotated[
        str,
        typer.Option("--metric", help="Pairwise distance metric: blosum or identity."),
    ] = "blosum",
    linkage_method: Annotated[
        str,
        typer.Option(
            "--linkage-method", help="SciPy linkage method: average or complete."
        ),
    ] = "average",
    weighting: Annotated[
        str,
        typer.Option(
            "--weighting", help="Per-node consensus weighting: none or henikoff."
        ),
    ] = "none",
    conservation_threshold: Annotated[
        float,
        typer.Option(
            "--conservation-threshold",
            help="Minimum consensus-residue frequency for a conserved column.",
        ),
    ] = 0.80,
    exclude_pseudokinases: Annotated[
        bool,
        typer.Option(
            "--exclude-pseudokinases",
            help="Drop predicted pseudokinases from the panel before clustering.",
        ),
    ] = False,
    config_path: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            help="Shared study YAML; the 'conservation' section supplies figure aesthetics. "
            "Each figure renders only when its section is present (defaults otherwise).",
        ),
    ] = None,
    figs_only: Annotated[
        bool,
        typer.Option(
            "--figs-only",
            help="Only (re)render the figures from the existing artifact; skip the rebuild.",
        ),
    ] = False,
    no_figs: Annotated[
        bool,
        typer.Option("--no-figs", help="Build the data artifact only; skip figures."),
    ] = False,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to write the artifact into. Default: the mkt.schema "
            "package directory (shipped as package data).",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose (DEBUG) logging."),
    ] = False,
) -> None:
    """Generate and serialize the KLIFS conservation-data artifact and its figures.

    Examples:
        # default panel (BLOSUM62 + UPGMA), shipped as package data, + the static tree figure
        generate_conservation_data

        # rebuild + render all figures configured under the 'conservation' section
        generate_conservation_data --config configs/paper_2026.yaml

        # re-render figures only, without rebuilding the artifact
        generate_conservation_data --config configs/paper_2026.yaml --figs-only
    """
    configure_logging(verbose=verbose)

    if figs_only:
        _run_figures(config_path)
        return

    # imported lazily: constructing the engine builds the KLIFS panel from DICT_KINASE
    from mkt.databases.conservation import KLIFSHierarchicalConservation

    engine = KLIFSHierarchicalConservation(
        metric=metric,
        linkage_method=linkage_method,
        weighting=weighting,
        conservation_threshold=conservation_threshold,
        exclude_pseudokinases=exclude_pseudokinases,
    )
    logger.info(
        "built conservation engine: %d kinases, metric=%s, linkage=%s",
        len(engine.names),
        metric,
        linkage_method,
    )

    conservation_data = engine.to_conservation_data()
    str_path = str(output_dir) if output_dir is not None else None
    filepath = serialize_conservation_data(conservation_data, str_path=str_path)
    typer.echo(f"KLIFSConservationData written to: {filepath}")

    if not no_figs:
        _run_figures(config_path)


if __name__ == "__main__":
    app()
