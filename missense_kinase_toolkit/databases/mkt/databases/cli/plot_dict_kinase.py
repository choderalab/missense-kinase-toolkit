#!/usr/bin/env python3
"""CLI to plot the DICT_KINASE figures from YAML-configured aesthetics.

Renders the study-independent KinaseInfo figures built from the shipped
``DICT_KINASE`` archive: the source-coverage upset plot, the combined
UniProt->KLIFS residue map with inter-/intra-region gap violins, and the KLIFS
hierarchical conservation-tree supplement (static split figures + interactive
Bokeh explorer). Each figure is rendered only when its config section is present
in the YAML (the non-tree figures are rendered when no config is given).

Following ``mkt_impact.cli.run_analysis``, the figures are registered in a
``_PLOT_STEPS`` dict keyed by config section, so adding a figure is one entry.

This CLI deliberately does not import ``mkt.databases.datasets.process`` (which
has a network side effect on import); the conservation-tree renderers are
imported lazily inside their steps so the panel build only runs when requested.
"""

import logging
import os
from pathlib import Path
from typing import Annotated, Callable, Optional

import typer
from mkt.databases.log_config import configure_logging
from mkt.databases.plot import plot_dict_kinase_upset, plot_region_gap_violin
from mkt.databases.plot_config import DictKinaseFiguresConfig
from mkt.schema.io_utils import deserialize_kinase_dict, get_repo_root
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

app = typer.Typer()


def _plot_upset(
    dict_kinase: dict, output_dir: str, cfg: DictKinaseFiguresConfig
) -> None:
    """Source-coverage upset plot over the DICT_KINASE archive."""
    plot_dict_kinase_upset(dict_kinase, output_dir, cfg=cfg.upset_plot)


def _plot_region_gap_violin(
    dict_kinase: dict, output_dir: str, cfg: DictKinaseFiguresConfig
) -> None:
    """UniProt->KLIFS residue map with inter-/intra-region gap violins."""
    plot_region_gap_violin(
        dict_kinase,
        output_dir,
        cfg=cfg.region_gap_violin,
        rc=cfg.matplotlib_rc,
    )


def _plot_conservation_tree(
    dict_kinase: dict, output_dir: str, cfg: DictKinaseFiguresConfig
) -> None:
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


def _plot_conservation_tree_explorer(
    dict_kinase: dict, output_dir: str, cfg: DictKinaseFiguresConfig
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


def _plot_residue_dot(
    dict_kinase: dict, output_dir: str, cfg: DictKinaseFiguresConfig
) -> None:
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
        output_dir, aa=dot_cfg.amino_acid, formats=tuple(dot_cfg.formats)
    )


def _plot_residue_dot_explorer(
    dict_kinase: dict, output_dir: str, cfg: DictKinaseFiguresConfig
) -> None:
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


# registry of figure steps keyed by config section, rendered in this order
_PLOT_STEPS: dict[str, Callable[[dict, str, DictKinaseFiguresConfig], None]] = {
    "upset_plot": _plot_upset,
    "region_gap_violin": _plot_region_gap_violin,
    "conservation_tree": _plot_conservation_tree,
    "conservation_tree_explorer": _plot_conservation_tree_explorer,
    "residue_dot": _plot_residue_dot,
    "residue_dot_explorer": _plot_residue_dot_explorer,
}

# figures rendered when no --config is provided (the non-tree, always-cheap ones)
_DEFAULT_SECTIONS = {"upset_plot", "region_gap_violin"}


@app.command()
def main(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to YAML configuration file. If not provided, uses defaults.",
    ),
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose (DEBUG) logging."),
    ] = False,
) -> None:
    """Generate the DICT_KINASE figures.

    Loads aesthetics from a YAML config file when ``--config`` is provided,
    otherwise uses defaults. Each figure is rendered only when its config section
    (one of :data:`_PLOT_STEPS`) is present in the YAML; with no config the
    always-cheap non-tree figures (:data:`_DEFAULT_SECTIONS`) are rendered.
    """
    configure_logging(verbose=verbose)

    if config_path is not None:
        logger.info(f"Loading plot config from {config_path}")
        cfg = DictKinaseFiguresConfig.from_yaml(config_path)
        present = set(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
    else:
        logger.info("Using default plot config")
        cfg = DictKinaseFiguresConfig()
        present = set(_DEFAULT_SECTIONS)

    repo_root = get_repo_root()

    # output directory uses the config name (stem) as a subdirectory
    config_name = Path(config_path).stem if config_path is not None else "default"
    output_dir = os.path.join(repo_root, cfg.output.subdir, config_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    dict_kinase = deserialize_kinase_dict(str_name="DICT_KINASE")

    for section, render in _PLOT_STEPS.items():
        if section in present:
            logger.info(f"Rendering {section}")
            render(dict_kinase, output_dir, cfg)

    logger.info("All plots generated successfully!")


if __name__ == "__main__":
    app()
