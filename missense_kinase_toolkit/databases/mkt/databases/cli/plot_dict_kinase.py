#!/usr/bin/env python3
"""CLI to plot the DICT_KINASE figures from YAML-configured aesthetics.

Renders the KinaseInfo figures built from the shipped ``DICT_KINASE`` archive:
the source-coverage upset plot and the combined UniProt->KLIFS residue map with
inter-/intra-region gap violins. Each figure is rendered only when its config
section is present in the YAML (both are rendered when no config is given).

This CLI deliberately does not import ``mkt.databases.datasets.process`` (which
has a network side effect on import), so it never needs the Davis/PKIS2 data.
"""

import logging
import os
from pathlib import Path
from typing import Annotated, Optional

import typer
from mkt.databases.log_config import configure_logging
from mkt.databases.plot import plot_dict_kinase_upset, plot_region_gap_violin
from mkt.databases.plot_config import DictKinaseFiguresConfig
from mkt.schema.io_utils import deserialize_kinase_dict, get_repo_root
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

app = typer.Typer()

# figures rendered when no --config is provided
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
    """Generate the DICT_KINASE figures (upset plot and region-gap map/violin).

    Loads aesthetics from a YAML config file when ``--config`` is provided,
    otherwise uses defaults. Each figure is rendered only when its config section
    (``upset_plot`` / ``region_gap_violin``) is present in the YAML.
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

    if "upset_plot" in present:
        plot_dict_kinase_upset(dict_kinase, output_dir, cfg=cfg.upset_plot)

    if "region_gap_violin" in present:
        plot_region_gap_violin(
            dict_kinase,
            output_dir,
            cfg=cfg.region_gap_violin,
            rc=cfg.matplotlib_rc,
        )

    logger.info("All plots generated successfully!")


if __name__ == "__main__":
    app()
