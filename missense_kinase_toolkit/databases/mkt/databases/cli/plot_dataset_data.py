#!/usr/bin/env python3
"""CLI to plot processed kinase dataset data.

Entry point that renders figures from processed dataset data using YAML-configured plot
aesthetics. Each figure is rendered only when its config section is present in the YAML,
so a config can select any subset of the dataset plots and/or the KinaseInfo upset and
region-gap figures.
"""

import logging
import os
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from mkt.databases import config
from mkt.databases.log_config import configure_logging
from mkt.databases.plot import (
    generate_kinase_info_plot,
    plot_dynamic_range,
    plot_metrics_boxplot,
    plot_region_gap_violin,
    plot_ridgeline,
    plot_stacked_barchart,
    plot_venn_diagram,
)
from mkt.databases.plot_config import PlotDatasetConfig
from mkt.schema.io_utils import get_repo_root
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

app = typer.Typer()

# sections rendered when no --config is provided (preserves prior behavior)
_DEFAULT_SECTIONS = {
    "ridgeline",
    "stacked_barchart",
    "dynamic_range",
    "venn_diagram",
    "metrics_boxplot",
}


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
    """Generate plots for dataset data.

    Loads plot aesthetics and data source paths from a YAML config file when
    ``--config`` is provided, otherwise uses hardcoded defaults. Each figure is
    rendered only when its config section is present in the YAML, so a config may
    select any subset of the dataset plots (ridgeline, stacked_barchart,
    dynamic_range, venn_diagram, metrics_boxplot) and/or the KinaseInfo figures
    (upset_plot, region_gap_violin).
    """
    configure_logging(verbose=verbose)

    # load config and determine which figure sections were requested
    if config_path is not None:
        logger.info(f"Loading plot config from {config_path}")
        cfg = PlotDatasetConfig.from_yaml(config_path)
        present = set(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True))
    else:
        logger.info("Using default plot config")
        cfg = PlotDatasetConfig()
        present = set(_DEFAULT_SECTIONS)

    try:
        config.set_request_cache(os.path.join(get_repo_root(), "requests_cache.sqlite"))
    except Exception as e:
        logger.warning(f"Failed to set request cache, using current directory: {e}")
        config.set_request_cache(os.path.join(".", "requests_cache.sqlite"))

    repo_root = get_repo_root()

    # output directory uses the config name (stem) as a subdirectory
    config_name = Path(config_path).stem if config_path is not None else "default"
    output_dir = os.path.join(repo_root, cfg.output.subdir, config_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # --- dataset plots (require the Davis / PKIS2 processed CSVs) ---
    dataset_sections = {
        "ridgeline",
        "stacked_barchart",
        "dynamic_range",
        "venn_diagram",
    }
    if present & dataset_sections:
        # imported lazily: this module has a UniProt-querying import-time side
        # effect, so configs that only render KinaseInfo figures avoid it
        from mkt.databases.datasets.process import (
            generate_ridgeline_df,
            generate_stacked_barchart_df,
        )

        df_davis = pd.read_csv(os.path.join(repo_root, cfg.data_sources.davis_csv))
        df_pkis2 = pd.read_csv(os.path.join(repo_root, cfg.data_sources.pkis2_csv))

        if "ridgeline" in present:
            df_ridgeline = pd.concat(
                [
                    generate_ridgeline_df(df_davis, source="Davis"),
                    generate_ridgeline_df(df_pkis2, source="PKIS2"),
                ],
                axis=0,
            )
            plot_ridgeline(
                df_ridgeline,
                os.path.join(output_dir, f"{cfg.ridgeline.filename}.svg"),
                cfg=cfg.ridgeline,
                rc=cfg.matplotlib_rc,
                family_cfg=cfg.family_colors,
            )

        if "stacked_barchart" in present:
            df_stack = pd.concat(
                [
                    generate_stacked_barchart_df(df_davis, source="Davis"),
                    generate_stacked_barchart_df(df_pkis2, source="PKIS2"),
                ],
                axis=0,
            )
            plot_stacked_barchart(
                df_stack,
                os.path.join(output_dir, f"{cfg.stacked_barchart.filename}.svg"),
                cfg=cfg.stacked_barchart,
                rc=cfg.matplotlib_rc,
                family_cfg=cfg.family_colors,
            )

        if "dynamic_range" in present:
            plot_dynamic_range(
                df_davis,
                df_pkis2,
                os.path.join(output_dir, f"{cfg.dynamic_range.filename}.svg"),
                cfg=cfg.dynamic_range,
                rc=cfg.matplotlib_rc,
            )

        if "venn_diagram" in present:
            plot_venn_diagram(
                df_davis,
                os.path.join(output_dir, f"{cfg.venn_diagram.filename}_davis.svg"),
                "Davis",
                cfg=cfg.venn_diagram,
                rc=cfg.matplotlib_rc,
                color_cfg=cfg.col_kinase_colors,
            )
            plot_venn_diagram(
                df_pkis2,
                os.path.join(output_dir, f"{cfg.venn_diagram.filename}_pkis2.svg"),
                "PKIS2",
                cfg=cfg.venn_diagram,
                rc=cfg.matplotlib_rc,
                color_cfg=cfg.col_kinase_colors,
            )

    # --- metrics boxplot (requires the metrics CSV) ---
    if "metrics_boxplot" in present:
        metrics_path = os.path.join(repo_root, cfg.data_sources.metrics_csv)
        if os.path.exists(metrics_path):
            plot_metrics_boxplot(
                pd.read_csv(metrics_path),
                os.path.join(output_dir, f"{cfg.metrics_boxplot.filename}.svg"),
                cfg=cfg.metrics_boxplot,
                rc=cfg.matplotlib_rc,
                color_cfg=cfg.col_kinase_colors,
            )
        else:
            logger.warning(f"Metrics file not found: {metrics_path}")

    # --- KinaseInfo figures (require the shipped DICT_KINASE) ---
    if present & {"upset_plot", "region_gap_violin"}:
        from mkt.schema.io_utils import deserialize_kinase_dict

        dict_kinase = deserialize_kinase_dict(str_name="DICT_KINASE")
        if "upset_plot" in present:
            generate_kinase_info_plot(dict_kinase, output_dir, cfg=cfg.upset_plot)
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
