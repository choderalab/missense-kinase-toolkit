#!/usr/bin/env python3

import logging
import os
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from mkt.databases import config
from mkt.databases.datasets.process import (
    generate_ridgeline_df,
    generate_stacked_barchart_df,
)
from mkt.databases.log_config import configure_logging
from mkt.databases.plot import (
    plot_dynamic_range,
    plot_metrics_boxplot,
    plot_ridgeline,
    plot_stacked_barchart,
    plot_venn_diagram,
)
from mkt.databases.plot_config import PlotDatasetConfig
from mkt.schema.io_utils import get_repo_root

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def main(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to YAML configuration file. If not provided, uses defaults.",
    ),
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose (DEBUG) logging.",
        ),
    ] = False,
) -> None:
    """Generate plots for dataset data.

    Loads plot aesthetics and data source paths from a YAML config file
    when --config is provided, otherwise uses hardcoded defaults.
    """
    configure_logging(verbose=verbose)

    # load config
    if config_path is not None:
        logger.info(f"Loading plot config from {config_path}")
        cfg = PlotDatasetConfig.from_yaml(config_path)
    else:
        logger.info("Using default plot config")
        cfg = PlotDatasetConfig()

    try:
        config.set_request_cache(os.path.join(get_repo_root(), "requests_cache.sqlite"))
    except Exception as e:
        logger.warning(f"Failed to set request cache, using current directory: {e}")
        config.set_request_cache(os.path.join(".", "requests_cache.sqlite"))

    # load processed data
    repo_root = get_repo_root()
    df_davis = pd.read_csv(os.path.join(repo_root, cfg.data_sources.davis_csv))
    df_pkis2 = pd.read_csv(os.path.join(repo_root, cfg.data_sources.pkis2_csv))

    # generate ridgeline data
    df_davis_ridgeline = generate_ridgeline_df(df_davis, source="Davis")
    df_pkis2_ridgeline = generate_ridgeline_df(df_pkis2, source="PKIS2")
    df_ridgeline = pd.concat([df_davis_ridgeline, df_pkis2_ridgeline], axis=0)

    # generate stacked barchart data
    df_davis_stack = generate_stacked_barchart_df(df_davis, source="Davis")
    df_pkis2_stack = generate_stacked_barchart_df(df_pkis2, source="PKIS2")
    df_stack = pd.concat([df_davis_stack, df_pkis2_stack], axis=0)

    # create output directory using config name as subdirectory
    config_name = Path(config_path).stem if config_path is not None else "default"
    output_dir = os.path.join(repo_root, cfg.output.subdir, config_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # generate and save plots
    ridgeline_path = os.path.join(output_dir, f"{cfg.ridgeline.filename}.svg")
    plot_ridgeline(
        df_ridgeline,
        ridgeline_path,
        cfg=cfg.ridgeline,
        rc=cfg.matplotlib_rc,
        family_cfg=cfg.family_colors,
    )

    stacked_path = os.path.join(output_dir, f"{cfg.stacked_barchart.filename}.svg")
    plot_stacked_barchart(
        df_stack,
        stacked_path,
        cfg=cfg.stacked_barchart,
        rc=cfg.matplotlib_rc,
        family_cfg=cfg.family_colors,
    )

    # generate dynamic range comparison plot
    dynamic_range_path = os.path.join(output_dir, f"{cfg.dynamic_range.filename}.svg")
    plot_dynamic_range(
        df_davis,
        df_pkis2,
        dynamic_range_path,
        cfg=cfg.dynamic_range,
        rc=cfg.matplotlib_rc,
    )

    # load metrics data and generate boxplot
    metrics_path = os.path.join(repo_root, cfg.data_sources.metrics_csv)
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        boxplot_path = os.path.join(output_dir, f"{cfg.metrics_boxplot.filename}.svg")
        plot_metrics_boxplot(
            df_metrics,
            boxplot_path,
            cfg=cfg.metrics_boxplot,
            rc=cfg.matplotlib_rc,
            color_cfg=cfg.col_kinase_colors,
        )
    else:
        logger.warning(f"Metrics file not found: {metrics_path}")

    # generate Venn diagrams for Davis and PKIS2
    venn_davis_path = os.path.join(output_dir, f"{cfg.venn_diagram.filename}_davis.svg")
    plot_venn_diagram(
        df_davis,
        venn_davis_path,
        "Davis",
        cfg=cfg.venn_diagram,
        rc=cfg.matplotlib_rc,
        color_cfg=cfg.col_kinase_colors,
    )

    venn_pkis2_path = os.path.join(output_dir, f"{cfg.venn_diagram.filename}_pkis2.svg")
    plot_venn_diagram(
        df_pkis2,
        venn_pkis2_path,
        "PKIS2",
        cfg=cfg.venn_diagram,
        rc=cfg.matplotlib_rc,
        color_cfg=cfg.col_kinase_colors,
    )

    logger.info("All plots generated successfully!")


if __name__ == "__main__":
    app()
