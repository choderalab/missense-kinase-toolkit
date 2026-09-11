#!/usr/bin/env python3
"""CLI to build the processed kinase dataset CSVs and render their figures.

Entry point (``generate_dataset_csv_files``) that builds and writes the processed dataset CSV
files (Davis, PKIS2) and, mirroring ``generate_kinaseinfo_objects``, renders the dataset figures
from them: ``--figs-only`` re-renders figures without rebuilding the CSVs, ``--no-figs`` builds
the CSVs alone. Figure aesthetics come from the ``dataset`` section of a shared study YAML
(``--config``); each figure renders only when its section is present (all render when no config
is given). Figures write to ``<output.subdir>/<config-stem>/dataset/``.
"""

import logging
import os
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from mkt.databases import config
from mkt.databases.log_config import configure_logging
from mkt.databases.plot_config import DatasetFiguresConfig, load_task_config
from mkt.schema.io_utils import get_repo_root
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Build the processed kinase dataset CSVs and render their figures.",
    no_args_is_help=False,
)

TASK_KEY = "dataset"
"""str: Top-level namespace this CLI reads from the shared study YAML."""

# figure sections, rendered in this order; all render when no --config is given
_FIGURE_SECTIONS = [
    "dynamic_range",
    "ridgeline",
    "stacked_barchart",
    "metrics_boxplot",
    "venn_diagram",
]


def _set_request_cache() -> None:
    """Point requests-cache at the repo-root SQLite (fallback to cwd)."""
    try:
        config.set_request_cache(os.path.join(get_repo_root(), "requests_cache.sqlite"))
    except Exception as e:
        logger.warning(f"Failed to set request cache, using current directory: {e}")
        config.set_request_cache(os.path.join(".", "requests_cache.sqlite"))


def _build_datasets() -> None:
    """Build and save the processed dataset CSVs (Davis, PKIS2)."""
    from mkt.databases.datasets.davis import DavisDataset
    from mkt.databases.datasets.pkis2 import PKIS2Dataset

    _set_request_cache()
    DavisDataset(bool_save=True)
    PKIS2Dataset(bool_save=True)


def _run_figures(config_path: Optional[Path]) -> None:
    """Render the dataset figures whose config section is present (all when no config).

    ``mkt.databases.datasets.process`` (whose import triggers a network-backed build) is
    imported lazily inside only the branches that need it, so figures like dynamic_range /
    venn / metrics render without pulling it in.
    """
    from mkt.databases.plot import (
        plot_dynamic_range,
        plot_metrics_boxplot,
        plot_ridgeline,
        plot_stacked_barchart,
        plot_venn_diagram,
    )

    cfg = load_task_config(DatasetFiguresConfig, config_path, TASK_KEY)
    if config_path is not None:
        section = OmegaConf.load(config_path).get(TASK_KEY, {}) or {}
        sections = set(section.keys())
    else:
        sections = set(_FIGURE_SECTIONS)

    _set_request_cache()
    repo_root = get_repo_root()
    df_davis = pd.read_csv(os.path.join(repo_root, cfg.data_sources.davis_csv))
    df_pkis2 = pd.read_csv(os.path.join(repo_root, cfg.data_sources.pkis2_csv))

    config_name = Path(config_path).stem if config_path is not None else "default"
    output_dir = os.path.join(repo_root, cfg.output.subdir, config_name, TASK_KEY)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Rendering dataset figures into {output_dir}")

    if "dynamic_range" in sections:
        logger.info("Rendering dynamic_range")
        plot_dynamic_range(
            df_davis,
            df_pkis2,
            os.path.join(output_dir, f"{cfg.dynamic_range.filename}.svg"),
            cfg=cfg.dynamic_range,
            rc=cfg.matplotlib_rc,
        )

    if "ridgeline" in sections:
        logger.info("Rendering ridgeline")
        from mkt.databases.datasets.process import generate_ridgeline_df

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

    if "stacked_barchart" in sections:
        logger.info("Rendering stacked_barchart")
        from mkt.databases.datasets.process import generate_stacked_barchart_df

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

    if "metrics_boxplot" in sections:
        metrics_path = os.path.join(repo_root, cfg.data_sources.metrics_csv)
        if os.path.exists(metrics_path):
            logger.info("Rendering metrics_boxplot")
            plot_metrics_boxplot(
                pd.read_csv(metrics_path),
                os.path.join(output_dir, f"{cfg.metrics_boxplot.filename}.svg"),
                cfg=cfg.metrics_boxplot,
                rc=cfg.matplotlib_rc,
                color_cfg=cfg.col_kinase_colors,
            )
        else:
            logger.warning(f"Metrics file not found: {metrics_path}")

    if "venn_diagram" in sections:
        logger.info("Rendering venn_diagram")
        for df, name in ((df_davis, "davis"), (df_pkis2, "pkis2")):
            plot_venn_diagram(
                df,
                os.path.join(output_dir, f"{cfg.venn_diagram.filename}_{name}.svg"),
                name.capitalize() if name == "davis" else "PKIS2",
                cfg=cfg.venn_diagram,
                rc=cfg.matplotlib_rc,
                color_cfg=cfg.col_kinase_colors,
            )


@app.command()
def main(
    config_path: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            help="Shared study YAML; the 'dataset' section supplies figure aesthetics + data "
            "source paths. Each figure renders only when its section is present.",
        ),
    ] = None,
    figs_only: Annotated[
        bool,
        typer.Option(
            "--figs-only",
            help="Only (re)render the figures from the existing CSVs; skip the rebuild.",
        ),
    ] = False,
    no_figs: Annotated[
        bool,
        typer.Option("--no-figs", help="Build the dataset CSVs only; skip figures."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose (DEBUG) logging."),
    ] = False,
) -> None:
    """Build the processed dataset CSVs and render their figures.

    Examples:
        # build the Davis/PKIS2 CSVs + render all dataset figures with defaults
        generate_dataset_csv_files

        # rebuild + render the figures configured under the 'dataset' section
        generate_dataset_csv_files --config configs/paper_2026.yaml

        # re-render figures only, without rebuilding the CSVs
        generate_dataset_csv_files --config configs/paper_2026.yaml --figs-only
    """
    configure_logging(verbose=verbose)

    if figs_only:
        _run_figures(config_path)
        return

    _build_datasets()
    if not no_figs:
        _run_figures(config_path)


if __name__ == "__main__":
    app()
