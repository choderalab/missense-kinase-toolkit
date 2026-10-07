#!/usr/bin/env python
"""CLI to build :class:`KinaseInfo` objects from APIs and serialize them to tar.gz.

Entry point (``generate_kinaseinfo_objects``) that drives the compositional build
pipeline (:mod:`mkt.databases.generator.pipeline`). Supports a full kinome regeneration
(default), a per-step run via ``--only``/``--skip`` over the enrichment registry, and a
one-off per-entry update via ``--kinase`` that rebuilds and splices in targeted entries
without a full rebuild. ``--only``/``--skip`` and ``--kinase`` compose freely.
"""

import logging
from typing import Annotated, Optional

import typer
from mkt.databases.generator import pipeline
from mkt.databases.generator import steps as build_steps
from mkt.databases.plot_config import ArgumentError
from mkt.schema.log_config import configure_logging

logger = logging.getLogger(__name__)

STR_COMPONENTS = ", ".join(build_steps.return_component_names())
"""str: Every component name, in run order, for the --only help."""

STR_STEPS = ", ".join(build_steps.return_component_names("step"))
"""str: Enrichment-step names, in run order, for the --skip help."""

STR_EPILOG = "Components, in run order:\n\n" + "\n\n".join(
    f"{name}: {component.help}" for name, component in build_steps.COMPONENTS.items()
)
"""str: Help epilog describing each component."""

app = typer.Typer(
    help="Generate KinaseInfo objects from API or scraper.",
    no_args_is_help=False,
)


@app.command(epilog=STR_EPILOG)
def main(
    only: Annotated[
        Optional[list[str]],
        typer.Option(
            "--only",
            help="Rebuild only these component(s) on the existing archive, plus every "
            f"step downstream of them (other fields are kept); repeatable. One of: "
            f"{STR_COMPONENTS}. Mutually exclusive with --skip.",
        ),
    ] = None,
    skip: Annotated[
        Optional[list[str]],
        typer.Option(
            "--skip",
            help="Skip these step(s); repeatable. Steps downstream of a skipped step still "
            f"run, with a warning listing them. One of: {STR_STEPS}.",
        ),
    ] = None,
    kinase: Annotated[
        Optional[list[str]],
        typer.Option(
            "--kinase",
            help="HGNC name(s) to update one-off; repeatable. Only these entries are "
            "rebuilt and spliced into the existing archive (with --only, only the named "
            "components run on them). Omit to regenerate the full kinome.",
        ),
    ] = None,
    path_objects: Annotated[
        Optional[str],
        typer.Option(
            "--pathObjects",
            help="Objects directory, relative to repo root (or the current directory "
            "outside a Github repo); KinaseInfo.tar.gz is written beside it. Entries "
            "are staged in a temporary directory.",
        ),
    ] = None,
    path_reports: Annotated[
        Optional[str],
        typer.Option(
            "--pathReports",
            help="Where to save reports, relative to repo root; if not a Github repo, "
            "relative to the current directory.",
        ),
    ] = None,
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config",
            help="Shared study YAML supplying report aesthetics (the 'kinaseinfo' section). "
            "When given, reports go to <output.subdir>/<config-stem>/kinaseinfo/; otherwise "
            "to dict_kinase/<generated_at>/ from the archive manifest.",
        ),
    ] = None,
    data: Annotated[
        Optional[bool],
        typer.Option(
            "--data/--no-data",
            help="Build or update the KinaseInfo archive; --no-data draws figures from the "
            "existing archive. Defaults to the config's kinaseinfo.data (true if unset).",
            show_default=False,
        ),
    ] = None,
    figs: Annotated[
        bool,
        typer.Option(
            "--figs/--no-figs",
            help="Draw the report figures.",
        ),
    ] = True,
    recompute: Annotated[
        bool,
        typer.Option(
            "--recompute",
            help="Recompute structure-derived properties (AlphaFold slice, SASA, "
            "superposition) even when already stored.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose (DEBUG) logging."),
    ] = False,
):
    configure_logging(verbose=verbose)

    try:
        pipeline.run(
            only=only,
            skip=skip,
            list_kinase=kinase,
            path_objects=path_objects,
            path_reports=path_reports,
            bool_data=data,
            bool_figs=figs,
            force=recompute,
            config_path=config_path,
        )
    except ArgumentError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
