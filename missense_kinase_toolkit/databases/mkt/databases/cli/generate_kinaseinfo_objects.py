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
from mkt.databases.log_config import configure_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Generate KinaseInfo objects from API or scraper.",
    no_args_is_help=False,
)


@app.command()
def main(
    only: Annotated[
        Optional[list[str]],
        typer.Option(
            "--only",
            help="Rebuild only these component(s); repeatable. A base-build source "
            "(hgnc/uniprot/kinhub/klifs/pfam/kincore) does a partial rebuild on the "
            "existing dict; an enrichment step name runs that step. Mutually exclusive "
            "with --skip.",
        ),
    ] = None,
    skip: Annotated[
        Optional[list[str]],
        typer.Option(
            "--skip",
            help="Skip these enrichment step(s); repeatable. All other default-on "
            "steps run.",
        ),
    ] = None,
    kinase: Annotated[
        Optional[list[str]],
        typer.Option(
            "--kinase",
            help="HGNC name(s) to update one-off; repeatable. Only these entries are "
            "rebuilt and spliced into the existing archive. Omit to regenerate the full "
            "kinome.",
        ),
    ] = None,
    path_objects: Annotated[
        Optional[str],
        typer.Option(
            "--pathObjects",
            help="Where to save KinaseInfo objects, relative to repo root; if not a "
            "Github repo, relative to the current directory.",
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
    no_figs: Annotated[
        bool,
        typer.Option(
            "--no-figs",
            help="Skip regenerating the report figures after the build. By default "
            "figures are refreshed on any dict regeneration, into a datetime-stamped "
            "subdirectory keyed by the archive's modified time.",
        ),
    ] = False,
    figs_only: Annotated[
        bool,
        typer.Option(
            "--figs-only",
            help="Only regenerate the report figures from the existing archive (no "
            "rebuild), reusing the subdirectory keyed by its modified time. Mutually "
            "exclusive with --only/--skip/--kinase.",
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
            bool_figs=not no_figs,
            figs_only=figs_only,
        )
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
