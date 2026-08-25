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
            help="Run only these enrichment step(s); repeatable. Mutually exclusive "
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
            "rebuilt and spliced into the existing archive (reports skipped). Omit to "
            "regenerate the full kinome.",
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
        )
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
