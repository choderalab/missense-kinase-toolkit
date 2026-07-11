#!/usr/bin/env python3
"""CLI to build and serialize the KLIFS hierarchical-conservation artifact.

Entry point (``generate_conservation_data``) that constructs a
:class:`mkt.databases.conservation.KLIFSHierarchicalConservation` engine over the
shipped ``DICT_KINASE`` KLIFS panel, packages its distances + dendrogram +
provenance as a :class:`mkt.schema.conservation_schema.KLIFSConservationData`, and
serializes it as ``mkt.schema`` package data so downstream renderers (e.g. the
``plot_dict_kinase`` figures) can deserialize it rather than recomputing the tree.
"""

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer
from mkt.databases.log_config import configure_logging
from mkt.schema.io_utils import serialize_conservation_data

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Generate the persisted KLIFS conservation-data artifact.",
    no_args_is_help=False,
)


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
    """Generate and serialize the KLIFS conservation-data artifact.

    Examples:
        # default panel (BLOSUM62 + UPGMA), shipped as package data
        generate_conservation_data

        # percent-identity metric, complete linkage, custom output directory
        generate_conservation_data --metric identity --linkage-method complete -o ./out
    """
    configure_logging(verbose=verbose)

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


if __name__ == "__main__":
    app()
