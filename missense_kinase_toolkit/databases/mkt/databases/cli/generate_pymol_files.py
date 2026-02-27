#!/usr/bin/env python3
"""CLI for generating PyMOL visualization files for kinase structures."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from mkt.databases.app.schema import StandardConfig, StandardConfigChoice
from mkt.databases.app.sequences import SequenceAlignment
from mkt.databases.app.utils import create_structure_visualizer
from mkt.databases.colors import DICT_COLORS
from mkt.databases.log_config import configure_logging
from mkt.databases.pymol import PyMOLGenerator
from mkt.schema.io_utils import get_repo_root

app = typer.Typer(
    help="Generate PyMOL files for kinase structure visualization.",
    no_args_is_help=True,
)


@app.command()
def main(
    gene: Annotated[
        str,
        typer.Option(
            "--gene",
            "-g",
            help="Gene name of the kinase to visualize.",
        ),
    ] = "ABL1",
    config_type: Annotated[
        StandardConfigChoice,
        typer.Option(
            "--config",
            "-c",
            help="Configuration type for structure highlighting.",
            case_sensitive=False,
        ),
    ] = StandardConfigChoice.KLIFS_IMPORTANT,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for PyMOL files. Default: <repo_root>/images/pymol_output/<gene>/<config>",
        ),
    ] = None,
    json_mutations: Annotated[
        Optional[Path],
        typer.Option(
            "--json-mutations",
            "-j",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="JSON file of mutations to highlight (required for MUTATIONS_* configs).",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose (DEBUG) logging.",
        ),
    ] = False,
) -> None:
    """Generate PyMOL visualization files for kinase structures.

    Examples:
        # Generate KLIFS pocket visualization for ABL1
        generate_pymol_files --gene ABL1 --config KLIFS_IMPORTANT

        # Generate phosphosite visualization
        generate_pymol_files --gene EGFR --config PHOSPHOSITES

        # Generate mutation visualization (requires JSON file)
        generate_pymol_files --gene ABL1 --config MUTATIONS_KLIFS --json-mutations mutations.json

        # Generate group-averaged mutations
        generate_pymol_files --gene ABL1 --config MUTATIONS_GROUP --json-mutations mutations.json
    """
    configure_logging(verbose=verbose)

    # validate that mutations file is provided for MUTATIONS_* configs
    config_name = config_type.value
    if config_name.startswith("MUTATIONS") and json_mutations is None:
        raise typer.BadParameter(
            f"--json-mutations is required when using {config_name} config.",
            param_hint="--json-mutations",
        )

    # e.g., "klifs" or "phosphosites"
    str_final_subdir = config_name.lower().split("_")[0]
    str_attr = "_".join(config_name.lower().split("_")[1:])

    # create sequence alignment
    seq_align = SequenceAlignment(
        str_kinase=gene,
        # this is for sequence viewer, not PyMOL colors
        dict_color=DICT_COLORS["ALPHABET_PROJECT"]["DICT_COLORS"],
    )

    # Get the config class from StandardConfig enum
    config_class = StandardConfig[config_name].value

    # prepare config kwargs
    config_kwargs: dict = {}
    if config_name.startswith("MUTATIONS"):
        config_kwargs["str_filepath_json"] = str(json_mutations)

    # create structure visualizer using the config
    viz = create_structure_visualizer(
        seq_align=seq_align,
        config_class=config_class,
        config_kwargs=config_kwargs,
    )

    # generate PyMOL files
    pymol_generator = PyMOLGenerator(viz=viz, str_attr=str_attr)

    str_subdirs = Path("images") / "pymol_output" / gene / str_final_subdir
    if output_dir:
        out_dir = output_dir / str_subdirs
    else:
        out_dir = Path(get_repo_root()) / str_subdirs

    pymol_generator.save_pymol_files(str(out_dir))

    typer.echo(f"PyMOL files generated in: {out_dir}")


if __name__ == "__main__":
    app()
