#!/usr/bin/env python3
"""CLI for generating PyMOL visualization files for kinase structures."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from mkt.databases.app.schema import StandardConfig, StandardConfigChoice
from mkt.databases.app.sequences import SequenceAlignment
from mkt.databases.app.utils import (
    create_structure_visualizer,
    validate_uniprot_indices,
)
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
    indices: Annotated[
        Optional[str],
        typer.Option(
            "--indices",
            "-i",
            help="Comma-separated 1-indexed full-length UniProt positions to highlight "
            "as sticks (required for KLIFS_CUSTOM config).",
        ),
    ] = None,
    colors: Annotated[
        Optional[str],
        typer.Option(
            "--colors",
            help="Comma-separated colors (names or hex) matching --indices "
            "(required for KLIFS_CUSTOM config).",
        ),
    ] = None,
    transparency: Annotated[
        float,
        typer.Option(
            "--transparency",
            "-t",
            min=0.0,
            max=1.0,
            help="Cartoon transparency for the colored KLIFS regions in KLIFS_CUSTOM "
            "config (0 = opaque, 1 = invisible).",
        ),
    ] = 0.3,
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

        # Generate KLIFS regions (semi-transparent cartoon) with custom stick residues
        generate_pymol_files --gene ABL1 --config KLIFS_CUSTOM --indices 315,317 --colors red,blue
    """
    configure_logging(verbose=verbose)

    # validate that mutations file is provided for MUTATIONS_* configs
    config_name = config_type.value
    if config_name.startswith("MUTATIONS") and json_mutations is None:
        raise typer.BadParameter(
            f"--json-mutations is required when using {config_name} config.",
            param_hint="--json-mutations",
        )

    # parse and validate custom indices/colors for KLIFS_CUSTOM config
    list_uniprot_idx: list[int] = []
    list_custom_color: list[str] = []
    if config_name == "KLIFS_CUSTOM":
        if indices is None or colors is None:
            raise typer.BadParameter(
                "--indices and --colors are both required when using KLIFS_CUSTOM config.",
                param_hint="--indices / --colors",
            )
        try:
            list_uniprot_idx = [int(i.strip()) for i in indices.split(",") if i.strip()]
        except ValueError:
            raise typer.BadParameter(
                "--indices must be a comma-separated list of integers.",
                param_hint="--indices",
            )
        list_custom_color = [c.strip() for c in colors.split(",") if c.strip()]
        if len(list_uniprot_idx) != len(list_custom_color):
            raise typer.BadParameter(
                f"--indices ({len(list_uniprot_idx)}) and --colors "
                f"({len(list_custom_color)}) must have the same number of entries.",
                param_hint="--indices / --colors",
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

    # validate custom indices fall within the protein for KLIFS_CUSTOM config
    if config_name == "KLIFS_CUSTOM":
        try:
            validate_uniprot_indices(seq_align, list_uniprot_idx)
        except ValueError as e:
            raise typer.BadParameter(str(e), param_hint="--indices")

    # Get the config class from StandardConfig enum
    config_class = StandardConfig[config_name].value

    # prepare config kwargs
    config_kwargs: dict = {}
    if config_name.startswith("MUTATIONS"):
        config_kwargs["str_filepath_json"] = str(json_mutations)
    elif config_name == "KLIFS_CUSTOM":
        config_kwargs["list_uniprot_idx"] = list_uniprot_idx
        config_kwargs["list_custom_color"] = list_custom_color
        config_kwargs["highlight_cartoon_transparency"] = transparency

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
