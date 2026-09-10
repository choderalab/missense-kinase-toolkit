#!/usr/bin/env python3
"""CLI for generating PyMOL visualization files for kinase structures.

Two modes: a single-gene one-off from flags (``--gene``/``--config-type``/...), or a batch run
from the ``pymol`` section of a shared study YAML (``--config``) that iterates its ``views``
list -- one output per view. Config-mode files go to ``<output.subdir>/<config-stem>/pymol/
<gene>/<config>/``; the flag one-off keeps the familiar ``images/pymol_output/<gene>/<config>/``.
"""

import logging
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
from mkt.databases.plot_config import PymolConfig, load_task_config
from mkt.databases.pymol import PyMOLGenerator
from mkt.schema.io_utils import get_repo_root

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Generate PyMOL files for kinase structure visualization.",
    no_args_is_help=True,
)

TASK_KEY = "pymol"
"""str: Top-level namespace this CLI reads from the shared study YAML."""


def _generate_one_view(
    gene: str,
    config_name: str,
    base_dir: Path,
    indices: Optional[str] = None,
    colors: Optional[str] = None,
    json_mutations: Optional[str] = None,
    transparency: float = 0.3,
    force_alphafold: bool = False,
) -> Path:
    """Generate the PyMOL files for one (gene, config) view under ``base_dir/<gene>/<config>``.

    Parameters
    ----------
    gene : str
        HGNC gene name of the kinase.
    config_name : str
        A :class:`StandardConfig` member name (e.g. ``"KLIFS_IMPORTANT"``).
    base_dir : Path
        Directory the view's ``<gene>/<config>`` output tree is created under.
    indices, colors : str | None
        KLIFS_CUSTOM: comma-separated UniProt positions and matching colors.
    json_mutations : str | None
        MUTATIONS_* configs: path to the mutations JSON.
    transparency : float
        KLIFS_CUSTOM cartoon transparency for the colored regions.
    force_alphafold : bool
        Render the AlphaFold DB structure even when a KinCoRe CIF is available.

    Returns
    -------
    Path
        The output directory the PyMOL files were written to.

    Raises
    ------
    ValueError
        If the config name is unknown or its required parameters are missing/invalid.
    """
    if config_name not in StandardConfig.__members__:
        raise ValueError(
            f"unknown config '{config_name}'; valid: {list(StandardConfig.__members__)}"
        )
    if config_name.startswith("MUTATIONS") and json_mutations is None:
        raise ValueError(f"json_mutations is required for {config_name}.")

    list_uniprot_idx: list[int] = []
    list_custom_color: list[str] = []
    if config_name == "KLIFS_CUSTOM":
        if indices is None or colors is None:
            raise ValueError("indices and colors are both required for KLIFS_CUSTOM.")
        try:
            list_uniprot_idx = [int(i.strip()) for i in indices.split(",") if i.strip()]
        except ValueError:
            raise ValueError("indices must be a comma-separated list of integers.")
        list_custom_color = [c.strip() for c in colors.split(",") if c.strip()]
        if len(list_uniprot_idx) != len(list_custom_color):
            raise ValueError(
                f"indices ({len(list_uniprot_idx)}) and colors "
                f"({len(list_custom_color)}) must have the same number of entries."
            )

    # e.g. "klifs" / "phosphosites"; str_attr is the config suffix
    str_final_subdir = config_name.lower().split("_")[0]
    str_attr = "_".join(config_name.lower().split("_")[1:])

    seq_align = SequenceAlignment(
        str_kinase=gene,
        # for the sequence viewer, not the PyMOL colors
        dict_color=DICT_COLORS["ALPHABET_PROJECT"]["DICT_COLORS"],
    )
    if config_name == "KLIFS_CUSTOM":
        validate_uniprot_indices(seq_align, list_uniprot_idx)

    config_kwargs: dict = {"prefer_alphafold": force_alphafold}
    if config_name.startswith("MUTATIONS"):
        config_kwargs["str_filepath_json"] = str(json_mutations)
    elif config_name == "KLIFS_CUSTOM":
        config_kwargs["list_uniprot_idx"] = list_uniprot_idx
        config_kwargs["list_custom_color"] = list_custom_color
        config_kwargs["highlight_cartoon_transparency"] = transparency

    viz = create_structure_visualizer(
        seq_align=seq_align,
        config_class=StandardConfig[config_name].value,
        config_kwargs=config_kwargs,
    )
    out_dir = base_dir / gene / str_final_subdir
    PyMOLGenerator(viz=viz, str_attr=str_attr).save_pymol_files(str(out_dir))
    return out_dir


@app.command()
def main(
    gene: Annotated[
        str,
        typer.Option(
            "--gene",
            "-g",
            help="Gene name of the kinase to visualize (single-gene flag mode).",
        ),
    ] = "ABL1",
    config_type: Annotated[
        StandardConfigChoice,
        typer.Option(
            "--config-type",
            "-c",
            help="Configuration type for structure highlighting (single-gene flag mode).",
            case_sensitive=False,
        ),
    ] = StandardConfigChoice.KLIFS_IMPORTANT,
    config_path: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            help="Shared study YAML; batch-generate every view under its 'pymol' section.",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            "-o",
            help="Base output directory. Default: <repo_root>/images/... ",
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
    force_alphafold: Annotated[
        bool,
        typer.Option(
            "--force-alphafold",
            "-f",
            help="Render the AlphaFold DB structure even when a KinCoRe active-state "
            "structure is available.",
        ),
    ] = False,
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
        # single-gene one-off (flag mode)
        generate_pymol_files --gene ABL1 --config-type KLIFS_IMPORTANT
        generate_pymol_files --gene ABL1 --config-type KLIFS_CUSTOM --indices 315,317 --colors red,blue

        # batch mode: every view under the 'pymol' section of a study YAML
        generate_pymol_files --config configs/paper_2026.yaml
    """
    configure_logging(verbose=verbose)

    # batch mode: iterate the study YAML's pymol.views
    if config_path is not None:
        pym_cfg = load_task_config(PymolConfig, config_path, TASK_KEY)
        config_name = Path(config_path).stem
        default_base = (
            Path(get_repo_root()) / pym_cfg.output.subdir / config_name / TASK_KEY
        )
        base = output_dir / TASK_KEY if output_dir is not None else default_base
        n_ok = 0
        for view in pym_cfg.views:
            view_base = Path(view.output_dir) if view.output_dir else base
            try:
                out = _generate_one_view(
                    view.gene,
                    view.config_type,
                    view_base,
                    indices=view.indices,
                    colors=view.colors,
                    json_mutations=view.json_mutations,
                    transparency=view.transparency,
                    force_alphafold=view.force_alphafold,
                )
                logger.info(f"generated {view.gene}/{view.config_type} in {out}")
                n_ok += 1
            except Exception as e:
                logger.error(f"skipping {view.gene}/{view.config_type}: {e}")
        typer.echo(f"PyMOL: generated {n_ok}/{len(pym_cfg.views)} view(s).")
        return

    # single-gene flag mode
    base = (output_dir or Path(get_repo_root())) / "images" / "pymol_output"
    try:
        out = _generate_one_view(
            gene,
            config_type.value,
            base,
            indices=indices,
            colors=colors,
            json_mutations=str(json_mutations) if json_mutations else None,
            transparency=transparency,
            force_alphafold=force_alphafold,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e))
    typer.echo(f"PyMOL files generated in: {out}")


if __name__ == "__main__":
    app()
