#!/usr/bin/env python3

import argparse
from os import path

from mkt.databases.app.utils import generate_sequence_and_structure_viewers
from mkt.databases.colors import DICT_COLORS
from mkt.databases.log_config import add_logging_flags, configure_logging
from mkt.databases.pymol import PyMOLGenerator
from mkt.schema.io_utils import get_repo_root


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate PyMOL files for kinase structure visualization."
    )

    parser.add_argument(
        "--gene",
        type=str,
        required=False,
        default="ABL1",
        help="Gene name of the kinase to visualize (default: ABL1)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Output directory for PyMOL files (default if not used: <repo_root>/images/pymol_output/<gene>)",
    )

    parser = add_logging_flags(parser)

    return parser


def main():
    args = get_parser().parse_args()

    if args.verbose == "DEBUG":
        configure_logging(True)
    else:
        configure_logging(False)

    gene = args.gene

    _, obj_viz = generate_sequence_and_structure_viewers(
        str_kinase=gene,
        dict_colors=DICT_COLORS["ALPHABET_PROJECT"]["DICT_COLORS"],
        str_attr="KLIFS",
    )

    pymol_generator = PyMOLGenerator(viz=obj_viz)
    if args.output_dir:
        output_directory = args.output_dir
    else:
        output_directory = path.join(get_repo_root(), "images", "pymol_output", gene)
    pymol_generator.save_pymol_files(output_directory)


if __name__ == "__main__":
    main()
