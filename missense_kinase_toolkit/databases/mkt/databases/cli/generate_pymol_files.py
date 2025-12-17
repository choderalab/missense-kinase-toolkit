#!/usr/bin/env python3

import argparse
from os import path

from mkt.databases.app.sequences import SequenceAlignment
from mkt.databases.app.structures import StructureVisualizer
from mkt.databases.colors import DICT_COLORS
from mkt.databases.log_config import add_logging_flags, configure_logging
from mkt.databases.pymol import PyMOLGenerator
from mkt.schema.io_utils import deserialize_kinase_dict, get_repo_root

DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE")


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

    obj_temp = DICT_KINASE[gene]

    obj_alignment = SequenceAlignment(
        obj_temp,
        DICT_COLORS["ALPHABET_PROJECT"]["DICT_COLORS"],
    )

    viz = StructureVisualizer(
        obj_temp,
        obj_alignment.dict_align,
        str_attr="KLIFS",
    )

    pymol_generator = PyMOLGenerator(viz=viz)

    if args.output_dir:
        output_directory = args.output_dir
    else:
        output_directory = path.join(get_repo_root(), "images", "pymol_output", gene)

    pymol_generator.save_pymol_files(output_directory)


if __name__ == "__main__":
    main()
