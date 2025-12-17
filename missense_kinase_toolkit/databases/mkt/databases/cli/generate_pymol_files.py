#!/usr/bin/env python3

import argparse

from mkt.databases.app.sequences import SequenceAlignment
from mkt.databases.app.structures import StructureVisualizer
from mkt.databases.colors import DICT_COLORS
from mkt.databases.pymol import PyMOLGenerator
from mkt.schema.io_utils import deserialize_kinase_dict

DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE")


def parse_args():
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
    return parser.parse_args()


def main():
    args = parse_args()

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
        bool_show=False,
    )

    pymol_generator = PyMOLGenerator(viz=viz)
    output_directory = f"./pymol_output/{gene}"
    pymol_generator.save_pymol_files(output_directory, gene)
