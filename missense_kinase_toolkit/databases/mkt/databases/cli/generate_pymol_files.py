#!/usr/bin/env python3

import argparse
import os

from mkt.databases.app.sequences import SequenceAlignment
from mkt.databases.app.structures import StructureVisualizer
from mkt.databases.colors import DICT_COLORS
from mkt.databases.pymol import generate_simple_pymol_files
from mkt.schema import io_utils

DICT_KINASE = io_utils.deserialize_kinase_dict(str_name="DICT_KINASE")


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

    output_directory = f"./pymol_output/{gene}"
    _, script_file = generate_simple_pymol_files(viz, output_directory, gene)

    f"""
    Files generated:
      PDB: {os.path.join(output_directory, f"{gene}_structure.pdb")}
      Script: {os.path.join(output_directory, f"{gene}_pymol_script.py")}

    To run in PyMOL:
      1. Open PyMOL
      2. Navigate to {output_directory}
      3. Run: run {gene}_pymol_script.py
      4. Save PNG: set ray_trace_mode, 3; png filename.png, ray=1, dpi=300
    """

    print("\n" + "=" * 60)
    print("MANUAL PYMOL INSTRUCTIONS:")
    print("=" * 60)
    print("1. Open PyMOL GUI or command line")
    print("2. Change to the output directory:")
    print(f"   cd {os.path.abspath(output_directory)}")
    print("3. Run the script:")
    print(f"   run {os.path.basename(script_file)}")
    print("4. To save as high-res PNG:")
    print("   set ray_trace_mode, <mode>")
    print("   png your_filename.png, ray=1, dpi=300")
    print("=" * 60)
