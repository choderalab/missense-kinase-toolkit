#!/usr/bin/env python
"""
Script to align multiple CIF files to a reference PDB structure using PyMOL.
The aligned structures are saved as new CIF files with updated coordinates.
"""

import os
import shutil

from io_utils import (
    create_tar_without_metadata,
    get_parser,
    untar_files_in_memory,
)
from pymol import cmd, finish_launching
from tqdm import tqdm


def align_structures(
    reference_pdb,
    input_tar,
    output_tar,
    selection="all",
    method="align",
):
    """
    Align multiple CIF files to a reference PDB structure and save with new coordinates

    Parameters:
    -----------
    reference_pdb : str
        Path to the reference PDB file
    cif_dir : str
        Directory containing input CIF files
    output_dir : str
        Directory to save aligned CIF files
    selection : str
        PyMOL selection string for alignment (default: "all")
    method : str
        PyMOL alignment method to use (options: "align", "super", "cealign", default: "align")
    """

    # Start PyMOL in quiet mode
    finish_launching(["pymol", "-qc"])

    # Load reference structure
    cmd.load(reference_pdb, "reference")

    _, dict_cif = untar_files_in_memory(input_tar)
    dict_cif = {k: v for k, v in dict_cif.items() if k.endswith(".cif")}

    if len(dict_cif) == 0:
        print(f"No CIF files found in {input_tar}.")
        return

    print(f"Found {len(dict_cif)} CIF files to process...")

    temp_dir = os.path.join(os.path.dirname(input_tar), "aligned_cif_temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Process each CIF file
    for str_filename, str_cif in tqdm(dict_cif.items(), desc="Aligning structures..."):
        str_filename_update = str_filename.split("/")[1].replace(".cif", ".pdb")
        str_filepath = os.path.join(temp_dir, str_filename_update)

        # Clear everything except reference
        cmd.delete("not reference")

        # Load target structure
        mobile_name = "mobile"
        cmd.load_raw(str_cif, format="cif", object=mobile_name)

        # Perform alignment
        if method == "align":
            cmd.align(f"{mobile_name} and {selection}", f"reference and {selection}")
        elif method == "super":
            cmd.super(f"{mobile_name} and {selection}", f"reference and {selection}")
        elif method == "cealign":
            cmd.cealign(f"reference and {selection}", f"{mobile_name} and {selection}")
        else:
            print(f"Unknown alignment method: {method}. Using default 'align'")
            cmd.align(f"{mobile_name} and {selection}", f"reference and {selection}")

        # Save aligned structure
        cmd.save(str_filepath, mobile_name)

    cmd.quit()

    create_tar_without_metadata(temp_dir, output_tar)
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    print(f"Aligned structures have been saved to {output_tar}...")


def main():
    args = get_parser().parse_args()

    align_structures(
        args.referencePDB,
        args.inpuTar,
        args.outputTar,
        args.molecSelection,
        args.methodAlign,
    )


if __name__ == "__main__":
    main()
