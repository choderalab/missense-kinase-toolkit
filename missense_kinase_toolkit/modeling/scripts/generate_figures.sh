#!/usr/bin/env bash -l
# Script to generate publication-quality figures for kinase structures using PyMOL
# Usage: ./generate_figures.sh <path_to_script_directory> <output_directory>

if [ "$#" -ne 2 ]; then
    echo "You must enter exactly 2 command line arguments: <path_to_script_directory> <output_directory>"
    exit
fi

PATH_TO_SCRIPT_DIR="$1"
OUTPUT_DIR="$2"

# hardcoded variables
FILE_SCRIPT="${PATH_TO_SCRIPT_DIR}/render_publication.py"
STR_IDX_ABL1_KLIFS="13-152"
STR_IDX_POMK_KLIFS="5-151"

if [ ! -f "${FILE_SCRIPT}" ]; then
    echo "File ${FILE_SCRIPT} does not exist. Exiting."
    exit
fi

if [ ! -d "${OUTPUT_DIR}/pse/" ]; then
    echo "Directory ${OUTPUT_DIR}/pse/ does not exist. Exiting."
    exit
fi

mkdir -p "${OUTPUT_DIR}/images/"

for kinase in "abl1" "pomk"; do
    # UniProt sequence (default color scheme)
    FILE_PSE_UNIPROT=${OUTPUT_DIR}/pse/${kinase}_uniprot.pse
    if [ -f ${FILE_PSE_UNIPROT} ]; then
        echo "Generating figure for ${kinase} UniProt structure..."
        pymol ${OUTPUT_DIR}/pse/${kinase}_uniprot.pse \
            -d "run ${FILE_SCRIPT}; \
                generate_pymol_figure ${OUTPUT_DIR}/images/${kinase}_uniprot.png; \
                quit"
    else
        echo "File ${FILE_PSE_UNIPROT} does not exist. \
            Skipping UniProt structure for ${kinase}."
        continue
    fi

    # Pfam kinase domains (cyan color scheme)
    FILE_PSE_PFAM=${OUTPUT_DIR}/pse/${kinase}_pfam.pse
    if [ -f ${FILE_PSE_PFAM} ]; then
        echo "Generating figure for ${kinase} Pfam kinase domain..."
        pymol ${OUTPUT_DIR}/pse/${kinase}_pfam.pse \
            -d "run ${FILE_SCRIPT}; \
                generate_pymol_figure ${OUTPUT_DIR}/images/${kinase}_pfam.png, \
                    cyan; \
                quit"
    else
        echo "File ${FILE_PSE_PFAM} does not exist. \
            Skipping Pfam kinase domain for ${kinase}."
        continue
    fi

    # KinCore kinase domains (magenta color scheme)
    FILE_PSE_KINCORE=${OUTPUT_DIR}/pse/${kinase}_kincore.pse
    if [ -f ${FILE_PSE_KINCORE} ]; then

        echo "Generating figure for ${kinase} KinCore kinase domain..."
        pymol ${OUTPUT_DIR}/pse/${kinase}_kincore.pse \
            -d "run ${FILE_SCRIPT}; \
                generate_pymol_figure ${OUTPUT_DIR}/images/${kinase}_kincore.png, \
                    magenta; \
                quit"

        # KinCore kinase domains with KLIFS pocket highlighted (orange color scheme)
        if [ "${kinase}" == "abl1" ]; then
            STR_IDX_KLIFS=${STR_IDX_ABL1_KLIFS}
        else
            STR_IDX_KLIFS=${STR_IDX_POMK_KLIFS}
        fi
        echo "Generating figure for ${kinase} KinCore kinase domain with KLIFS pocket highlighted..."
        pymol ${OUTPUT_DIR}/pse/${kinase}_kincore.pse \
            -d "run ${FILE_SCRIPT}; \
                generate_pymol_figure ${OUTPUT_DIR}/images/${kinase}_klifs.png, \
                    str_idx=${STR_IDX_KLIFS}, \
                    highlight_color=orange; \
                quit"
    else
        echo "File ${FILE_PSE_KINCORE} does not exist. \
            Skipping KinCore kinase domain and KLIFS region for ${kinase}."
        continue
    fi
done

# RUN IN PYTHON TO GET KLIFS INDICES IN KINCORE NUMBERING SCHEME (mkt.schema needed)
# from mkt.schema.io_utils import deserialize_kinase_dict
# DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE")
# str_kinase = "ABL1"/"POMK"
# obj_temp = DICT_KINASE[str_kinase]
# idx_start_kincore = obj_temp.kincore.fasta.start
# idx_start_klifs = min(list(obj_temp.KLIFS2UniProtIdx.values()))
# idx_end_klifs = max(list(obj_temp.KLIFS2UniProtIdx.values()))
# idx_start_klifs_kincore = idx_start_klifs - idx_start_kincore + 1
# idx_end_klifs_kincore = idx_start_klifs_kincore + (idx_end_klifs - idx_start_klifs)
# print(f"{idx_start_klifs_kincore}-{idx_end_klifs_kincore}")
