#!/usr/bin/env python3

import argparse
import json
from os import path

from mkt.databases.app.utils import generate_sequence_and_structure_viewers
from mkt.databases.colors import DICT_COLORS
from mkt.databases.log_config import add_logging_flags, configure_logging
from mkt.databases.pymol import PyMOLGenerator
from mkt.schema.io_utils import get_repo_root

LIST_ATTR_OPTIONS = ["KLIFS", "Phosphosites", "Mutations"]
"""list[str]: List of attribute options for highlighting in the structure.

- KLIFS: Highlight KLIFS pocket regions with region-specific colors and stick representation for key residues
- Phosphosites: Highlight phosphorylation sites with red color and stick representation
- Mutations: Highlight mutations with gradient coloring based on mutation counts
"""


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

    parser.add_argument(
        "--strAttr",
        type=str,
        default=None,
        help=(
            "Attribute to highlight in the structure (options: "
            f"{', '.join(LIST_ATTR_OPTIONS)}; default: None)"
        ),
        choices=LIST_ATTR_OPTIONS,
    )

    parser.add_argument(
        "--jsonMutations",
        type=str,
        required=False,
        help="JSON file of mutations to highlight in the structure (default: None)",
    )

    parser.add_argument(
        "--boolKLIFSConserved",
        action="store_false",
        help=(
            "If set, use conserved KLIFS pocket residues for stick representation; "
            "if not set (default), use manually curated KLIFS pocket residues."
        ),
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
    str_attr = args.strAttr
    filepath_json = args.jsonMutations

    structure_kwargs = {}
    if str_attr == "Mutations":
        # make sure mutations are provided
        if not args.jsonMutations:
            raise ValueError(
                "Mutations file must be provided in "
                "--jsonMutations when --strAttr is 'Mutations'."
            )
        else:
            if not path.exists(filepath_json):
                raise FileNotFoundError(f"Mutations file {filepath_json} not found.")
            # load mutations from JSON string
            with open(filepath_json) as f:
                dict_mutations = json.load(f)
                dict_mutations = {
                    k1: {int(k2): float(v2) for k2, v2 in v1.items()}
                    for k1, v1 in dict_mutations.items()
                }

            # add mutations to structure kwargs
            if gene in dict_mutations:
                structure_kwargs["dict_mutations"] = dict_mutations[gene]
            else:
                raise ValueError(
                    f"No mutations found for gene {gene} in provided JSON file."
                )

    # add flag for KLIFS conserved residues
    structure_kwargs["bool_klifs_manual"] = args.boolKLIFSConserved

    _, obj_viz = generate_sequence_and_structure_viewers(
        str_kinase=gene,
        dict_colors=DICT_COLORS["ALPHABET_PROJECT"]["DICT_COLORS"],
        str_attr=str_attr,
        structure_kwargs=structure_kwargs,
    )

    pymol_generator = PyMOLGenerator(viz=obj_viz)
    str_subdirs = path.join("images", "pymol_output", gene, str_attr.lower())
    if args.output_dir:
        out_dir = path.join(args.output_dir, str_subdirs)
    else:
        out_dir = path.join(get_repo_root(), str_subdirs)
    pymol_generator.save_pymol_files(out_dir)


if __name__ == "__main__":
    main()
