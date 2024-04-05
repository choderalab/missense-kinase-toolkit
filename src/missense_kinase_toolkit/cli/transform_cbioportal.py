#!/usr/bin/env python

import argparse

from missense_kinase_toolkit import config, scrapers, io_utils


def parsearg_utils():
    parser = argparse.ArgumentParser(
        description="Concatenate, remove duplicates, and extract genes and mutation types of interest"
    )

    parser.add_argument(
        "--mutations",
        type=str,
        help="Optional: Mutation type(s) to extract, separated by commas (e.g., `Missense_Mutation`) (str)",
        default="Missense_Mutation",
    )

    parser.add_argument(
        "--outDir",
        type=str,
        help="Required: Output directory path (str)",
    )

    parser.add_argument(
        "--requestsCache",
        type=bool,
        default=False,
        help="Optional: Requests cache; default False (bool)",
    )

    # TODO: add logging functionality
    return parser


def main():
    args = parsearg_utils().parse_args()

    str_mutations = args.mutations
    list_mutations = str_mutations.split(",")
    list_mutations = [mutation.strip() for mutation in list_mutations]

    # required argument
    config.set_output_dir(args.outDir)

    # optional argument
    config.set_request_cache(args.requestsCache)

    df_cbioportal = io_utils.concatenate_csv_files_with_glob("*_mutations.csv")

    df_kinhub = scrapers.kinhub()
    io_utils.save_dataframe_to_csv(df_kinhub, "kinhub.csv")

    list_kinase_hgnc = df_kinhub["HGNC Name"].to_list()

    df_subset = df_cbioportal.loc[df_cbioportal["mutationType"].isin(list_mutations), ].reset_index(drop=True)
    df_subset = df_subset.loc[df_subset["hugoGeneSymbol"].isin(list_kinase_hgnc), ].reset_index(drop=True)

    list_cols = ["HGNC Name", "UniprotID"]
    df_subset_merge = df_subset.merge(df_kinhub[list_cols],
                                      how = "left",
                                      left_on = "hugoGeneSymbol",
                                      right_on = "HGNC Name")
    df_subset_merge = df_subset_merge.drop(["HGNC Name"], axis=1)

    io_utils.save_dataframe_to_csv(df_subset_merge, "transformed_mutations.csv")
