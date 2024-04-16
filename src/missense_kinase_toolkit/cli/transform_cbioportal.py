#!/usr/bin/env python

import argparse

from missense_kinase_toolkit import config, io_utils


def parsearg_utils():
    parser = argparse.ArgumentParser(
        description="Concatenate, remove duplicates, and extract genes and mutation types of interest from cBioPortal data."
    )

    parser.add_argument(
        "--mutationTypes",
        type=str,
        help="Optional: Mutation type(s) to extract, separated by commas; default: `Missense_Mutation` (str)",
        default="Missense_Mutation",
    )

    parser.add_argument(
        "--outDir",
        type=str,
        help="Required: Output directory path (str)",
    )

    parser.add_argument(
        "--requestsCache",
        type=str,
        default="requests_cache",
        help="Optional: Requests cache; default: requests_cache (str)",
    )

    parser.add_argument(
        "--listCols",
        type=str,
        default="HGNC Name, UniprotID",
        help="Optional: List of columns to merge separated by comma with the first as the merge on column of HGNC gene names; default: `HGNC Name, UniprotID` (str)",
    )

    parser.add_argument(
        "--csvRef",
        type=str,
        default="kinhub.csv",
        help="Optional: CSV file in outDir that contains; default: `kinhub.csv` (str)",
    )

    # TODO: add logging functionality
    return parser


def main():
    args = parsearg_utils().parse_args()

    config.set_output_dir(args.outDir)
    config.set_request_cache(args.requestsCache)

    list_mutations = io_utils.convert_str2list(args.mutationTypes)
    list_cols = io_utils.convert_str2list(args.listCols)

    df_cbioportal = io_utils.concatenate_csv_files_with_glob("*_mutations.csv")

    df_kinhub = io_utils.load_csv_to_dataframe(args.csvRef)
    # df_kinhub = scrapers.kinhub()
    # io_utils.save_dataframe_to_csv(df_kinhub, "kinhub.csv")

    list_kinase_hgnc = df_kinhub[list_cols[0]].to_list()
    # list_kinase_hgnc = df_kinhub["HGNC Name"].to_list()

    df_subset = df_cbioportal.loc[df_cbioportal["mutationType"].isin(list_mutations), ].reset_index(drop=True)
    df_subset = df_subset.loc[df_subset["hugoGeneSymbol"].isin(list_kinase_hgnc), ].reset_index(drop=True)

    # list_cols = ["HGNC Name", "UniprotID"]
    df_subset_merge = df_subset.merge(df_kinhub[list_cols],
                                      how = "left",
                                      left_on = "hugoGeneSymbol",
                                      right_on = list_cols[0])
    df_subset_merge = df_subset_merge.drop([list_cols[0]], axis=1)

    io_utils.save_dataframe_to_csv(df_subset_merge, "transformed_mutations.csv")
