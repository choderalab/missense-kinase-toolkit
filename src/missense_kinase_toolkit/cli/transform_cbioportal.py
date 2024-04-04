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
        default="",
    )

    parser.add_argument(
        "--outDir",
        type=str,
        help="Required: Output directory path (str)",
    )

    parser.add_argument(
        "--requestsCache",
        type=str,
        default="",
        help="Optional: Requests cache (str)",
    )

    # TODO: add logging functionality
    return parser


def main():
    args = parsearg_utils().parse_args()

    str_mutations = args.mutation
    list_mutations = str_mutations.split(",")
    list_mutations = [mutation.strip() for mutation in list_mutations]

    # required argument
    config.set_output_dir(args.outDir)

    try:
        if args.requestsCache != "":
            config.set_request_cache(args.requestsCache)
    except AttributeError:
        pass

    df = concatenate_csv_files_with_glob()
