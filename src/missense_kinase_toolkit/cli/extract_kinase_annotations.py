#!/usr/bin/env python

import argparse

import pandas as pd

from missense_kinase_toolkit import config, io_utils, scrapers, klifs

def parsearg_utils():
    parser = argparse.ArgumentParser(
        description="Get kinase annotations from KinHub and KLIFS databases."
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
        help="Optional: Requests cache; default: `requests_cache` (str)",
    )

    parser.add_argument(
        "--csvKinhub",
        type=str,
        default="kinhub.csv",
        help="Optional: CSV file in outDir that contains Kinhub kinase info; default: `kinhub.csv` (str)",
    )

    parser.add_argument(
        "--csvKLIFS",
        type=str,
        default="klifs.csv",
        help="Optional: CSV file in outDir that contains KLIFS kinase info; default: `klifs.csv` (str)",
    )

    # TODO: add logging functionality
    return parser


def main():
    args = parsearg_utils().parse_args()

    config.set_output_dir(args.outDir)
    config.set_request_cache(args.requestsCache)

    # get KinHub list of kinases
    df_kinhub = scrapers.kinhub()
    io_utils.save_dataframe_to_csv(df_kinhub, args.csvKinhub)

    # get KLIFS annotations
    list_kinase_hgnc = df_kinhub["HGNC Name"].to_list()
    dict_kinase_info = {}
    for kinase in list_kinase_hgnc:
        dict_kinase_info[kinase] = klifs.KinaseInfo(kinase)._kinase_info
    df_klifs = pd.DataFrame(dict_kinase_info).T
    df_klifs = df_klifs.rename_axis("HGNC Name").reset_index()
    io_utils.save_dataframe_to_csv(df_klifs, args.csvKLIFS)
