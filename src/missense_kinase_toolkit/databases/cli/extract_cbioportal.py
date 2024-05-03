#!/usr/bin/env python

import argparse

from missense_kinase_toolkit.databases import config, io_utils, cbioportal

def parsearg_utils():
    parser = argparse.ArgumentParser(
        description="Get mutations from cBioPortal instance for all specified studies."
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
        "--cohort",
        type=str,
        help="Optional: cBioPortal cohort IDs separated by commas (e.g., `msk_impact_2017` for Zehir, 2017 and `mskimpact` for MSKCC clinical sequencing cohort) (str)",
        default="msk_impact_2017",
    )

    parser.add_argument(
        "--instance",
        type=str,
        help="Optional: cBioPortal instance (e.g., `cbioportal.mskcc.org`). Default: `www.cbioportal.org` (str)",
        default="www.cbioportal.org",
    )

    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="Optional: cBioPortal API token (str)",
    )

    # TODO: add logging functionality
    return parser


def main():
    args = parsearg_utils().parse_args()

    list_studies = io_utils.convert_str2list(args.cohort)

    config.set_output_dir(args.outDir)
    config.set_cbioportal_instance(args.instance)

    try:
        if args.token != "":
            config.set_cbioportal_token(args.token)
    except AttributeError:
        pass

    for study in list_studies:
        cbioportal.Mutations(study).get_and_save_cbioportal_cohort_mutations()
