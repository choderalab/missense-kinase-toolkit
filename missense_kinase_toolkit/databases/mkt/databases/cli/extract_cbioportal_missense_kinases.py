#!/usr/bin/env python

import argparse
import logging
import os
from ast import literal_eval

from mkt.databases import cbioportal, config
from mkt.databases.io_utils import get_repo_root
from mkt.databases.log_config import add_logging_flags, configure_logging

logger = logging.getLogger(__name__)


def get_parser():
    """Generate a parser for the command line interface.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the command line interface

    """
    parser = argparse.ArgumentParser(
        description="Generate missense kinase mutation data for cBioPortal cohort."
    )

    parser.add_argument(
        "--cbioportalInstance",
        type=str,
        default="www.cbioportal.org",
        help="cBioPortal instance; default: public (alternative: cbioportal.mskcc.org, requires token).",
    )

    # CBIO_TOKEN=$(cat '<PATH_TO_FILE>/cbioportal_data_access_token.txt' | sed 's/^token: //')
    parser.add_argument(
        "--cbioportalToken",
        type=str,
        default="",
        help="cBioportal token, if using non-public instance; default: ''.",
    )

    parser.add_argument(
        "--studyId",
        type=str,
        default=None,
        help="Study ID for the cBioPortal cohort.",
    )

    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Path to save requests cache; default: None (will try root repo, else current wd).",
    )

    parser.add_argument(
        "--outputPath",
        type=str,
        default=None,
        help="Path to save the extracted missense kinase data; default: None (will try root repo, else current wd).",
    )

    parser.add_argument(
        "--generateHeatmap",
        action="store_true",
        help="Invoke flag to generate a heatmap of missense kinase mutations (default: False).",
    )

    parser.add_argument(
        "--dictHeatmap",
        type=str,
        default=None,
        help="Path to a dictionary file for heatmap generation; default: None (will use default arguments).",
    )

    parser = add_logging_flags(parser)

    return parser


def main():
    configure_logging()

    args = get_parser().parse_args()

    if args.cache:
        if not os.path.exists(args.cache):
            logger.info(f"Cache directory {args.cache} does not exist. Creating...")
            os.makedirs(args.cache)
        else:
            logger.info(
                f"Using existing cache at {args.cache} for cBioPortal requests."
            )
        config.set_request_cache(args.cache)
    else:
        try:
            config.set_request_cache(
                os.path.join(get_repo_root(), "requests_cache.sqlite")
            )
            logger.info("Using default cache at requests_cache.sqlite in repo root.")
        except Exception as e:
            logger.error(
                f"Could not determine repo root: {e}\n"
                "Using default cache at requests_cache.sqlite in current directory."
            )

            config.set_request_cache(os.path.join(".", "requests_cache.sqlite"))

    config.set_cbioportal_instance(args.cbioportalInstance)
    config.set_cbioportal_token(args.cbioportalToken)

    if not args.studyId:
        cBioPortal = cbioportal.cBioPortal()
        logger.error(
            "Study ID is required. Please provide a study ID with --studyId.\n"
            f"Available studies for {args.cbioportalInstance}:\n"
        )
        for study in (
            cBioPortal._cbioportal.Studies.getAllStudiesUsingGET().response().result
        ):
            logger.error(f" - {study.studyId}: {study.description}")
        exit(1)

    logger.info(
        f"Extracting missense kinase data from {args.cbioportalInstance} for study {args.studyId}..."
    )

    kinase_missense_muts = cbioportal.KinaseMissenseMutations(args.studyId)
    df_kin_mis = kinase_missense_muts.get_kinase_missense_mutations()

    if args.outputPath:
        if not os.path.exists(args.outputPath):
            logger.info(f"Output path {args.outputPath} does not exist. Creating...")
            os.makedirs(args.outputPath)
        else:
            logger.info(f"Using existing output path {args.outputPath}.")
        path_out = args.outputPath
    else:
        try:
            path_out = get_repo_root()
            logger.info(f"Using repo root {path_out} as output path.")
        except Exception as e:
            logger.error(
                f"Could not determine repo root: {e}\n"
                "Using current directory as output path."
            )
            path_out = "."

    output_file = os.path.join(
        path_out, f"{args.studyId}_kinase_missense_mutations.csv"
    )
    df_kin_mis.to_csv(output_file, index=False)
    logger.info(f"Missense kinase mutations saved to {output_file}.")

    if args.generateHeatmap:
        output_heatmap = os.path.join(
            path_out, f"{args.studyId}_kinase_missense_heatmap.png"
        )
        if args.dictHeatmap:
            try:
                dict_heatmap = literal_eval(args.dictHeatmap)
                suffix = "".join([f"_{key}-{val}" for key, val in dict_heatmap.items()])
                output_heatmap = output_heatmap.replace(".png", f"{suffix}.png")
                logger.info(
                    f"Using user-provided heatmap settings from --dictHeatmap: {dict_heatmap}"
                )
                kinase_missense_muts.generate_heatmap_fig(
                    df_kin_mis,
                    filename=output_heatmap,
                    dict_clustermap_args=dict_heatmap,
                )
            except Exception as e:
                logger.error(
                    f"Failed to parse dictionary from --dictHeatmap: {e}\n"
                    f"Using default heatmap settings...\n"
                )
                kinase_missense_muts.generate_heatmap_fig(
                    df_kin_mis, filename=output_heatmap
                )
        else:
            kinase_missense_muts.generate_heatmap_fig(
                df_kin_mis, filename=output_heatmap
            )
        logger.info(f"Heatmap saved to {output_heatmap}.")


if __name__ == "__main__":
    main()
