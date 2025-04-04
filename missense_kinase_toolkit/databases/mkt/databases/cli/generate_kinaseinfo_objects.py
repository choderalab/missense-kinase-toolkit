#!/usr/bin/env python

import argparse
import logging
from os import path

from mkt.databases.config import set_request_cache
from mkt.databases.io_utils import get_repo_root
from mkt.databases.kinase_schema import (
    combine_kinaseinfo,
    combine_kinaseinfo_kd,
    combine_kinaseinfo_uniprot,
    generate_dict_obj_from_api_or_scraper,
)
from mkt.databases.log_config import add_logging_flags, configure_logging
from mkt.schema import io_utils

logger = logging.getLogger(__name__)


def get_parser():
    """Generate a parser for the command line interface.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the command line interface

    """
    parser = argparse.ArgumentParser(
        description="Generate KinaseInfo objects from API or scraper."
    )

    parser.add_argument(
        "--pathSave",
        type=str,
        default=None,
        help="Where to save KinaseInfo objects, relative to repo root; if not Github repo relative to current directory.",
    )

    parser = add_logging_flags(parser)

    return parser


def main():
    configure_logging()

    args = get_parser().parse_args()

    if args.pathSave is None:
        path_out = path.join(
            get_repo_root(),
            "missense_kinase_toolkit/schema/mkt/schema/KinaseInfo",
        )
        logger.info(f"No path provided; default directory is {path_out}...")
    else:
        path_out = path.join(get_repo_root(), args.pathSave)
        logger.info(f"Using user-provided path provided {path_out}...")

    if not path.exists(path_out):
        logger.error(f"Output directory does not exist: {path_out}")
        exit(1)
    if not path.isdir(path_out):
        logger.error(f"Output path is not a directory: {path_out}")
        exit(1)

    set_request_cache(path.join(get_repo_root(), "requests_cache.sqlite"))

    dict_obj = generate_dict_obj_from_api_or_scraper()
    dict_kinaseinfo_uniprot = combine_kinaseinfo_uniprot(dict_obj)
    dict_kinaseinfo_kd = combine_kinaseinfo_kd(dict_obj)
    dict_kinaseinfo = combine_kinaseinfo(dict_kinaseinfo_uniprot, dict_kinaseinfo_kd)

    io_utils.serialize_kinase_dict(dict_kinaseinfo, str_path=path_out)


if __name__ == "__main__":
    main()
