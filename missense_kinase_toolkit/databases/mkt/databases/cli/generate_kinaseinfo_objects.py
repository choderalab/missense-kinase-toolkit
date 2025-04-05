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
from mkt.databases.plot import generate_kinase_info_plot
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
        "--pathObjects",
        type=str,
        default=None,
        help="Where to save KinaseInfo objects, relative to repo root; if not Github repo relative to current directory.",
    )

    parser.add_argument(
        "--pathReports",
        type=str,
        default=None,
        help="Where to save reports, relative to repo root; if not Github repo relative to current directory.",
    )

    parser = add_logging_flags(parser)

    return parser


def main():
    configure_logging()

    args = get_parser().parse_args()

    path_repo = get_repo_root()

    dict_path = dict(zip(["objects", "reports"], [args.pathObjects, args.pathReports]))
    for key, val in dict_path.items():
        if val is not None:
            path_out = path.join(path_repo, val)
            logger.info(f"Using user-provided path for {key} provided {path_out}...")
        else:
            if key == "reports":
                path_out = path.join(path_repo, "images")
            else:
                path_out = path.join(
                    path_repo, "missense_kinase_toolkit/schema/mkt/schema/KinaseInfo"
                )
            logger.info(f"Using default path for {key} provided {path_out}...")
        if not path.exists(path_out):
            logger.error(f"Output directory for {key} does not exist: {path_out}")
            exit(1)
        if not path.isdir(path_out):
            logger.error(f"Output path for {key} is not a directory: {path_out}")
            exit(1)
        dict_path[key] = path_out

    set_request_cache(path.join(get_repo_root(), "requests_cache.sqlite"))

    dict_obj = generate_dict_obj_from_api_or_scraper()
    dict_kinaseinfo_uniprot = combine_kinaseinfo_uniprot(dict_obj)
    dict_kinaseinfo_kd = combine_kinaseinfo_kd(dict_obj)
    dict_kinaseinfo = combine_kinaseinfo(dict_kinaseinfo_uniprot, dict_kinaseinfo_kd)

    io_utils.serialize_kinase_dict(dict_kinaseinfo, str_path=dict_path["objects"])

    generate_kinase_info_plot(dict_kinaseinfo, dict_path["reports"])


if __name__ == "__main__":
    main()
