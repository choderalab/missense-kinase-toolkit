#!/usr/bin/env python

import argparse
import logging
import os
import shutil

from mkt.databases.config import set_request_cache
from mkt.databases.io_utils import create_tar_without_metadata, get_repo_root
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
            path_out = os.path.join(path_repo, val)
            logger.info(f"Using user-provided path for {key} provided {path_out}...")
        else:
            if key == "reports":
                path_out = os.path.join(path_repo, "images")
            else:
                path_out = os.path.join(
                    path_repo, "missense_kinase_toolkit/schema/mkt/schema/KinaseInfo"
                )
            logger.info(f"Using default path for {key} provided {path_out}...")
        if not os.path.exists(path_out):
            logger.info(
                f"Output directory for {key} does not exist: {path_out}. Creating..."
            )
            try:
                os.makedirs(path_out)
            except Exception as e:
                logger.error(f"Failed to create directory {path_out}: {e}")
                exit(1)
        if not os.path.isdir(path_out):
            logger.error(f"Output path for {key} is not a directory: {path_out}")
            exit(1)
        dict_path[key] = path_out
    path_objects, path_reports = dict_path["objects"], dict_path["reports"]

    set_request_cache(os.path.join(get_repo_root(), "requests_cache.sqlite"))

    # perform the API or scraper call to generate KinaseInfo objects
    dict_obj = generate_dict_obj_from_api_or_scraper()

    # harmonize the KinaseInfo objects
    dict_kinaseinfo_uniprot = combine_kinaseinfo_uniprot(dict_obj)
    dict_kinaseinfo_kd = combine_kinaseinfo_kd(dict_obj)
    dict_kinaseinfo = combine_kinaseinfo(dict_kinaseinfo_uniprot, dict_kinaseinfo_kd)

    # serialize the KinaseInfo objects to the objects directory
    io_utils.serialize_kinase_dict(dict_kinaseinfo, str_path=path_objects)

    # create tar file without metadata one level from the objects directory
    create_tar_without_metadata(
        path_source=path_objects,
        filename_tar=os.path.join(dict_path["objects"], "..", "KinaseInfo.tar.gz"),
    )

    # generate kinase info plot
    generate_kinase_info_plot(dict_kinaseinfo, path_reports)

    # remove all files in the objects directory
    shutil.rmtree(path_objects)


if __name__ == "__main__":
    main()
