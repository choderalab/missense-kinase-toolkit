#!/usr/bin/env python3

import logging
from os import path

from mkt.databases import config
from mkt.databases.datasets.davis import DavisDataset
from mkt.databases.datasets.pkis2 import PKIS2Dataset
from mkt.databases.log_config import configure_logging
from mkt.schema.io_utils import get_repo_root

logger = logging.getLogger(__name__)


def main():
    """Generate dataset CSV files for all datasets."""

    configure_logging()

    try:
        config.set_request_cache(path.join(get_repo_root(), "requests_cache.sqlite"))
    except Exception as e:
        logger.warning(f"Failed to set request cache: {e}")
        config.set_request_cache(path.join(".", "requests_cache.sqlite"))

    # generate Davis dataset CSV file
    DavisDataset(bool_save=True)

    # generate PKIS2 dataset CSV file
    PKIS2Dataset(bool_save=True)


if __name__ == "__main__":
    main()
