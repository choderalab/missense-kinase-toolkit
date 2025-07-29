import argparse
import logging
from os import path

import numpy as np
import pandas as pd
from mkt.ml.datasets.process import DavisDataset, PKIS2Dataset
from mkt.ml.log_config import add_logging_flags, configure_logging
from mkt.ml.utils import get_repo_root

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate datasets for ML models.")

    parser.add_argument(
        "--drop_pkis2",
        action="store_true",
        help="Drop PKIS2 values where percent displacement is 0.",
    )

    parser.add_argument(
        "--drop_davis",
        action="store_true",
        help="Drop Davis values where Kd exceeds 10,000.",
    )

    parser.add_argument(
        "--col_dropna",
        type=str,
        nargs="+",
        default=["klifs", "kincore_kd"],
        help="Column to drop NA values from (default: kincore_kd).",
    )

    parser = add_logging_flags(parser)

    return parser.parse_args()


def main():
    """Main function to process datasets and print DataFrame shapes."""

    args = parse_args()

    configure_logging()

    try:
        pkis2_dataset = PKIS2Dataset()
        logger.info(
            "Initialized PKIS2Dataset successfully and saved.\n"
            f"Number of kinases: {pkis2_dataset.df["kinase_name"].nunique():,}\n"
            f"Number of compounds: {pkis2_dataset.df["smiles"].nunique():,}\n"
        )
    except Exception as e:
        logger.error(f"Failed to initialize PKIS2Dataset: {e}")
        return

    try:
        davis_dataset = DavisDataset()
        logger.info(
            "Initialized DavisDataset successfully and saved.\n"
            f"Number of kinases: {davis_dataset.df["kinase_name"].nunique():,}\n"
            f"Number of compounds: {davis_dataset.df["smiles"].nunique():,}\n"
        )
    except Exception as e:
        logger.error(f"Failed to process DavisDataset: {e}")

    df_pkis2 = pkis2_dataset.df.copy()
    df_davis = davis_dataset.df.copy()

    list_drop = args.col_dropna
    if list_drop != []:
        logger.info(f"Dropping NAs from columns: {list_drop}")

        logger.info(
            f"PKIS2: {df_pkis2[list_drop].isna().any(axis=1).sum():,} rows with NAs"
        )
        df_pkis2 = df_pkis2.dropna(subset=list_drop).reset_index(drop=True)
        if args.drop_pkis2:
            logger.info(
                f"PKIS2: {(df_pkis2['y'] == 0).sum():,} rows with 0 percent displacement"
            )
            df_pkis2 = df_pkis2[df_pkis2["y"] != 0].reset_index(drop=True)
        # higher percent displacement values are more potent
        df_pkis2["z-score"] = (df_pkis2["y"] - df_pkis2["y"].mean()) / df_pkis2[
            "y"
        ].std()

        logger.info(
            f"Davis: {df_davis[list_drop].isna().any(axis=1).sum():,} rows with NAs"
        )
        df_davis = df_davis.dropna(subset=list_drop).reset_index(drop=True)
        if args.drop_davis:
            logger.info(
                f"Davis: {(df_davis['y'] == 10000).sum():,} rows with Kd = 10,000"
            )
            df_davis = df_davis[df_davis["y"] < 10000].reset_index(drop=True)
        # lower Kd values are more potent - convert to micromolar and then pKd
        series_pkd = df_davis["y"].apply(lambda x: -np.log10(x / 1000))
        df_davis["z-score"] = (series_pkd - series_pkd.mean()) / series_pkd.std()

    df_concat = pd.concat([df_pkis2, df_davis], ignore_index=True)
    logger.info(
        f"\nKinases: {df_concat.loc[df_concat['source'] == 'PKIS2', 'kinase_name'].nunique()} (PKIS2), "
        f"{df_concat.loc[df_concat['source'] == 'Davis', 'kinase_name'].nunique()} (Davis)\n"
        f"Compounds: {df_concat.loc[df_concat['source'] == 'PKIS2', 'smiles'].nunique()} (PKIS2), "
        f"{df_concat.loc[df_concat['source'] == 'Davis', 'smiles'].nunique()} (Davis)\n"
    )

    filepath = path.join(get_repo_root(), "data/concat_data_processed.csv")
    df_concat.to_csv(filepath, index=False)
