import logging
from dataclasses import field
from os import path

import pandas as pd
from mkt.ml.datasets.finetune import FineTuneDataset
from mkt.ml.utils import get_repo_root

logger = logging.getLogger(__name__)


class PKIS2KinaseSplit(FineTuneDataset):
    """PKIS2 dataset kinase split.

    Parameters
    ----------
    col_kinase_split : str
        Column name for kinase groups in the dataset.
    list_kinase_split : list[str] | None
        List of kinase groups to use for testing. If None, no split is applied.

    """

    # FineTuneDataset arguments
    filepath: str | None = None
    col_labels: str | None = None
    col_kinase: str | None = None
    col_drug: str | None = None

    # kinase split arguments
    col_kinase_split: str = None
    list_kinase_split: list[str] | None = None

    def __init__(self, **kwargs):
        """Initialize the PKIS2KinaseSplit dataset."""
        if "filepath" not in kwargs:
            kwargs["filepath"] = path.join(get_repo_root(), "data/pkis_data.csv")
        if "col_labels" not in kwargs:
            kwargs["col_labels"] = "percent_displacement"
        if "col_kinase" not in kwargs:
            kwargs["col_kinase"] = "klifs"
        if "col_drug" not in kwargs:
            kwargs["col_drug"] = "Smiles"
        if "col_kinase_split" not in kwargs:
            kwargs["col_kinase_split"] = "kincore_group"
        if "list_kinase_split" not in kwargs:
            kwargs["list_kinase_split"] = ["TK", "TKL"]

        super().__init__(**kwargs)

    def prepare_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset based on kincore_group values.

        Returns:
        --------
        df_train, df_test: pd.DataFrame
            Training and testing DataFrames
        """
        # Make sure the column exists in the dataframe
        if self.col_kinase_split not in self.df.columns:
            logger.error(f"Column '{self.col_kinase_split}' not found in the dataset")
            raise ValueError(
                f"Column '{self.col_kinase_split}' not found in the dataset"
            )

        # Make sure list_kinase_split is not None
        if self.list_kinase_split is None:
            logger.warning("list_kinase_split is None. Using the entire dataset.")
            return self.df, self.df

        idx_test = self.df[self.col_kinase_split].apply(
            lambda x: x in self.list_kinase_split
        )

        df_train, df_test = (
            self.df.loc[~idx_test, :].copy(),
            self.df.loc[idx_test, :].copy(),
        )

        logger.info(
            f"Training set size: {len(df_train)}\n"
            f"Test set size: {len(df_test)}\n"
            f"Test kinase groups: {self.list_kinase_split}\n"
        )

        return df_train, df_test


class PKIS2CrossValidation(FineTuneDataset):
    """PKIS2 dataset split."""

    # FineTuneDataset arguments
    filepath: str | None = None
    col_labels: str | None = None
    col_kinase: str | None = None
    col_drug: str | None = None

    # kinase split arguments
    k_folds: int | None = None
    seed: int | None = None

    def __init__(self, **kwargs):
        """Initialize the PKIS2CrossValidation dataset."""
        if "filepath" not in kwargs:
            kwargs["filepath"] = path.join(get_repo_root(), "data/pkis_data.csv")
        if "col_labels" not in kwargs:
            kwargs["col_labels"] = "percent_displacement"
        if "col_kinase" not in kwargs:
            kwargs["col_kinase"] = "klifs"
        if "col_drug" not in kwargs:
            kwargs["col_drug"] = "Smiles"
        if "k_folds" not in kwargs:
            kwargs["k_folds"] = 5
        if "seed" not in kwargs:
            kwargs["seed"] = 42

        super().__init__(**kwargs)

    def prepare_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset based on kincore_group values.

        Returns:
        --------
        df_train, df_test: pd.DataFrame
            Training and testing DataFrames
        """
        raise NotImplementedError(
            "This method is not implemented. Please use the PKIS2KinaseSplit class to split in Kinase Group."
        )
