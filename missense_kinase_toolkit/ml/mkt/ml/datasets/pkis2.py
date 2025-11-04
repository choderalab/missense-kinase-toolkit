import logging
from os import path

import pandas as pd
from mkt.ml.datasets.finetune import FineTuneDataset
from mkt.schema.io_utils import get_repo_root

logger = logging.getLogger(__name__)


DICT_PKIS2_KWARGS = {
    "base": {
        "filepath": path.join(get_repo_root(), "data/pkis2_data_processed.csv"),
        "col_labels": "y",
        "col_kinase": "seq_construct_unaligned",
        "col_drug": "smiles",
    },
    "kinase_split": {
        "col_kinase_split": "group_consensus",
        "list_kinase_split": ["TK", "TKL"],
    },
    "cross_validation": {
        "k_folds": 5,
        "fold_idx": None,
        "seed": 42,
    },
}


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
        dict_kwargs = DICT_PKIS2_KWARGS["base"] | DICT_PKIS2_KWARGS["kinase_split"]
        for key, value in dict_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

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
    fold_idx: int | None = None
    seed: int | None = None

    def __init__(self, **kwargs):
        """Initialize the PKIS2CrossValidation dataset."""
        dict_kwargs = DICT_PKIS2_KWARGS["base"] | DICT_PKIS2_KWARGS["cross_validation"]
        for key, value in dict_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

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
