import logging
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class FineTuneDataset:
    """Fine-tune dataset for kinase and drug interactions.

    Parameters
    ----------
    filepath : str
        Path to the dataset file.
    col_kinase_split : str
        Column name for kinase groups in the dataset.
    col_drug_split : str
        Column name for drug groups in the dataset.
    col_target : str
        Column name for target values in the dataset.
    col_percent_displacement : str
        Column name for percent displacement values in the dataset.
    col_kinase : str
        Column name for kinase identifiers in the dataset.
    col_drug : str
        Column name for drug identifiers in the dataset.
    list_kinase_split : list[str] | None
        List of kinase groups to use for testing. If None, no split is applied.
    list_drug_split : list[str] | None
        List of drug groups to use for testing. If None, no split is applied.
    model_kinase : str
        Pre-trained model name for kinase sequences.
    model_drug : str
        Pre-trained model name for drug sequences.

    """

    filepath: str
    col_kinase_split: str
    col_yval: str
    col_kinase: str
    col_drug: str
    list_kinase_split: list[str] | None = None
    # TODO: Add drug_split parameter - add col_col too
    # list_drug_split: list[str] | None = None
    model_drug: str = "DeepChem/ChemBERTa-77M-MTR"
    model_kinase: str = "facebook/esm2_t6_8M_UR50D"

    def __post_init__(self):
        """Post-initialization method to load the dataset."""
        self.df = self.load_dataset()

        self.tokenizer_drug = AutoTokenizer.from_pretrained(self.model_drug)
        self.tokenizer_kinase = AutoTokenizer.from_pretrained(self.model_kinase)

        if self.list_kinase_split is not None:
            df_train, df_test = self.prepare_splits()
        else:
            logger.info("No split applied. Using the entire dataset.")
            df_train, df_test = self.df, self.df

        df_train, df_test, self.scaler = self.standardize_target(df_train, df_test)
        max_drug, max_kinase = self.find_max_length()

        self.dataset_train = Dataset.from_pandas(df_train).map(
            lambda x: self.tokenize_and_combine(x, max_drug, max_kinase),
            batched=True,
        )
        self.dataset_test = Dataset.from_pandas(df_test).map(
            lambda x: self.tokenize_and_combine(x, max_drug, max_kinase),
            batched=True,
        )

    def check_filepath(self) -> bool | None:
        """Check if the file path is valid."""
        if not os.path.exists(self.filepath):
            logger.error(f"File not found: {self.filepath}")
        if not self.filepath.endswith(".csv"):
            logger.error("File must be a CSV file.")
        return True

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset from CSV file."""
        if self.check_filepath():
            df = pd.read_csv(self.filepath)
            return df

    def prepare_splits(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset based on kincore_group values.

        Returns:
        --------
        df_train, df_test: pd.DataFrame
            Training and testing DataFrames
        """
        idx_test = self.df[self.col_kinase_split].apply(
            lambda x: x in self.list_kinase_split
        )

        df_train, df_test = self.df.loc[~idx_test, :], self.df.loc[idx_test, :]

        logger.info(
            f"Training set size: {len(df_train)}\n" f"Test set size: {len(df_test)}\n"
        )

        return df_train, df_test

    def standardize_target(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """Standardize the percent_displacement column using z-score from training data.

        Parameters:
        -----------
        df_train: pd.DataFrame
            Training DataFrame
        df_test: pd.DataFrame
            Testing DataFrame

        Returns:
        --------
        df_train, df_test: pd.DataFrame
            DataFrames with standardized percent_displacement
        scaler: StandardScaler
            Fitted StandardScaler for later use

        """
        scaler = StandardScaler()

        colname_new = self.col_yval + "_std"

        # fit scaler on training data only
        df_train[colname_new] = scaler.fit_transform(df_train[[self.col_yval]])

        # transform test data using the scaler fitted on training data
        df_test[colname_new] = scaler.transform(df_test[[self.col_yval]])

        return df_train, df_test, scaler

    def find_max_length(self) -> int:
        """Find the maximum length of sequences in the dataset.

        Parameters:
        -----------
        dataset_in: Dataset
            Input dataset

        Returns:
        --------
        int: Maximum length of sequences
        """
        # add 2 for special tokens like [CLS] and [SEP]
        max_smiles_length = (
            max(
                [
                    len(self.tokenizer_drug.tokenize(x))
                    for x in self.df[self.col_drug].unique()
                ]
            )
            + 2
        )

        max_klifs_length = (
            max(
                [
                    len(self.tokenizer_kinase.tokenize(x))
                    for x in self.df[self.col_kinase].unique()
                ]
            )
            + 2
        )

        return max_smiles_length, max_klifs_length

    def tokenize_and_combine(
        self,
        batch_in: Dataset,
        max_drug: int,
        max_kinase: int,
    ) -> dict[str, Any]:
        """Tokenize and combine kinase and drug sequences.

        Parameters:
        -----------
        batch_in: Dataset
            Batch of input dataset
        max_drug: int
            Maximum length for drug sequences
        max_kinase: int
            Maximum length for kinase sequences

        Returns:
        --------
        result: dict[str, Any]
            Dictionary with tokenized sequences, masks, and labels

        """
        smiles_tokenized = self.tokenizer_drug(
            batch_in[self.col_drug],
            padding="max_length",
            truncation=True,
            max_length=max_drug,
            return_tensors="pt",
        )

        klifs_tokenized = self.tokenizer_kinase(
            batch_in[self.col_kinase],
            padding="max_length",
            truncation=True,
            max_length=max_kinase,
            return_tensors="pt",
        )

        result = {
            "smiles_input_ids": smiles_tokenized.input_ids,
            "smiles_attention_mask": smiles_tokenized.attention_mask,
            "klifs_input_ids": klifs_tokenized.input_ids,
            "klifs_attention_mask": klifs_tokenized.attention_mask,
            "labels": batch_in[self.col_yval + "_std"],
        }

        return result
