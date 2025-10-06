import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class FineTuneDataset(ABC):
    """Fine-tune dataset for kinase and drug interactions."""

    filepath: str
    """Path to the dataset file."""
    col_labels: str
    """Column name for labels values in the dataset."""
    col_kinase: str
    """Column name for kinase identifiers in the dataset."""
    col_drug: str
    """Column name for drug identifiers in the dataset."""
    model_drug: str
    """Pre-trained model name for drug sequences; default is DeepChem/ChemBERTa-77M-MTR."""
    model_kinase: str
    """Pre-trained model name for kinase sequences; default is facebook/esm2_t6_8M_UR50D."""
    k_folds: int | None = None
    """Number of folds for cross-validation. If None, splits are applied via prepare_splits."""
    fold_idx: int | None = None
    """Index of the fold to use for testing."""
    seed: int | None = None
    """Random seed for reproducibility. If None, no seed is set."""
    col_kinase_split: str | None = None
    """Column name for kinase groups in the dataset."""
    list_kinase_split: list[str] | None = None
    """List of kinase groups to use for testing. If None, no split is applied."""

    def __post_init__(self):
        """Post-initialization method to load the dataset."""
        self.df = self.load_dataset()

        self.tokenizer_drug = AutoTokenizer.from_pretrained(self.model_drug)
        self.tokenizer_kinase = AutoTokenizer.from_pretrained(self.model_kinase)
        self.max_drug, self.max_kinase = self.find_max_length()

        if self.k_folds is not None:
            datasets_dict = self.create_cv_folds()
            self.dataset_train = {k: v["train"] for k, v in datasets_dict.items()}
            self.dataset_test = {k: v["val"] for k, v in datasets_dict.items()}
            self.scaler = {k: v["scaler"] for k, v in datasets_dict.items()}
        else:
            try:
                df_train, df_test = self.prepare_splits()
            except Exception as e:
                logger.info(
                    f"Exception: {e}\nNo split applied. Using the entire dataset.\n"
                )
                df_train, df_test = self.df, self.df
            df_train, df_test, self.scaler = self.standardize_target(df_train, df_test)
            self.dataset_train = Dataset.from_pandas(df_train).map(
                lambda x: self.tokenize_and_combine(x),
                batched=True,
            )
            self.dataset_test = Dataset.from_pandas(df_test).map(
                lambda x: self.tokenize_and_combine(x),
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

    def create_cv_folds(self) -> dict:
        """Create cross-validation folds for the dataset.

        Returns:
        --------
        dict_dataset: dict
            Dictionary containing training and testing datasets for each fold.

        """
        full_dataset = Dataset.from_pandas(self.df)
        if self.seed is not None:
            full_dataset = full_dataset.shuffle(seed=self.seed)

        total_size = len(full_dataset)
        fold_size = total_size // self.k_folds

        dict_dataset = {}

        for fold_idx in tqdm(range(self.k_folds), desc="Creating CV folds..."):
            if self.fold_idx is not None and fold_idx != self.fold_idx:
                logger.info(f"Skipping fold {fold_idx + 1}...")
                continue
            else:
                # calculate start and end indices for validation set
                val_start = fold_idx * fold_size
                val_end = (
                    val_start + fold_size if fold_idx < self.k_folds - 1 else total_size
                )

                # create train and validation indices
                val_indices = list(range(val_start, val_end))
                train_indices = [i for i in range(total_size) if i not in val_indices]

                # select samples by indices
                train_dataset = full_dataset.select(train_indices)
                val_dataset = full_dataset.select(val_indices)

                # standardize the labels
                scaler = StandardScaler()
                train_column_values = np.array(train_dataset[self.col_labels]).reshape(
                    -1, 1
                )
                scaler.fit(train_column_values)

                def add_standardized_column(example):
                    value = np.array([example[self.col_labels]]).reshape(-1, 1)
                    standardized_value = scaler.transform(value).item()
                    example[self.col_labels + "_std"] = standardized_value
                    return example

                logger.info(f"Standardizing labels fold-{fold_idx+1}...")
                train_dataset_scaled = train_dataset.map(add_standardized_column)
                val_dataset_scaled = val_dataset.map(add_standardized_column)

                logger.info(f"Tokenizing and combining sequences fold-{fold_idx+1}...")
                train_dataset_tokenize = train_dataset_scaled.map(
                    lambda x: self.tokenize_and_combine(x),
                    batched=True,
                )
                val_dataset_tokenize = val_dataset_scaled.map(
                    lambda x: self.tokenize_and_combine(x),
                    batched=True,
                )

                dict_dataset[f"fold_{fold_idx+1}"] = {
                    "train": train_dataset_tokenize,
                    "val": val_dataset_tokenize,
                    "scaler": scaler,
                }

        return dict_dataset

    @abstractmethod
    def prepare_splits(self):
        """Prepare training and testing splits based kinase or drug groups."""
        ...

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

        colname_new = self.col_labels + "_std"

        # fit scaler on training data only
        df_train[colname_new] = scaler.fit_transform(df_train[[self.col_labels]])

        # transform test data using the scaler fitted on training data
        df_test[colname_new] = scaler.transform(df_test[[self.col_labels]])

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
    ) -> dict[str, Any]:
        """Tokenize and combine kinase and drug sequences.

        Parameters:
        -----------
        batch_in: Dataset
            Batch of input dataset

        Returns:
        --------
        result: dict[str, Any]
            Dictionary with tokenized sequences, masks, and labels

        """
        smiles_tokenized = self.tokenizer_drug(
            batch_in[self.col_drug],
            padding="max_length",
            truncation=True,
            max_length=self.max_drug,
            return_tensors="pt",
        )

        klifs_tokenized = self.tokenizer_kinase(
            batch_in[self.col_kinase],
            padding="max_length",
            truncation=True,
            max_length=self.max_kinase,
            return_tensors="pt",
        )

        result = {
            "smiles_input_ids": smiles_tokenized.input_ids,
            "smiles_attention_mask": smiles_tokenized.attention_mask,
            "klifs_input_ids": klifs_tokenized.input_ids,
            "klifs_attention_mask": klifs_tokenized.attention_mask,
            "labels": batch_in[self.col_labels + "_std"],
        }

        return result
