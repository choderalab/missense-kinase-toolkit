import argparse
import datetime
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.metrics import mean_squared_error


def parsearg_utils():
    """Argument parser to finetune ESM-2 model from HuggingFace."""

    parser = argparse.ArgumentParser(
        description="Run ESM-2 model from transformer library on PKIS2 Km, ATP data."
    )

    parser.add_argument(
        "-b",
        "--loadBest",
        help="Load best model at end (bool)",
        default=True,
        type=bool,
    )

    parser.add_argument(
        "-c",
        "--columnSeq",
        help="Column containing (str; default: kd)",
        default="seq_kincore",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--weightDecay",
        help="Weight decay (float; default: 0.1)",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--inputData",
        help="Path to csv file to load (str)",
        default="assets/pkis2_km_atp.csv",
        type=str,
    )

    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of training epochs (int; default: 500)",
        default=500,
        type=int,
    )

    parser.add_argument(
        "-g",
        "--loggingSteps",
        help="Logging steps (int; default: 1)",
        default=1,
        type=int,
    )

    parser.add_argument(
        "-k",
        "--kFold",
        help="K-fold (int; default: 5)",
        default=5,
        type=int,
    )

    parser.add_argument(
        "-l",
        "--saveLim",
        help="Save total limit (int; default: 2)",
        default=2,
        type=int,
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Model name (str; default: facebook/esm2_t6_8M_UR50D)",
        default="facebook/esm2_t6_8M_UR50D",
        type=str,
    )

    parser.add_argument(
        "-n",
        "--noSplit",
        help="Model name (store_true; default: False)",
        action="store_true",
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        help="Overwrite output directory (bool; default: True)",
        default=True,
        type=bool,
    )

    parser.add_argument(
        "-p",
        "--path",
        help="Path to save data model and data, if applicable (str)",
        default="/data1/tanseyw/projects/whitej/esm_km_atp",
        type=str,
    )

    parser.add_argument(
        "-r",
        "--learningRate",
        help="Learning rate (float; default: 0.000001)",
        default=0.000001,
        type=float,
    )

    parser.add_argument(
        "-s",
        "--seed",
        help="Random seed (int; default: 42)",
        default=42,
        type=int,
    )

    parser.add_argument(
        "-t",
        "--tBatch",
        help="Training batch size (int; default: 16)",
        default=8,
        type=int,
    )

    parser.add_argument(
        "-v",
        "--vBatch",
        help="Validation batch size (int; default: 16)",
        default=8,
        type=int,
    )

    parser.add_argument(
        "-w",
        "--warmup",
        help="Number of warm-up steps (int; default: 500)",
        default=500,
        type=int,
    )

    parser.add_argument(
        "--evalStrategy",
        help="Evaluation strategy (str; default: steps)",
        default="steps",
        type=str,
    )

    parser.add_argument(
        "--saveStrategy",
        help="Save strategy (str; default: steps)",
        default="steps",
        type=str,
    )

    parser.add_argument(
        "--wandbProject",
        help="Weights and Biases project (str; default: seq_atp_affinity)",
        default="seq_atp_affinity",
        type=str,
    )

    parser.add_argument(
        "--wandbRun",
        help='Weights and Biases run (str; default: "")',
        default="",
        type=str,
    )

    args = parser.parse_args()

    return args


def calc_zscore(
    list_in: list[float | int],
) -> list[float]:
    """Calculate z-scores for a list of values."""
    mean = sum(list_in) / len(list_in)
    std = (sum([(x - mean) ** 2 for x in list_in]) / (len(list_in) - 1)) ** 0.5
    list_out = [(x - mean) / std for x in list_in]
    return list_out


def invert_zscore(
    list_zscore: list[float],
    list_orig: list[float],
):
    """Convert back to original scale from z-scores."""
    mean = sum(list_orig) / len(list_orig)
    std = (sum([(x - mean) ** 2 for x in list_orig]) / (len(list_orig) - 1)) ** 0.5
    list_out = [(z * std) + mean for z in list_zscore]
    return list_out


def save_csv2csv(
    df: pd.DataFrame,
    path: str,
    csv_name: str | None = None,
    seed: int = 42,
    col_seq: str = "kd",
    col_lab: str = "ATP Conc.(uM)",
):
    """
    Process data for ESM-2 model from HuggingFace.
        Extracts sequence and labels and saves as Dataset in assets sub-dir.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns for sequence and label.
    path : str
        Path to save data.
    csv_name : str
        Name of csv file; default is None.
    seed : int
        Random seed.
    col_seq : str
        Column name for sequence; default is "kd".
    col_lab : str
        Column name for label; default is "ATP Conc.(uM)".

    Returns
    -------
    None
    """
    df = df.loc[df["Mutant"].apply(lambda x: x is False),].reset_index(drop=True)
    df_shuffle = df.copy().sample(frac=1, random_state=seed).reset_index(drop=True)
    df_out = df_shuffle[[col_seq, col_lab]]
    df_out.columns = ["seq", "label"]
    # df_out["label"] = df_out["label"].astype(float)
    df_out["label"] = calc_zscore(df_out["label"].apply(np.log10))

    if csv_name is None:
        x = datetime.datetime.now()
        csv_name = f"{x.strftime('%Y%m%d_%H%M%S')}_data.csv"

    data = Dataset.from_pandas(df_out)
    data.to_csv(os.path.join(path, "assets", csv_name), index=False)

    return csv_name


def load_csv2dataset(
    path: str,
    k_fold: int,
    csv_name: str,
):
    """Load data from csv file to dataset."""
    k_interval = int(100 / k_fold)
    file_path = os.path.join(path, "assets", csv_name)
    list_val = [f"train[{k}%:{k+k_interval}%]" for k in range(0, 100, k_interval)]
    list_train = [
        f"train[:{k}%]+train[{k+k_interval}%:]" for k in range(0, 100, k_interval)
    ]

    ds_val = load_dataset("csv", data_files=file_path, split=list_val)
    ds_train = load_dataset("csv", data_files=file_path, split=list_train)

    return ds_val, ds_train


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}


def parse_stats_dataframes(
    file: str,
    idx: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save training, evaluation, and final stats from trainer state log to dataframes.

    Parameters
    ----------
    file : str
        File name.
    idx : int | None
        Index of file (e.g., split). If None, no split annotation will be added
    """
    df = pd.read_csv(file)

    df_train = df.loc[[i for i in range(0, df.shape[0] - 1, 2)],]
    df_eval = df.loc[[i for i in range(1, df.shape[0], 2)],]
    df_final = pd.DataFrame(df.iloc[-1]).T

    if idx is not None:
        for df in (df_train, df_eval, df_final):
            df["fold"] = idx + 1

    for df in (df_train, df_eval, df_final):
        df.dropna(axis=1, how="all", inplace=True)

    return df_train, df_eval, df_final


# TODO add labels as parameter
def plot_label_histogram(
    val_df: pd.DataFrame,
    bool_orig: bool = True,
    labels: list[float] | None = None,
    path: str = "/data1/tanseyw/projects/whitej/esm_km_atp/",
):
    """Plot histograms of labels for validation set.

    Parameters
    ----------
    val_df : pd.DataFrame
        Validation dataframe from trainer state log.
    bool_orig : bool
        If True, plot labels in original scale.
    labels : list[float] | None
        List of labels for original scale.

    Returns
    -------
    None
    """
    list_fold = val_df["fold"].unique().tolist()
    list_replace = [f"Fold: {i}\n(n = {sum(val_df["fold"] == i)})" for i in list_fold]
    val_df["fold_label"] = val_df["fold"].map(dict(zip(list_fold, list_replace)))

    if bool_orig and labels is not None:
        val_df["orig_label"] = invert_zscore(val_df["label"], labels)
        val_df["orig_label"] = val_df["orig_label"].apply(lambda x: 10**x)

    g = sns.FacetGrid(val_df, col="fold_label", hue="fold")

    if bool_orig:
        g.map(plt.hist, "orig_label")
        g.set_axis_labels("Km, ATP", "Frequency")
        y, x, _ = plt.hist(val_df["orig_label"])
    else:
        g.map(plt.hist, "label")
        g.set_axis_labels("z-score, $log_{10}$Km, ATP", "Frequency")
        y, x, _ = plt.hist(val_df["label"])

    for idx, ax in enumerate(g.axes.flat):
        loc = val_df.loc[val_df["fold"] == idx + 1, "label"].mean()
        ax.axvline(loc, color="r", linestyle="dashed", linewidth=1)
        ax.text(
            loc + (x.max() - x.min()) * 0.1,
            y.max() * 0.9,
            "Mean: " + str(round(loc, 2)),
            color="r",
        )

    g.set_titles("{col_name}")

    if bool_orig:
        plt.savefig(
            os.path.join(path, "images/val_label_hist_orig.png"), bbox_inches="tight"
        )
    else:
        plt.savefig(
            os.path.join(path, "images/val_label_hist_zscore.png"), bbox_inches="tight"
        )
