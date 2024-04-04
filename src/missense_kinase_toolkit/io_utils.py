import os
import pandas as pd


OUTPUT_DIR_VAR = "OUTPUT_DIR"


def save_dataframe_to_csv(
    df: pd.DataFrame,
    filename: str,
) -> None:
    """Save a dataframe to a CSV file

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    filename : str
        Filename to save (either with or without "csv" suffix)


    Returns
    -------
    None
    """
    filename = filename.replace(".csv", "") + ".csv"

    try:
        path_data = os.environ[OUTPUT_DIR_VAR]
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        df.to_csv(os.path.join(path_data, filename), index=False)
    except KeyError:
        print("OUTPUT_DIR not found in environment variables...")
