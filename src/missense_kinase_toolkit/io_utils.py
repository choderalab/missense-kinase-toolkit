import os
import pandas as pd


DATA_CACHE_DIR = "DATA_CACHE"


def save_dataframe_to_csv(
    df: pd.DataFrame, 
    filename: str,
) -> None:
    """Save a dataframe to a CSV file

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    output_path : str
        Path to save the CSV file

    Returns
    -------
    None
    """
    filename = filename.replace(".csv", "") + ".csv"

    try:
        path_data = os.environ[DATA_CACHE_DIR]
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        df.to_csv(os.path.join(path_data, f"{filename}_mutations.csv"), index=False)
    except KeyError:
        print("DATA_CACHE not found in environment variables...")