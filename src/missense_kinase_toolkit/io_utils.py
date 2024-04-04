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
    filename = f"{filename.replace(".csv", "")}.csv"

    try:
        path_data = os.environ[OUTPUT_DIR_VAR]
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        df.to_csv(os.path.join(path_data, filename), index=False)
    except KeyError:
        print("OUTPUT_DIR not found in environment variables...")


def concatenate_csv_files_with_glob(
    str_find: str,
) -> pd.DataFrame:
    """Use glob to find csv files to concatenate

    Parameters
    ----------
    str_find: str
        String to use to find files containing csv files of interest

    Return
    ------
    pd.DataFrame
        Concatenated dataframe
    """
    import glob

    str_find = f"{str_find.replace(".csv", "")}.csv"

    csv_files = glob.glob(str_find)

    df_combo = pd.DataFrame()

    try:
        path_data = os.environ[OUTPUT_DIR_VAR]
        if not os.path.exists(path_data):
            os.makedirs(path_data)
    except KeyError:
        print("OUTPUT_DIR not found in environment variables...")

    try:
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df_combo = pd.concat([df_combo, df])
    except:
        print(f"No files matching {str_find} found in {path_data}...")


    return df_combo
