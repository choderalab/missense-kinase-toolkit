import os
import pandas as pd


OUTPUT_DIR_VAR = "OUTPUT_DIR"


def check_outdir_exists(
) -> str:
    """Check if OUTPUT_DIR in environmental variables and create directory if doesn't exist

    Returns
    -------
    str
    """
    try:
        path_data = os.environ[OUTPUT_DIR_VAR]
        if not os.path.exists(path_data):
            os.makedirs(path_data)
    except KeyError:
        print("OUTPUT_DIR not found in environment variables...")

    return path_data


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
    path_data = check_outdir_exists()
    df.to_csv(os.path.join(path_data, filename), index=False)


def concatenate_csv_files_with_glob(
    str_find: str,
    str_remove: str = "transformed_mutations.csv",
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

    str_find = str_find.replace(".csv", "") + ".csv"
    path_data = check_outdir_exists()
    csv_files = glob.glob(os.path.join(path_data, str_find))
    csv_files = [csv_file for csv_file in csv_files if str_remove not in csv_file]

    df_combo = pd.DataFrame()
    if len(csv_files) > 0:
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, low_memory=False)
            df_combo = pd.concat([df_combo, df])
    else:
        print(f"No files matching {str_find} found in {path_data}...")

    #TODO: implement remove duplicates

    return df_combo
