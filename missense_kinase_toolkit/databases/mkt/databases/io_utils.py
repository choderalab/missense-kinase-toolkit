import logging
import os
import tarfile

import git
import pandas as pd
from mkt.databases.config import OUTPUT_DIR_VAR

logger = logging.getLogger(__name__)


def check_outdir_exists() -> str:
    """Check if OUTPUT_DIR in environmental variables and create directory if doesn't exist.

    Returns
    -------
    str | None
        Path to OUTPUT_DIR
    """
    try:
        path_data = os.environ[OUTPUT_DIR_VAR]
        if not os.path.exists(path_data):
            os.makedirs(path_data)
    except KeyError:
        logger.error(f"{OUTPUT_DIR_VAR} not found in environment variables...")

    return path_data


def convert_str2list(input_str: str) -> list[str]:
    """Convert a string to a list.

    Parameters
    ----------
    str : str
        String to convert to list

    Returns
    -------
    list[str]
        List of strings

    """
    list_str = input_str.split(",")
    list_str = [str_in.strip() for str_in in list_str]
    return list_str


def load_csv_to_dataframe(
    filename: str,
) -> None:
    """Load a CSV file as a dataframe

    Parameters
    ----------
    filename : str
        Filename to load (either with or without "csv" suffix)

    Returns
    -------
    df : pd.DataFrame
        Dataframe loaded from CSV file

    """
    filename = filename.replace(".csv", "") + ".csv"
    path_data = check_outdir_exists()
    try:
        df = pd.read_csv(os.path.join(path_data, filename))
    except FileNotFoundError:
        logger.info(f"File {filename} not found in {path_data}...")
    return df


def save_dataframe_to_csv(
    df: pd.DataFrame,
    filename: str,
) -> None:
    """Save a dataframe to a CSV file.

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
    """Use glob to find csv files to concatenate.

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
        logger.info(f"No files matching {str_find} found in {path_data}...")

    # TODO: implement remove duplicates

    return df_combo


def parse_iterabc2dataframe(
    input_object: iter,
) -> pd.DataFrame:
    """Parse an iterable containing Abstract Base Classes into a dataframe.

    Parameters
    ----------
    input_object : iter
        Iterable of Abstract Base Classes objects

    Returns
    -------
    pd.DataFrame
        Dataframe for the input list of Abstract Base Classes objects

    """
    list_dir = [dir(entry) for entry in input_object]
    set_dir = {item for sublist in list_dir for item in sublist}

    dict_dir = {attr: [] for attr in set_dir}
    for entry in input_object:
        for attr in dict_dir.keys():
            try:
                dict_dir[attr].append(getattr(entry, attr))
            except AttributeError:
                dict_dir[attr].append(None)

    df = pd.DataFrame.from_dict(dict_dir)
    df = df[sorted(df.columns.to_list())]

    return df


def get_repo_root():
    """Get the root of the git repository.

    Returns
    -------
    str
        Path to the root of the git repository; if not found, return current directory
    """
    try:
        repo = git.Repo(".", search_parent_directories=True)
        return repo.working_tree_dir
    except git.InvalidGitRepositoryError:
        logger.info("Not a git repository; using current directory as root...")
        return "."


def create_tar_without_metadata(
    path_source: str,
    filename_tar: str,
) -> None:
    """Create a tar file without metadata.

    Parameters
    ----------
    path_source : str
        Path to the source directory to be tarred
    filename_tar : str
        Path and filename to the save the tar file

    Returns
    -------
    None

    """
    # check if the source directory exists
    if not os.path.exists(path_source):
        logging.error(f"Source directory {path_source} does not exist.")
    # check if the source directory is a directory
    if not os.path.isdir(path_source):
        logging.error(f"Source path {path_source} is not a directory.")
    # check if the output tar file already exists
    if os.path.exists(filename_tar):
        logging.error(f"Output tar file {filename_tar} already exists.")

    with tarfile.open(filename_tar, "w:gz") as tar:
        for root, _, files in os.walk(path_source):
            for file in files:
                file_path = os.path.join(root, file)
                if not file.startswith("._"):
                    tar.add(file_path, arcname=os.path.relpath(file_path, path_source))
