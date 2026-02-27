import os
import sys

OUTPUT_DIR_VAR = "OUTPUT_DIR"
"""str: Environment variable for output directory"""


def set_output_dir(val: str) -> None:
    """Set the output directory in environment variables.

    Parameters:
    -----------
    val : str
        Output directory path

    Returns:
    --------
    None

    """
    os.environ[OUTPUT_DIR_VAR] = val


def get_output_dir() -> str | None:
    """Get the output directory from the environment.

    Returns:
    --------
    str | None
        Output directory path if exists, otherwise None
    """
    try:
        return os.environ[OUTPUT_DIR_VAR]
    except KeyError:
        print(
            "Output directory not found in environment variables. This is necessary to run analysis. Exiting..."
        )
        sys.exit(1)
