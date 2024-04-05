import os
import sys


OUTPUT_DIR_VAR = "OUTPUT_DIR"
CBIOPORTAL_INSTANCE_VAR = "CBIOPORTAL_INSTANCE"
CBIOPORTAL_TOKEN_VAR = "CBIOPORTAL_TOKEN"
REQUEST_CACHE_VAR = "REQUESTS_CACHE"


def set_output_dir(
    val: str
) -> None:
    """Set the output directory in environment variables

    Parameters
    ----------
    val : str
        Output directory path

    Returns
    -------
    None
    """
    os.environ[OUTPUT_DIR_VAR] = val


def get_output_dir(
) -> str | None:
    """Get the output directory from the environment

    Returns
    -------
    str | None
        Output directory path if exists, otherwise None
    """
    try:
        return os.environ[OUTPUT_DIR_VAR]
    except KeyError:
        print("Output directory not found in environment variables. This is necessary to run analysis. Exiting...")
        sys.exit(1)


def set_cbioportal_instance(
    val: str
) -> None:
    """Set the cBioPortal instance in the environment variables

    Parameters
    ----------
    val : str
        cBioPortal instance; e.g., "cbioportal.mskcc.org" for MSKCC or

    Returns
    -------
    None
    """
    os.environ[CBIOPORTAL_INSTANCE_VAR] = val


def get_cbioportal_instance(
) -> str | None:
    """Get the cBioPortal instance from the environment

    Returns
    -------
    str | None
        cBioPortal instance as string if exists, otherwise None
    """
    try:
        return os.environ[CBIOPORTAL_INSTANCE_VAR]
    except KeyError:
        print("cBioPortal isntance not found in environment variables. This is necessary to run analysis. Exiting...")
        sys.exit(1)


def set_cbioportal_token(
    val: str
) -> None:
    """Set the cBioPortal token in the environment variables

    Parameters
    ----------
    val : str
        cBioPortal token

    Returns
    -------
    None
    """
    os.environ[CBIOPORTAL_TOKEN_VAR] = val


def maybe_get_cbioportal_token(
) -> str | None:
    """Get the cBioPortal token from the environment

    Returns
    -------
    str | None
        cBioPortal token as string if exists, otherwise None
    """
    try:
        return os.environ[CBIOPORTAL_TOKEN_VAR]
    except KeyError:
        return None


def set_request_cache(
    val: bool
) -> None:
    """Set the request cache path in environment variables

    Parameters
    ----------
    val : str
        Request cache path

    Returns
    -------
    None
    """
    #TODO: val should be bool but doesn't work with env, fix
    os.environ[REQUEST_CACHE_VAR] = str(val)


def maybe_get_request_cache(
) -> str | None:
    """Get the request cache path from the environment

    Returns
    -------
    str | None
        Request cache path as string if exists, otherwise None
    """
    try:
        return os.environ[REQUEST_CACHE_VAR]
    except KeyError:
        return None
