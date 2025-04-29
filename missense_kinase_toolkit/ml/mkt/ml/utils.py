import logging

import git
import torch

logger = logging.getLogger(__name__)


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


def return_device():
    """Return device

    Returns:
    --------
    str
        Device; either "cuda" or "cpu"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def try_except_string_in_list(str_in, list_in):
    """Check if entry is in list.

    Params:
    -------
    str_in: str
        String to check
    list_in: list
        List to check against

    Returns:
    --------
    bool
        Whether string is in list
    """
    try:
        return str_in in list_in
    except Exception as e:
        logging.info(
            f"Error in try_except_string_in_list: {e}. str_in: {str_in}, list_in: {list_in}"
        )
        return False


def set_seed(
    seed: int = 42,
    bool_deterministic: bool = False,
) -> None:
    """Set all seeds to make results reproducible.

    Parameters:
    -----------
    seed: int
        Seed value for random number generators
    bool_deterministic: bool
        If True, set PyTorch to deterministic mode

    """
    # python random seed
    import random
    random.seed(seed)
    
    # numpy random seed
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        logger.warning("NumPy is not installed. Skipping NumPy seed setting...")
    
    # pytorch random seeds
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU setups

        if bool_deterministic:
            # make PyTorch deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.warning("PyTorch is not installed. Skipping PyTorch seed setting...")
    
    # transformers seed
    try:
        import transformers
        transformers.set_seed(seed, deterministic=bool_deterministic)
    except ImportError:
        logger.warning("Transformers is not installed. Skipping transformers seed setting...")
    
    # set environment variable for python hash seed
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"All random seeds have been set to {seed}")
    