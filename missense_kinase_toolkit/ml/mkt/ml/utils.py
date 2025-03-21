import torch


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
    except:
        return False