from typing import Any

import numpy as np
import pandas as pd


def try_except_split_concat_str(
    str_in: str,
    idx1: int,
    idx2: int,
    delim: str = "-",
) -> str:
    """
    Split str_in on delim with exception handling.

    Parameters
    ----------
    str_in : str
        Input string
    idx1 : int
        Starting index
    idx2 : int
        Ending index
    delim : str
        Delimiter to split on

    Returns
    -------
    str
        Concatenated string containing the strings split on delim from idx1:idx2

    """
    try:
        str_out = ("").join(
            [str_in.split(delim)[i].upper() for i in range(idx1, idx2 + 1)]
        )
        return str_out
    except (IndexError, AttributeError):
        try:
            str_out = str_in.split(delim)[0]
            return str_out
        except AttributeError:
            str_out = str_in
            return str_out


def create_strsplit_list(
    list_in: list[str],
    idx_start: int = 0,
    idx_end: int = 2,
) -> list[str]:
    """
    Split list or Series of strings on delim with exception handling.

    """
    return [
        [
            try_except_split_concat_str(x, idx_start, i)
            for i in range(idx_start, idx_end + 1)
        ]
        for x in list_in
    ]


def try_except_match_str2dict(
    str_in: str,
    dict_in: dict[str, Any],
    bool_keyout: bool = True,
) -> Any:
    """
    Dictionary match with exception handling.

    Parameters
    ----------
    str_in : str
        Input string
    dict_in : dict[str, Any]
        Dictionary where keys are strings to match with str_in
    bool_keyout : bool
        If true and no match, return string

    Returns
    -------
    Any
        Returns either dictionary value if match, str_in if no match and bool_keyout = True else None

    """
    try:
        return dict_in[str_in]
    except KeyError:
        if bool_keyout:
            return str_in
        else:
            None


def return_list_match_indices(
    str_in: str,
    list_in: list[str | list[str]],
) -> list[int] | list | None:
    """
    Return list of indices where str_in matches entry or entries in list_in.

    Parameters
    ----------
    str_in : str
        Input string to check for matches in list_in
    list_in : list[str | list[str]]
        List of string or list of list of strings to check for str_in match

    Returns
    -------
    list[int] | None
        Returns index of matching entries in list

    """
    if type(list_in[0]) is str:
        list_out = [idx for idx, hgnc in enumerate(list_in) if hgnc.upper() in str_in]
        return list_out
    elif type(list_in[0]) is list:
        list_out = [
            idx
            for idx, list_nest in enumerate(list_in)
            for hgnc in list_nest
            if hgnc.upper() in str_in
        ]
        return list_out
    else:
        print(f"Input type of {type(list_in[0])} cannot be handled.")


def replace_string_using_dict(
    str_in: str,
    dict_in: dict[str, str],
) -> str:
    """
    Replace any partial matches in a string using a dictionary of {string match : string replace}.

    Parameters
    ----------
    str_in : str
        Input string to replace partial matches
    dict_in : dict[str, str]
        Dictionary of {string match : string replace}

    Returns
    -------
    str_out : str
        String with any partial matches replaced

    """
    str_out = str_in
    try:
        for key, val in dict_in.items():
            str_out = str_out.upper().replace(key, val)
        return str_out
    except AttributeError:
        return str_out


def return_list_out(
    list_kinhub_uniprot: list[str | float],
    list_assay_name: list[str],
):
    list_out = [
        return_list_match_indices(x, list_kinhub_uniprot) for x in list_assay_name
    ]
    set_out = [set(i) if i is not np.nan else np.nan for i in list_out]
    list_out = [i[0] if len(j) != 0 else np.nan for i, j in zip(list_out, set_out)]
    return list_out, set_out


def try_except_convert_str2int(str_in: str):
    try:
        return int(str_in)
    except ValueError:
        return str_in


def try_except_substraction(a, b):
    try:
        return b - a
    except TypeError:
        return None


def aggregate_df_by_col_set(
    df_in: pd.DataFrame,
    col_group: str,
) -> pd.DataFrame:
    list_cols = df_in.columns.to_list()
    list_cols.remove(col_group)

    # aggregate rows with the same HGNC Name (e.g., multiple kinase domains like JAK)
    df_in_agg = df_in.groupby([col_group], as_index=False, sort=False).agg(set)

    # join set elements into a single string
    df_in_agg[list_cols] = df_in_agg[list_cols].map(
        lambda x: ", ".join(str(s) for s in x)
    )

    return df_in_agg


def split_on_first_only(str_in, delim):
    list_split = str_in.split(delim)
    str1 = list_split[0]
    str2 = "".join(list_split[1:])
    return str1, str2


def flatten_iterables_in_iterable(data):
    flattened_list = []
    for item in data:
        if isinstance(item, (list, tuple)):
            flattened_list.extend(list(item))
        else:
            flattened_list.append(item)
    return flattened_list
