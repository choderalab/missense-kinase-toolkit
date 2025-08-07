import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

    Parameters
    ----------
    list_in : list[str]
        List of strings to split
    idx_start : int
        Starting index
    idx_end : int
        Ending index

    Returns
    -------
    list[str]
        List of concatenated strings split on delim from idx_start:idx_end

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
    """Return list of indices where str_in matches entry or entries in list_in.

    Parameters
    ----------
    list_kinhub_uniprot : list[str | float]
        List of string or list of list of strings to check for str_in match
    list_assay_name : list[str]
        List of string to check for matches in list_in

    Returns
    -------
    list_out : list[int] | None
        Returns index of matching entries in list

    """
    list_out = [
        return_list_match_indices(x, list_kinhub_uniprot) for x in list_assay_name
    ]
    set_out = [set(i) if i is not np.nan else np.nan for i in list_out]
    list_out = [i[0] if len(j) != 0 else np.nan for i, j in zip(list_out, set_out)]
    return list_out, set_out


def try_except_convert_str2int(str_in: str):
    """Convert string to int with exception handling.

    Parameters
    ----------
    str_in : str
        Input string to convert to int

    Returns
    -------
    int | str
        Returns int if conversion successful, otherwise returns str_in

    """
    try:
        return int(str_in)
    except ValueError:
        return str_in


def try_except_substraction(a, b):
    """Subtract two values with exception handling.

    Parameters
    ----------
    a : Any
        First value to subtract from
    b : Any
        Second value to subtract

    Returns
    -------
    Any
        Returns difference if subtraction successful, otherwise returns None

    """
    try:
        return b - a
    except TypeError:
        return None


def aggregate_df_by_col_set(
    df_in: pd.DataFrame,
    col_group: str,
    bool_str: bool = True,
) -> pd.DataFrame:
    """Aggregate DataFrame by column and convert to set.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame to aggregate
    col_group : str
        Column to group by
    bool_str : bool, optional
        If True, convert set to string, by default True

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with set values

    """
    list_cols = df_in.columns.to_list()
    list_cols.remove(col_group)

    # aggregate rows with the same HGNC Name (e.g., multiple kinase domains like JAK)
    df_in_agg = df_in.groupby([col_group], as_index=False, sort=False).agg(set)

    if bool_str:
        # join set elements into a single string
        df_in_agg[list_cols] = df_in_agg[list_cols].map(
            lambda x: ", ".join(str(s) for s in x)
        )

    return df_in_agg


def split_on_first_only(str_in, delim):
    """Split string on first occurrence of delim.

    Parameters
    ----------
    str_in : str
        Input string to split
    delim : str
        Delimiter to split on

    Returns
    -------
    tuple
        Tuple containing two strings: str1 and str2

    """
    list_split = str_in.split(delim)
    str1 = list_split[0]
    str2 = "".join(list_split[1:])
    return str1, str2


def flatten_iterables_in_iterable(data):
    """Flatten nested lists or tuples into a single list.

    Parameters
    ----------
    data : list or tuple
        Input data to flatten

    Returns
    -------
    list
        Flattened list containing all elements from the input data

    """
    flattened_list = []
    for item in data:
        if isinstance(item, (list, tuple)):
            flattened_list.extend(list(item))
        else:
            flattened_list.append(item)
    return flattened_list


def return_bool_at_index(
    list_in: list,
    list_bool: list,
    bool_return: bool = True,
):
    """Return list of elements from list_in where corresponding element in list_bool is bool_return.

    Parameters
    ----------
    list_in : list
        List of elements to filter.
    list_bool : list
        List of boolean values to filter by.
    bool_return : bool, optional
        Boolean value to filter by, by default True

    Returns
    -------
    list
        List of elements from list_in where corresponding element in list_bool is bool_return.

    """
    return [i for i, j in zip(list_in, list_bool) if j == bool_return]


def convert_input2list(obj_in: Any, bool_empty: bool = False) -> list | None:
    """Convert input to a list if it is not already a list.

    Parameters
    ----------
    obj_in : Any
        Input object to convert to a list.
    bool_empty : bool, optional
        If True, return an empty list if obj_in is None, by default False

    Returns
    -------
    list
        List containing the input if it is not already a list, otherwise returns the input as is.
    """
    if isinstance(obj_in, list):
        return obj_in
    elif isinstance(obj_in, (int, float, str)):
        return [obj_in]
    elif bool_empty and obj_in is None:
        return []
    else:
        logger.error("Input is not a list or convertible to a list.")
        return None


def add_one_hot_encoding_to_dataframe(
    df: pd.DataFrame,
    col_name: str | list[str],
    prefix: str | list[str] | None = None,
    bool_drop: bool = True,
    col_drop: str | list[str] | None = None,
) -> pd.DataFrame:
    """
    Add one-hot encoding for one or more specified columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col_name : str | list[str]
        Column name(s) to apply one-hot encoding.
    prefix : str | list[str] | None, optional
        Prefix(es) for the new columns. If None, uses column names as prefixes.
        If list, must match length of col_name, by default None.
    bool_drop : bool, optional
        If True, drop the original column(s) after encoding, by default True.
    col_drop : str | list[str] | None, optional
        If specified, drop these column(s) after encoding (i.e., to avoid multicollinearity).

    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot encoded columns added.
    """
    col_names = convert_input2list(col_name, bool_empty=False)
    cols_to_drop = convert_input2list(col_drop, bool_empty=True)

    if prefix is None:
        prefixes = [""] * len(col_names)
    elif isinstance(prefix, str):
        prefixes = [prefix] * len(col_names)
    else:
        if len(prefix) != len(col_names):
            raise ValueError(
                f"Length of prefix ({len(prefix)}) must "
                f"match length of col_name ({len(col_names)})"
            )
        prefixes = prefix

    one_hot_dfs = []
    for col, pref in zip(col_names, prefixes):
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame.")
            continue
        one_hot = pd.get_dummies(df[col], prefix=pref)
        one_hot_dfs.append(one_hot)
    if one_hot_dfs:
        combined_one_hot = pd.concat(one_hot_dfs, axis=1)
        for col_to_drop in cols_to_drop:
            if col_to_drop in combined_one_hot.columns:
                combined_one_hot = combined_one_hot.drop(columns=[col_to_drop])
            else:
                logger.warning(
                    f"Column '{col_to_drop}' not found in one-hot encoded columns."
                )
    else:
        combined_one_hot = pd.DataFrame(index=df.index)

    if bool_drop:
        df_base = df.drop(columns=col_names, errors="ignore")
    else:
        df_base = df.copy()

    if not combined_one_hot.empty:
        df_out = pd.concat([df_base, combined_one_hot], axis=1, copy=False)
    else:
        df_out = df_base

    return df_out
