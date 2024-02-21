from __future__ import annotations

import re

import requests
import pandas as pd


def create_setlist(
    input_object: requests.models.Response,
    attr: str,
) -> tuple[list, set]:
    """Create a list and set of unique values from a response object

    Parameters
    ----------
    input_object : requests.models.Response
        Response object from a request
    attr : str
        Attribute to extract from the response object

    Returns
    -------
    tuple[list, set]
        List and set of unique values from the response object
    """
    list_output = []
    set_output = set()

    for entry in input_object:
        list_output.append(entry[attr])
        set_output.add(entry[attr])

    return list_output, set_output


def print_counts(
    list_input: list,
) -> None:
    """Print the counts of unique values in a list

    Parameters
    ----------
    list_input : list
        List of values to count

    Returns
    -------
    None
    """
    for Unique in set(list_input):
        n = list_input.count(Unique)
        print(f"{Unique:<15} \t {n:>10}")


def parse_obj2dict(
    input_object: requests.models.Response,
) -> dict:
    """Parse a response object into a dictionary

    Parameters
    ----------
    input_object : requests.models.Response
        Response object from a request

    Returns
    -------
    dict
        Dictionary of values from the response object
    """
    dict_output = {}

    list_dir = dir(input_object[0])

    for attr in list_dir:
        list_attr = []
        for entry in input_object:
            try:
                add = int(entry[attr])
            except ValueError:
                add = str(entry[attr])
            list_attr.append(add)
        dict_output[attr] = list_attr

    return dict_output


def parse_series2dict(
    series: pd.Series,
    strwrap: None | str = None,
    delim1: None | str = None,
    delim2: None | str = None,
) -> dict:
    """Parse a series into a dictionary

    Parameters
    ----------
    series : pd.Series
        Series to parse
    strwrap : None | str
        Regular expression to wrap the values in the series
    delim1 : None | str
        Delimiter to split the values in the series
    delim2 : None | str
        Delimiter to split the values in the series

    Returns
    -------
    dict
        Dictionary of values from the series
    """
    if strwrap is None:
        strwrap = r"Gene\((.*)\)"
    if delim1 is None:
        delim1 = ", "
    if delim2 is None:
        delim2 = "="

    list_temp = series.apply(
        lambda x: re.search(strwrap, str(x)).group(1).split(delim1)
    )
    list_keys = [gene.split(delim2)[0] for gene in list_temp[0]]
    dict_out = {key: [] for key in list_keys}

    for row in list_temp:
        list_row = [col.split(delim2)[1] for col in row]
        for idx, col in enumerate(list_row):
            dict_out[list_keys[idx]].append(col)

    return dict_out


def calc_vaf(
    dataframe,
    alt: None | str = None,
    ref: None | str = None,
):
    if alt is None:
        alt = "tumorAltCount"
    if ref is None:
        ref = "tumorRefCount"

    vaf = dataframe[alt] / (dataframe[alt] + dataframe[ref])

    return vaf
