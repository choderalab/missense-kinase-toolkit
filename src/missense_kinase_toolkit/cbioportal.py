from __future__ import annotations

import re

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import os

from missense_kinase_toolkit import requests_wrapper


CBIOPORTAL_TOKEN_VAR = "CBIOPORTAL_TOKEN"


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


def get_cbioportal_token(

) -> str:
    """Get the cBioPortal token from the environment

    Returns
    -------
    str
        cBioPortal token
    """
    token = os.environ[CBIOPORTAL_TOKEN_VAR]

    return token


def get_cbioprotal_data() -> requests.models.Response:
    """Get the cBioPortal data

    Returns
    -------
    requests.models.Response
        cBioPortal data
    """
    token = get_cbioportal_token()
    url = "https://cbioportal.mskcc.org/api/v2/api-docs"

    headers =  {"Content-Type":"application/json", "Authorization": f"Bearer {token}"}

    res = requests_wrapper.get_cached_session().get(
        url, headers=headers
    )

    res.json().keys()
    res.json()["paths"].keys()
    res.json()["paths"]['/cancer-types']
    res.json()["paths"]["/molecular-profiles/{molecularProfileId}/molecular-data/fetch"]
    dir(res)
    res.Studies

    headers = {"X-Auth-Token": token}
    response = requests.get(url, headers=headers)


    url = 'https://api_url'
    headers = {'Accept': 'application/json'}
    auth = HTTPBasicAuth('apikey', '1234abcd')
    files = {'file': open('filename', 'rb')}

    req = requests.get(url, headers=headers, auth=auth, files=files)

    return response


# from bravado.client import SwaggerClient
# from bravado.requests_client import RequestsClient

# http_client = RequestsClient()
# http_client.set_api_key(
#     'genie.cbioportal.org', 'Bearer <TOKEN>',
#     param_name='Authorization', param_in='header'
# )

# cbioportal = SwaggerClient.from_url('https://genie.cbioportal.org/api/v2/api-docs',
#                                     http_client=http_client,
#                                     config={"validate_requests":False,
#                                             "validate_responses":False,
#                                             "validate_swagger_spec": False}
# )
