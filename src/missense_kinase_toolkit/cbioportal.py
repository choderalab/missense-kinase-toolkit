from __future__ import annotations

import os
import re
import pandas as pd

from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient


CBIOPORTAL_TOKEN_VAR = "CBIOPORTAL_TOKEN"
CBIOPORTAL_INSTANCE_VAR = "CBIOPORTAL_INSTANCE"


def maybe_get_cbioportal_token_from_env() -> str | None:
    """Get the cBioPortal token from the environment

    Returns
    -------
    str | None
        cBioPortal token as string if exists, otherwise None
    """
    try:
        token = os.environ[CBIOPORTAL_TOKEN_VAR]
    except KeyError:
        token = None

    return token


def maybe_get_cbioportal_instance_from_env(
) -> str | None:
    """Get the cBioPortal instance from the environment

    Returns
    -------
    str | None
        cBioPortal instance as string if exists, otherwise None
    """
    try:
        instance = os.environ[CBIOPORTAL_INSTANCE_VAR]
    except KeyError:
        instance = None

    return instance

def get_all_mutations_by_study(
    str_study_id: str
) -> list | None:
    """Get mutations  cBioPortal data

    Parameters
    ----------
    str_studyid : str
        Study ID within cBioPortal instance;
            e.g. MSKCC clinical sequencing cohort is "msk_impact_2017" and MSKCC clinical sequencing cohort is "mskimpact"

    Returns
    -------
    requests.models.Response
        cBioPortal data
    """
    token = maybe_get_cbioportal_token_from_env()

    instance = maybe_get_cbioportal_instance_from_env()
    if instance is not None:
        url = f"https://{instance}/api/v2/api-docs"
    else:
        url = "https://cbioportal.org/api/v2/api-docs"

    if all(v is not None for v in (token, instance)):
        http_client = RequestsClient()
        http_client.set_api_key(
            instance,
            f"Bearer {token}",
            param_name='Authorization',
            param_in='header'
        )
        cbioportal = SwaggerClient.from_url(
            url,
            http_client=http_client,
            config={
                "validate_requests":False,
                "validate_responses":False,
                "validate_swagger_spec": False
            }
        )
    else:
        cbioportal = SwaggerClient.from_url(
            url,
            config={
                "validate_requests":False,
                "validate_responses":False,
                "validate_swagger_spec": False
            }
        )

    studies = cbioportal.Studies.getAllStudiesUsingGET().result()
    study_ids = [study.studyId for study in studies]

    if str_study_id in study_ids:
        #TODO - add error handling
        #TODO - extract multiple studies
        muts = cbioportal.Mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(
            molecularProfileId=f"{str_study_id}_mutations",
            sampleListId=f"{str_study_id}_all",
            projection="DETAILED"
            ).result()
    else:
        raise ValueError(f"Study {str_study_id} not found in cBioPortal instance {instance}")

    return muts


def parse_muts2dataframe(
    list_input: list,
) -> pd.DataFrame:
    """Parse a list of abc.Mutation into a dictionary

    Parameters
    ----------
    input_object : list
        List of abc.Mutation objects

    Returns
    -------
    pd.DataFrame
        Dataframe for the input list of abc.Mutation objects
    """
    dict_output = {}
    list_dir = dir(list_input[0])
    for attr in list_dir:
        list_attr = []
        for entry in list_input:
            try:
                add = int(entry[attr])
            except ValueError:
                add = str(entry[attr])
            list_attr.append(add)
        dict_output[attr] = list_attr
    df = pd.DataFrame.from_dict(dict_output)
    return df


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


muts = get_all_mutations_by_study("mskimpact")
df_muts = parse_muts2dataframe(muts)


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
