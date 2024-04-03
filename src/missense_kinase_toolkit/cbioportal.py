#!/usr/bin/env python3

from __future__ import annotations

import os
import pandas as pd
import sys

from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient


CBIOPORTAL_TOKEN_VAR = "CBIOPORTAL_TOKEN"
CBIOPORTAL_INSTANCE_VAR = "CBIOPORTAL_INSTANCE"
DATA_CACHE_DIR = "DATA_CACHE"
CBIOPORTAL_COHORT_VAR = "CBIOPORTAL_COHORT"


def maybe_get_cbioportal_token_from_env(       
) -> str | None:
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


def maybe_get_cbioportal_cohort_from_env(
) -> str | None:
    """Get the cBioPortal instance from the environment

    Returns
    -------
    str | None
        cBioPortal instance as string if exists, otherwise None
    """
    try:
        instance = os.environ[CBIOPORTAL_COHORT_VAR]
    except KeyError:
        print("Cohort not found in environment variables. This is necessary to run analysis. Exiting...")
        sys.exit(1)

    return instance


def get_all_mutations_by_study(       
) -> list | None:
    """Get mutations  cBioPortal data

    Returns
    -------
    list | None
        cBioPortal data of Abstract Base Classes objects if successful, otherwise None
    """
    token = maybe_get_cbioportal_token_from_env()

    instance = maybe_get_cbioportal_instance_from_env()
    if instance is not None:
        url = f"https://{instance}/api/v2/api-docs"
    else:
        url = "https://cbioportal.org/api/v2/api-docs"

    # Zehir, 2017 MSKCC sequencing cohort is "msk_impact_2017"
    # MSKCC clinical sequencing cohort is "mskimpact"
    study_id = maybe_get_cbioportal_cohort_from_env()

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

    if study_id in study_ids:
        #TODO: add error handling
        #TODO: extract multiple studies
        muts = cbioportal.Mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(
            molecularProfileId=f"{study_id}_mutations",
            sampleListId=f"{study_id}_all",
            projection="DETAILED"
            ).result()
    else:
        raise ValueError(f"Study {study_id} not found in cBioPortal instance {instance}")

    return muts


def parse_iterabc2dataframe(
    list_input: iter,
) -> pd.DataFrame:
    """Parse an iterable containing Abstract Base Classes into a dataframe

    Parameters
    ----------
    input_object : iter
        Iterable of Abstract Base Classes objects

    Returns
    -------
    pd.DataFrame
        Dataframe for the input list of Abstract Base Classes objects
    """
    list_dir = [dir(entry) for entry in list_input]
    set_dir = set([item for sublist in list_dir for item in sublist])
    
    dict_dir = {attr: [] for attr in set_dir}
    for entry in list_input:
        for attr in dict_dir.keys():
            try:
                dict_dir[attr].append(getattr(entry, attr))
            except AttributeError:
                dict_dir[attr].append(None)
    
    df = pd.DataFrame.from_dict(dict_dir)
    
    return df


def save_cbioportal_data_to_csv(
    df: pd.DataFrame, 
) -> None:
    """Save cBioPortal data to a CSV file

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of cBioPortal data

    Returns
    -------
    None
    """
    try:
        path_data = os.environ[DATA_CACHE_DIR]
        if not os.path.exists(path_data):
            os.makedirs(path_data)
        study_id = maybe_get_cbioportal_cohort_from_env()
        df.to_csv(os.path.join(path_data, f"{study_id}_mutations.csv"), index=False)
    except KeyError:
        print("DATA_CACHE not found in environment variables...")


def main():
    muts = get_all_mutations_by_study()
    df_muts = parse_iterabc2dataframe(muts)
    df_genes = parse_iterabc2dataframe(df_muts["gene"])
    df_combo = pd.concat([df_muts, df_genes], axis=1)
    df_combo = df_combo.drop(['gene'], axis=1)
    save_cbioportal_data_to_csv(df_combo)


if __name__ == "__main__":
    main()