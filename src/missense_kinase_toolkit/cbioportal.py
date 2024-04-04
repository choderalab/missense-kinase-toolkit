#!/usr/bin/env python3

import os
import pandas as pd

from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient

from missense_kinase_toolkit import config, io_utils


# OUTPUT_DIR_VAR = "OUTPUT_DIR"
# CBIOPORTAL_INSTANCE_VAR = "CBIOPORTAL_INSTANCE"
# CBIOPORTAL_TOKEN_VAR = "CBIOPORTAL_TOKEN"
# REQUEST_CACHE_VAR = "REQUESTS_CACHE"
# CBIOPORTAL_COHORT_VAR = "CBIOPORTAL_COHORT"


def get_all_mutations_by_study(
    study_id: str,
) -> list | None:
    """Get mutations  cBioPortal data

    Returns
    -------
    list | None
        cBioPortal data of Abstract Base Classes objects if successful, otherwise None
    """
    # instance = os.environ[CBIOPORTAL_INSTANCE_VAR]
    instance = config.get_cbioportal_instance()
    url = f"https://{instance}/api/v2/api-docs"
    # token = os.environ[CBIOPORTAL_TOKEN_VAR]
    token = config.maybe_get_cbioportal_token()
    # study_id = os.environ[CBIOPORTAL_COHORT_VAR]

    # print(token)
    # print(url)
    # print(study_id)

    if token is not None:
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
    set_dir = {item for sublist in list_dir for item in sublist}

    dict_dir = {attr: [] for attr in set_dir}
    for entry in list_input:
        for attr in dict_dir.keys():
            try:
                dict_dir[attr].append(getattr(entry, attr))
            except AttributeError:
                dict_dir[attr].append(None)

    df = pd.DataFrame.from_dict(dict_dir)

    return df


# def save_cbioportal_data_to_csv(
#     df: pd.DataFrame,
#     study_id: str,
# ) -> None:
#     """Save cBioPortal data to a CSV file

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Dataframe of cBioPortal data
#     study_id : str
#         cBioPortal study ID

#     Returns
#     -------
#     None
#     """
#     try:
#         # path_data = os.environ[OUTPUT_DIR_VAR]
#         path_data = config.get_output_dir()
#         if not os.path.exists(path_data):
#             os.makedirs(path_data)
#         # study_id = os.environ[CBIOPORTAL_COHORT_VAR]
#         # study_id = config.get_cbioportal_cohort()
#         df.to_csv(os.path.join(path_data, f"{study_id}_mutations.csv"), index=False)
#     except KeyError:
#         print("OUTPUT_DIR not found in environment variables...")


def get_and_save_cbioportal_cohort(
# def main(
    study_id: str,
) -> None:
    # muts = get_all_mutations_by_study()
    muts = get_all_mutations_by_study(study_id)

    df_muts = parse_iterabc2dataframe(muts)
    df_genes = parse_iterabc2dataframe(df_muts["gene"])
    df_combo = pd.concat([df_muts, df_genes], axis=1)
    df_combo = df_combo.drop(['gene'], axis=1)

    filename = f"{study_id}_mutations.csv"
    io_utils.save_dataframe_to_csv(df_combo, filename)
    # save_cbioportal_data_to_csv(df_combo, study_id)


# if __name__ == "__main__":
#     main()