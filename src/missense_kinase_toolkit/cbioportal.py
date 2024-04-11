#!/usr/bin/env python3

import logging
import pandas as pd

from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
# from pydantic import BaseModel
# from typing import ClassVar

from missense_kinase_toolkit import config, io_utils, utils_requests


logger = logging.getLogger(__name__)


def parse_iterabc2dataframe(
    input_object: iter,
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
    list_dir = [dir(entry) for entry in input_object]
    set_dir = {item for sublist in list_dir for item in sublist}

    dict_dir = {attr: [] for attr in set_dir}
    for entry in input_object:
        for attr in dict_dir.keys():
            try:
                dict_dir[attr].append(getattr(entry, attr))
            except AttributeError:
                dict_dir[attr].append(None)

    df = pd.DataFrame.from_dict(dict_dir)
    df = df[sorted(df.columns.to_list())]

    return df


class cBioPortal():
    # instance: ClassVar[str] = f"{config.get_cbioportal_instance()}"
    # url: ClassVar[str] = f"https://{instance}/api/v2/api-docs"
    # cbioportal: ClassVar[SwaggerClient | None] = None

    def __init__(self):
        self.instance = config.get_cbioportal_instance()
        self.url = f"https://{self.instance}/api/v2/api-docs"
        self._cbioportal = self.get_cbioportal_api()

    def _set_api_key(self):
        token = config.maybe_get_cbioportal_token()
        http_client = RequestsClient()
        if token is not None:
            http_client.set_api_key(
                self.instance,
                f"Bearer {token}",
                param_name="Authorization",
                param_in="header"
            )
        else:
            print("No API token provided")
        return http_client

    def get_cbioportal_api(self):
        http_client = self._set_api_key()

        cbioportal_api = SwaggerClient.from_url(
            self.url,
            http_client=http_client,
            config={
                "validate_requests": False,
                "validate_responses": False,
                "validate_swagger_spec": False
            }
        )

        # response = cbioportal_api.Studies.getAllStudiesUsingGET().response().incoming_response
        # logger.error(utils_requests.print_status_code(response.status_code))

        return cbioportal_api

    def get_instance(self):
        return self.instance

    def get_url(self):
        return self.url

    def get_cbioportal(self):
        return self._cbioportal


class Mutations(cBioPortal):
    def __init__(
        self,
        study_id: str,
    ) -> None:
        super().__init__()
        self.study_id = study_id
        self._mutations = self.get_all_mutations_by_study()

    def get_all_mutations_by_study(
        self,
    ) -> list | None:
        """Get mutations  cBioPortal data

        Returns
        -------
        list | None
            cBioPortal data of Abstract Base Classes objects if successful, otherwise None
        """
        studies = self._cbioportal.Studies.getAllStudiesUsingGET().result()
        study_ids = [study.studyId for study in studies]

        if self.study_id in study_ids:
            #TODO: add incremental error handling beyond missing study
            muts = self._cbioportal.Mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(
                molecularProfileId=f"{self.study_id}_mutations",
                sampleListId=f"{self.study_id}_all",
                projection="DETAILED"
                ).result()
        else:
            logging.error(f"Study {self.study_id} not found in cBioPortal instance {self.instance}")

        return muts

    def get_and_save_cbioportal_cohort_mutations(
        self,
    ) -> None:
        df_muts = parse_iterabc2dataframe(self._mutations)
        df_genes = parse_iterabc2dataframe(df_muts["gene"])
        df_combo = pd.concat([df_muts, df_genes], axis=1)
        df_combo = df_combo.drop(["gene"], axis=1)

        filename = f"{self.study_id}_mutations.csv"

        io_utils.save_dataframe_to_csv(df_combo, filename)

    def get_study_id(self):
        return self.study_id

    def get_mutations(self):
        return self._mutations
