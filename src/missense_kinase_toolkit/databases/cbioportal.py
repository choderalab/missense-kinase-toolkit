import logging
import pandas as pd

from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient

from missense_kinase_toolkit.databases import config, io_utils


logger = logging.getLogger(__name__)


class cBioPortal():
    """Class to interact with the cBioPortal API."""
    def __init__(self):
        """Initialize cBioPortal Class object.

        Upon initialization, cBioPortal API is queried.

        Attributes
        ----------
        instance : str
            cBioPortal instance
        url : str
            cBioPortal API URL
        _cbioportal : bravado.client.SwaggerClient
            cBioPortal API object

        """
        self.instance = config.get_cbioportal_instance()
        self.url = f"https://{self.instance}/api/v2/api-docs"
        self._cbioportal = self.query_cbioportal_api()

    def _set_api_key(self):
        """Set API key for cBioPortal API.

        Returns
        -------
        RequestsClient
            RequestsClient object with API key set

        """
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

    def query_cbioportal_api(
        self
    ) -> SwaggerClient:
        """Queries cBioPortal API for instance as bravado.client.SwaggerClient object.

        Returns
        -------
        bravado.client.SwaggerClient
            cBioPortal API object

        """
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

        return cbioportal_api

    def get_instance(self):
        """Get cBioPortal instance."""
        return self.instance

    def get_url(self):
        """Get cBioPortal API URL."""
        return self.url

    def get_cbioportal(self):
        """Get cBioPortal API object."""
        return self._cbioportal


class Mutations(cBioPortal):
    """Class to get mutations from a cBioPortal study."""
    def __init__(
        self,
        study_id: str,
    ) -> None:
        """Initialize Mutations Class object.

        Upon initialization, cBioPortal API is queried and mutations for specificied study are retrieved.

        Parameters
        ----------
        study_id : str
            cBioPortal study ID

        Attributes
        ----------
        study_id : str
            cBioPortal study ID
        _mutations : list | None
            List of cBioPortal mutations

        """
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

    def get_cbioportal_cohort_mutations(
        self,
        bool_save = False,
    ) -> None:
        """Get and save cBioPortal cohort mutations to a CSV file.

        Notes
        -----
            The CSV file will be saved in the output directory specified in the configuration file.
            As the "gene" ABC object is nested within the "mutation" ABC object, the two dataframes are parsed and concatenated.

        """
        df_muts = io_utils.parse_iterabc2dataframe(self._mutations)
        df_genes = io_utils.parse_iterabc2dataframe(df_muts["gene"])
        df_combo = pd.concat([df_muts, df_genes], axis=1)
        df_combo = df_combo.drop(["gene"], axis=1)

        filename = f"{self.study_id}_mutations.csv"

        if bool_save:
            io_utils.save_dataframe_to_csv(df_combo, filename)
        else:
            return df_combo

    def get_study_id(self):
        """Get cBioPortal study ID."""
        return self.study_id

    def get_mutations(self):
        """Get cBioPortal mutations."""
        return self._mutations
