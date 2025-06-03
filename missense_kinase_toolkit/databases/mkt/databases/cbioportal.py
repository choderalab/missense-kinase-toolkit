import logging

import pandas as pd
from bravado.client import SwaggerClient
from mkt.databases.api_schema import APIKeySwaggerClient
from mkt.databases.config import get_cbioportal_instance, maybe_get_cbioportal_token
from mkt.databases.io_utils import parse_iterabc2dataframe, save_dataframe_to_csv

# from tqdm import tqdm

logger = logging.getLogger(__name__)


class cBioPortal(APIKeySwaggerClient):
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
        _cbioportal : bravado.client.SwaggerClient | None
            cBioPortal API object (post-init)

        """
        self.instance = get_cbioportal_instance()
        self.url: str = f"https://{self.instance}/api/v2/api-docs"
        self._cbioportal = self.query_api()

    def maybe_get_token(self):
        return maybe_get_cbioportal_token()

    def query_api(self):
        http_client = self.set_api_key()

        cbioportal_api = SwaggerClient.from_url(
            self.url,
            http_client=http_client,
            config={
                "validate_requests": False,
                "validate_responses": False,
                "validate_swagger_spec": False,
            },
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

        Parameters
        ----------
        study_id : str
            cBioPortal study ID

        Attributes
        ----------
        study_id : str
            cBioPortal study ID; default: `msk_impact_2017` (Zehir, 2017)
        _mutations : list | None
            List of cBioPortal mutations; None if study not found (post-init)

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
            # TODO: add incremental error handling beyond missing study
            muts = self._cbioportal.Mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(
                molecularProfileId=f"{self.study_id}_mutations",
                sampleListId=f"{self.study_id}_all",
                projection="DETAILED",
            ).result()
        else:
            logging.error(
                f"Study {self.study_id} not found in cBioPortal instance {self.instance}"
            )

        return muts

    def get_cbioportal_cohort_mutations(self, bool_save=False) -> None | pd.DataFrame:
        """Get cBioPortal cohort mutations and optionally save as a CSV file.

        Parameters
        ----------
        bool_save : bool
            Save cBioPortal cohort mutations as a CSV file if True

        Notes
        -----
            The CSV file will be saved in the output directory specified in the configuration file.
            As the "gene" ABC object is nested within the "mutation" ABC object,
                the two dataframes are parsed and concatenated.

        """
        df_muts = parse_iterabc2dataframe(self._mutations)

        # extract columns that are ABC objects
        list_abc_cols = df_muts.columns[
            df_muts.map(lambda x: type(x).__module__ == "abc").sum() == df_muts.shape[0]
        ].tolist()

        df_combo = df_muts.copy()
        if len(list_abc_cols) > 0:
            # parse the ABC object cols and concatenate with main dataframe
            for col in list_abc_cols:
                df_combo = pd.concat(
                    [df_combo, parse_iterabc2dataframe(df_muts[col], str_prefix=col)],
                    axis=1,
                ).drop([col], axis=1)

        if bool_save:
            filename = f"{self.study_id}_mutations.csv"
            save_dataframe_to_csv(df_combo, filename)
        else:
            return df_combo

    # def get_kinase_mutations(self, bool_save=False) -> None | pd.DataFrame:
    #     """Get cBioPortal kinase mutations and optionally save as a CSV file.

    #     Parameters
    #     ----------
    #     bool_save : bool
    #         Save cBioPortal kinase mutations as a CSV file if True

    #     Returns
    #     -------
    #     pd.DataFrame | None
    #         DataFrame of kinase mutations if successful, otherwise None

    #     """
    #     from mkt.databases.hgnc import hgnc

    #     df_muts = self.get_cbioportal_cohort_mutations(bool_save=False)

    #     # filter for single amino acid missense mutations
    #     df_muts_missense = df_muts.loc[
    #         df_muts["mutationType"] == "Missense_Mutation", :].reset_index(drop=True)
    #     df_muts_missense = df_muts_missense.loc[df_muts_missense["proteinChange"].apply(
    #         lambda x: type(self.try_except_middle_int(x)) is int), :].reset_index(drop=True)

    #     # extract the HGNC gene names from the mutations
    #     set_hgnc = set(df_chord_missense["hugoGeneSymbol"].tolist())
    #     dict_hgnc2uniprot = dict.fromkeys(set_hgnc)
    #     list_err = []
    #     for hgnc_name in tqdm(set_hgnc, desc="Querying HGNC..."):
    #         temp = hgnc.HGNC(input_symbol_or_id=hgnc_name)
    #         try:
    #             uniprot_id = temp.maybe_get_info_from_hgnc_fetch(
    #                 list_to_extract = ["uniprot_ids"]
    #             )["uniprot_ids"][0][0]
    #         except Exception as e:
    #             logging.error(f"Error retrieving Uniprot ID for {hgnc_name}: {e}")
    #             list_err.append(hgnc_name)
    #         dict_hgnc2uniprot[hgnc_name] = uniprot_id
    #     print(f"List errors:\n{list_err}")

    #     if df_muts is not None:
    #         df_kinase_muts = df_muts[df_muts["gene"].str.contains("KINASE")]
    #         if bool_save:
    #             filename = f"{self.study_id}_kinase_mutations.csv"
    #             save_dataframe_to_csv(df_kinase_muts, filename)
    #         else:
    #             return df_kinase_muts

    @staticmethod
    def try_except_middle_int(str_in):
        """Try to convert string [1:-1] characters to integer.

        Parameters
        ----------
        str_in : str
            String to convert to integer

        Returns
        -------
        int | None
            Integer if successful, otherwise None

        """

        try:
            return int(str_in[1:-1])
        except ValueError:
            return None

    def get_study_id(self):
        """Get cBioPortal study ID."""
        return self.study_id

    def get_mutations(self):
        """Get cBioPortal mutations."""
        return self._mutations


# TODO: implement clinical annotations class
