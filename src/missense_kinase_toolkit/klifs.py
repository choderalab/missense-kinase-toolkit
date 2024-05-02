import logging

from bravado.client import SwaggerClient


logger = logging.getLogger(__name__)


class KLIFS():
    """Class to interact with the KLIFS API."""
    def __init__(self):
        """Initialize KLIFS Class object.
        
        Attributes
        ----------
        url : str
            KLIFS API URL
        _klifs : bravado.client.SwaggerClient
            KLIFS API object

        """
        self.url = "https://dev.klifs.net/swagger_v2/swagger.json"
        self._klifs = self.get_klifs_api()

    def get_klifs_api(self):
        """Get KLIFS API as bravado.client.SwaggerClient object.

        Returns
        -------
        bravado.client.SwaggerClient
            KLIFS API object

        """
        klifs_api = SwaggerClient.from_url(
            self.url,
            config={
            "validate_requests": False,
            "validate_responses": False,
            "validate_swagger_spec": False
            }
        )
        return klifs_api

    def get_url(self):
        """Get KLIFS API URL."""
        return self.url

    def get_klifs(self):
        """Get KLIFS API object."""
        return self._klifs


class KinaseInfo(KLIFS):
    """Class to get information about a kinase from KLIFS.
    """
    def __init__(
        self,
        kinase_name: str,
        species: str = "Human",
    ) -> None:
        """Initialize KinaseInfo Class object.

        Parameters
        ----------
        kinase_name : str
            Name of the kinase
        species : str
            Species of the kinase; default "Human" but can also be "Mouse"
        
        Attributes
        ----------
        kinase_name : str
            Name of the kinase searched
        species : str
            Species of the kinase
        _kinase_info : dict[str, str | int | None]
            KLIFS API object

        """
        super().__init__()
        self.kinase_name = kinase_name
        self.species = species
        self._kinase_info = self.get_kinase_info()

    def get_kinase_info(
        self
    ) -> dict[str, str | int | None]:
        """Get information about a kinase from KLIFS.
        
        Returns
        -------
        dict[str, str | int | None]
            Dictionary with information about the kinase

        """
        try:
            kinase_info = (
                self._klifs.Information.get_kinase_ID(
                kinase_name=[self.kinase_name],
                species=self.species)
            .response()
            .result[0]
            )

            list_key = dir(kinase_info)
            list_val = [getattr(kinase_info, key) for key in list_key]

            dict_kinase_info = dict(zip(list_key, list_val))

        except Exception as e:
            print(e)
            list_key = [
                'family',
                'full_name',
                'gene_name',
                'group',
                'iuphar',
                'kinase_ID',
                'name',
                'pocket',
                'species',
                'subfamily',
                'uniprot'
                ]
            dict_kinase_info = dict(zip(list_key, [None]*len(list_key)))

        return dict_kinase_info

    def get_kinase_name(self):
        """Get name of the kinase."""
        return self.kinase_name

    def get_species(self):
        """Get species of the kinase."""
        return self.species
