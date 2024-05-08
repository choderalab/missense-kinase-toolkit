import logging

from bravado.client import SwaggerClient


logger = logging.getLogger(__name__)

# courtesy of OpenCADD
POCKET_KLIFS_REGIONS = [
    (1, "I"),
    (2, "I"),
    (3, "I"),
    (4, "g.l"),
    (5, "g.l"),
    (6, "g.l"),
    (7, "g.l"),
    (8, "g.l"),
    (9, "g.l"),
    (10, "II"),
    (11, "II"),
    (12, "II"),
    (13, "II"),
    (14, "III"),
    (15, "III"),
    (16, "III"),
    (17, "III"),
    (18, "III"),
    (19, "III"),
    (20, "αC"),
    (21, "αC"),
    (22, "αC"),
    (23, "αC"),
    (24, "αC"),
    (25, "αC"),
    (26, "αC"),
    (27, "αC"),
    (28, "αC"),
    (29, "αC"),
    (30, "αC"),
    (31, "b.l"),
    (32, "b.l"),
    (33, "b.l"),
    (34, "b.l"),
    (35, "b.l"),
    (36, "b.l"),
    (37, "b.l"),
    (38, "IV"),
    (39, "IV"),
    (40, "IV"),
    (41, "IV"),
    (42, "V"),
    (43, "V"),
    (44, "V"),
    (45, "GK"),
    (46, "hinge"),
    (47, "hinge"),
    (48, "hinge"),
    (49, "linker"),
    (50, "linker"),
    (51, "linker"),
    (52, "linker"),
    (53, "αD"),
    (54, "αD"),
    (55, "αD"),
    (56, "αD"),
    (57, "αD"),
    (58, "αD"),
    (59, "αD"),
    (60, "αE"),
    (61, "αE"),
    (62, "αE"),
    (63, "αE"),
    (64, "αE"),
    (65, "VI"),
    (66, "VI"),
    (67, "VI"),
    (68, "c.l"),
    (69, "c.l"),
    (70, "c.l"),
    (71, "c.l"),
    (72, "c.l"),
    (73, "c.l"),
    (74, "c.l"),
    (75, "c.l"),
    (76, "VII"),
    (77, "VII"),
    (78, "VII"),
    (79, "VIII"),
    (80, "xDFG"),
    (81, "xDFG"),
    (82, "xDFG"),
    (83, "xDFG"),
    (84, "a.l"),
    (85, "a.l"),
]
"""list[tuple[int, str]]: Mapping KLIFS pocket ID to region"""

POCKET_KLIFS_REGION_COLORS = {
    "I" : "khaki",
    "g.l": "green",
    "II": "khaki",
    "III": "khaki",
    "αC": "red",
    "b.l": "green",
    "IV": "khaki",
    "V": "khaki",
    "GK": "orange",
    "hinge": "magenta",
    "linker": "cyan",
    "αD": "red",
    "αE": "red",
    "VI": "khaki",
    "c.l": "darkorange",
    "VII": "khaki",
    "VIII": "khaki",
    "xDFG": "cornflowerblue",
    "a.l": "cornflowerblue",
}
"""dict[str, str]: Mapping KLIFS pocket region to color"""

class KLIFS():
    """Class to interact with the KLIFS API."""
    def __init__(self):
        """Initialize KLIFS Class object.

        Upon initialization, KLIFS API is queried.

        Attributes
        ----------
        url : str
            KLIFS API URL
        _klifs : bravado.client.SwaggerClient
            KLIFS API object

        """
        self.url = "https://dev.klifs.net/swagger_v2/swagger.json"
        self._klifs = self.query_klifs_api()

    def query_klifs_api(self):
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

        Upon initialization, KLIFS API is queried and kinase information for specificied kinase is retrieved.

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
        self._kinase_info = self.query_kinase_info()

    def query_kinase_info(
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
            print(f"Error in query_kinase_info for {self.kinase_name}:")
            print(e)
            list_key = [
                "family",
                "full_name",
                "gene_name",
                "group",
                "iuphar",
                "kinase_ID",
                "name",
                "pocket",
                "species",
                "subfamily",
                "uniprot"
                ]
            dict_kinase_info = dict(zip(list_key, [None]*len(list_key)))
            dict_kinase_info["name"] = self.kinase_name

        return dict_kinase_info

    def get_kinase_name(self):
        """Get name of the kinase."""
        return self.kinase_name

    def get_species(self):
        """Get species of the kinase."""
        return self.species

    def get_kinase_info(self):
        """Get information about the kinase."""
        return self._kinase_info
