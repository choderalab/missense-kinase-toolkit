import logging

import numpy as np
from bravado.client import SwaggerClient

from missense_kinase_toolkit.databases.api_schema import SwaggerAPIClient

logger = logging.getLogger(__name__)


# courtesy of OpenCADD
LIST_POCKET_KLIFS_REGIONS = [
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


def convert_pocket_region_list_to_dict() -> dict[str, dict[str, int]]:
    """Convert list of pocket regions to dictionary.

    Returns
    -------
    dict[str, dict[str, int]]
        Dictionary with pocket regions as keys and start and end indices as values

    """
    list_klifs_region = list({i[1] for i in LIST_POCKET_KLIFS_REGIONS})

    list_klifs_start = []
    list_klifs_end = []
    for region in list_klifs_region:
        list_region = [i[0] for i in LIST_POCKET_KLIFS_REGIONS if i[1] == region]
        start, end = min(list_region), max(list_region)
        list_klifs_start.append(start), list_klifs_end.append(end)

    idx_sort = np.argsort(np.array(list_klifs_start))
    list_klifs_region = list(np.array(list_klifs_region)[idx_sort])
    list_klifs_start = list(np.array(list_klifs_start)[idx_sort])
    list_klifs_end = list(np.array(list_klifs_end)[idx_sort])

    dict_klifs_regions = {
        region: {"start": list_klifs_start[idx], "end": list_klifs_end[idx]}
        for idx, region in enumerate(list_klifs_region)
    }

    return dict_klifs_regions


DICT_POCKET_KLIFS_REGIONS = {
    "I": {"start": 1, "end": 3},
    "g.l": {"start": 4, "end": 9},
    "II": {"start": 10, "end": 13},
    "III": {"start": 14, "end": 19},
    "αC": {"start": 20, "end": 30},
    "b.l": {"start": 31, "end": 37},
    "IV": {"start": 38, "end": 41},
    "V": {"start": 42, "end": 44},
    "GK": {"start": 45, "end": 45},
    "hinge": {"start": 46, "end": 48},
    "linker": {"start": 49, "end": 52},
    "αD": {"start": 53, "end": 59},
    "αE": {"start": 60, "end": 64},
    "VI": {"start": 65, "end": 67},
    "c.l": {"start": 68, "end": 75},
    "VII": {"start": 76, "end": 78},
    "VIII": {"start": 79, "end": 79},
    "xDFG": {"start": 80, "end": 83},
    "a.l": {"start": 84, "end": 85},
}
"""dict[str, dict[str, int]]: Mapping KLIFS pocket region to start and end indices"""


# courtesy of OpenCADD
POCKET_KLIFS_REGION_COLORS = {
    "I": "khaki",
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


def remove_gaps_from_klifs(klifs_string: str) -> str:
    """Remove gaps from KLIFS pocket sequence.

    Parameters
    ----------
    klifs_pocket : str
        KLIFS pocket sequence; can be entire sequence or substring

    Returns
    -------
    klifs_pocket_narm : str
        KLIFS pocket sequence without gaps (i.e., "-" removed)

    """
    klifs_pocket_narm = "".join([i for i in klifs_string if i != "-"])
    return klifs_pocket_narm


def return_idx_of_substring_in_superstring(
    superstring: str, substring: str
) -> list[int] | None:
    """

    Parameters
    ----------
    superstring : str
        String in which to find substring index
    substring : str
        String in which to find superstring index

    Returns
    -------
    list_out : list[int] | None
        Index where substring begins in superstring; None if substring not in superstring

    """
    list_out = [
        i for i in range(len(superstring)) if superstring.startswith(substring, i)
    ]
    return list_out


def align_klifs_pocket_to_uniprot_seq(
    idx_start: int,
    idx_end: int,
    str_uniprot: str,
    str_klifs: str,
) -> list[int] | None:
    """Align KLIFS region to UniProt canonical Uniprot sequence.

    Parameters
    ----------
    idx_start : int
        Start index of KLIFS region
    idx_end : int
        End index of KLIFS region
    str_uniprot : str
        UniProt canonical sequence
    str_klifs : str
        KLIFS pocket sequence

    Returns
    -------
    substring_klifs : str
        Substring of KLIFS pocket that maps to indices for the region(s) provided
    list_substring_idx : list[int] | None
        List of indices in UniProt sequence where KLIFS region starts

    """
    substring_klifs = str_klifs[idx_start:idx_end]
    substring_klifs_narm = remove_gaps_from_klifs(substring_klifs)
    if len(substring_klifs_narm) == 0:
        list_idx = None
    else:
        list_idx = return_idx_of_substring_in_superstring(
            str_uniprot, substring_klifs_narm
        )
    return substring_klifs, list_idx


# TODO "b.l" algorithm since non-contiguous region
# TODO keep match string separate from KLIFS substring at start region
def iterate_klifs_alignment(
    string_uniprot: str,
    string_klifs: str,
) -> dict[str, list[str | list[int] | None]]:
    """Align KLIFS region to UniProt canonical Uniprot sequence.

    Parameters
    ----------
    string_uniprot : str
        UniProt canonical sequence
    string_klifs : str
        KLIFS pocket sequence

    Returns
    -------
    dict_out : dict[str, list[str | list[int] | None]]
        Dictionary with keys (match part of KLIFSPocket object):
            list_klifs_region : list[str]
                List of start and end regions of KLIFS pocket separated by ":"; end region will be the
                    same as start region if no concatenation necessary to find a single exact match
            list_klifs_substr_actual : list[str]
                List of substring of KLIFS pocket that maps to the *start region* of the KLIFS pocket
            list_klifs_substr_match : list[str]
                List of the actual substring used to match to the KLIFS pocket for the region(s) provided;
                    will be the same as list_klifs_substr_actual if no concatenation necessary to find a single exact match
            list_substring_idxs : list[list[int | None]]
                List of indices in UniProt sequence where KLIFS region starts

    """
    list_klifs_region = []
    list_klifs_substr_actual = []
    list_klifs_substr_match = []
    list_substring_idxs = []

    dict_klifs = DICT_POCKET_KLIFS_REGIONS
    list_klifs = list(dict_klifs.keys())

    for klifs_index, klifs_region in enumerate(list_klifs):
        klifs_region_start, klifs_region_end = klifs_region, klifs_region
        klifs_idx_start, klifs_idx_end = (
            dict_klifs[klifs_region_start]["start"] - 1,
            dict_klifs[klifs_region_end]["end"],
        )

        str_klifs, list_substring_idx = align_klifs_pocket_to_uniprot_seq(
            idx_start=klifs_idx_start,
            idx_end=klifs_idx_end,
            str_uniprot=string_uniprot,
            str_klifs=string_klifs,
        )
        list_klifs_substr_actual.append(str_klifs)

        # will no longer extract a single index here so can process elsewhere
        # if list_substring_idx is None:
        #     substring_idx = None

        if list_substring_idx is not None:
            # TODO: analyze KLIFS PDB structures to determine contiguous regions
            if len(list_substring_idx) > 1:
                # if not last KLIFs region concatenate with susbequent regions
                if klifs_index + 1 < len(list_klifs):
                    klifs_region_end = list_klifs[klifs_index + 1]
                    klifs_idx_end = dict_klifs[klifs_region_end]["end"]
                # if last KLIFs region concatenate with previous region
                else:
                    len_klifs = len(remove_gaps_from_klifs(str_klifs))
                    klifs_region_start = list_klifs[klifs_index - 1]
                    klifs_idx_start = dict_klifs[klifs_region_start]["start"]
                klifs_idx_start, klifs_idx_end = (
                    dict_klifs[klifs_region_start]["start"] - 1,
                    dict_klifs[klifs_region_end]["end"],
                )
                str_klifs, list_substring_idx = align_klifs_pocket_to_uniprot_seq(
                    idx_start=klifs_idx_start,
                    idx_end=klifs_idx_end,
                    str_uniprot=string_uniprot,
                    str_klifs=string_klifs,
                )
                # if last KLIFS region offset by length of preceding KLIFS region with gaps removed
                if klifs_index + 1 == len(list_klifs) and len(list_substring_idx) != 0:
                    len_offset = len(remove_gaps_from_klifs(str_klifs)) - len_klifs
                    list_substring_idx = [i + len_offset for i in list_substring_idx]

        list_klifs_region.append(klifs_region_start + ":" + klifs_region_end)
        list_klifs_substr_match.append(str_klifs)
        list_substring_idxs.append(list_substring_idx)

        # will no longer extract a single index here so can process multiple ways elsewhere
        #     if len(list_substring_idx) != 1:
        #         substring_idx = np.nan
        #     else:
        #         substring_idx = list_substring_idx[0]
        # else:
        #     try:
        #         substring_idx = list_substring_idx[0]
        #     except:
        #         substring_idx = np.nan

    dict_out = {
        "list_klifs_region": list_klifs_region,
        "list_klifs_substr_actual": list_klifs_substr_actual,
        "list_klifs_substr_match": list_klifs_substr_match,
        "list_substring_idxs": list_substring_idxs,
    }

    return dict_out


class KLIFS(SwaggerAPIClient):
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
        self._klifs = self.query_api()

    def query_api(self):
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
                "validate_swagger_spec": False,
            },
        )
        return klifs_api

    def get_url(self):
        """Get KLIFS API URL."""
        return self.url

    def get_klifs(self):
        """Get KLIFS API object."""
        return self._klifs


class KinaseInfo(KLIFS):
    """Class to get information about a kinase from KLIFS."""

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

    def query_kinase_info(self) -> dict[str, str | int | None]:
        """Get information about a kinase from KLIFS.

        Returns
        -------
        dict[str, str | int | None]
            Dictionary with information about the kinase

        """
        try:
            kinase_info = (
                self._klifs.Information.get_kinase_ID(
                    kinase_name=[self.kinase_name], species=self.species
                )
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
                "uniprot",
            ]
            dict_kinase_info = dict(zip(list_key, [None] * len(list_key)))
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
