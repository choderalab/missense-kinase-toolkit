import logging

import numpy as np
from bravado.client import SwaggerClient
from dataclasses import dataclass

from missense_kinase_toolkit.databases.api_schema import SwaggerAPIClient

logger = logging.getLogger(__name__)


# start/end and colors courtesy of OpenCADD
DICT_POCKET_KLIFS_REGIONS = {
    "I": {
        "start": 1, 
        "end": 3,
        "contiguous": True,
        "color": "khaki",
        },
    "g.l": {
        "start": 4, 
        "end": 9,
        "contiguous": True,
        "color": "green",
        },
    "II": {
        "start": 10, 
        "end": 13,
        "contiguous": True,
        "color": "khaki",
        },
    "III": {
        "start": 14, 
        "end": 19,
        "contiguous": False,
        "color": "khaki",
        },
    "αC": {
        "start": 20, 
        "end": 30,
        "contiguous": True,
        "color": "red",
        },
    "b.l": {
        "start": 31, 
        "end": 37,
        "contiguous": True,
        "color": "green",
        },
    "IV": {
        "start": 38, 
        "end": 41,
        "contiguous": False,
        "color": "khaki",
        },
    "V": {
        "start": 42, 
        "end": 44,
        "contiguous": True,
        "color": "khaki",
        },
    "GK": {
        "start": 45, 
        "end": 45,
        "contiguous": True,
        "color": "orange",
        },
    "hinge": {
        "start": 46, 
        "end": 48, 
        "contiguous": True,
        "color": "magenta",
        },
    "linker": {
        "start": 49, 
        "end": 52,
        "contiguous": True,
        "color": "cyan",
        },
    "αD": {
        "start": 53, 
        "end": 59, 
        "contiguous": False,
        "color": "red",
        },
    "αE": {
        "start": 60, 
        "end": 64,
        "contiguous": True,
        "color": "red",
        },
    "VI": {
        "start": 65, 
        "end": 67,
        "contiguous": True,
        "color": "khaki",
        },
    "c.l": {
        "start": 68, 
        "end": 75,
        "contiguous": True,
        "color": "darkorange",
    },
    "VII": {
        "start": 76, 
        "end": 78, 
        "contiguous": False,
        "color": "khaki",
        },
    "VIII": {
        "start": 79, 
        "end": 79,
        "contiguous": True,
        "color": "khaki",
        },
    "xDFG": {
        "start": 80, 
        "end": 83,
        "contiguous": True,
        "color": "cornflowerblue",
        },
    "a.l": {
        "start": 84, 
        "end": 85,
        "contiguous": False,
        "color": "cornflowerblue",
        },
}
"""dict[str, dict[str, int | bool | str]]: Mapping KLIFS pocket region to start and end indices, \
    boolean denoting if subsequent regions are contiguous, and colors."""


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
    list_idx : list[int] | None
        List of indices in UniProt sequence where KLIFS region starts

    """
    substring_klifs = str_klifs[idx_start:idx_end]
    substring_klifs_narm = remove_gaps_from_klifs(substring_klifs)
    if len(substring_klifs_narm) == 0:
        list_idx = None
    else:
        list_idx = return_idx_of_substring_in_superstring(
            str_uniprot, 
            substring_klifs_narm
        )
    return substring_klifs, list_idx


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
                    same as list_klifs_substr_actual if no concatenation necessary to find a single exact match
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
            idx_start   = klifs_idx_start,
            idx_end     = klifs_idx_end,
            str_uniprot = string_uniprot,
            str_klifs   = string_klifs,
        )
        list_klifs_substr_actual.append(str_klifs)

        # if None KLIFS all "-"; if multiple KLIFS regions, 
        # concatenate with contiguous regions to identify single match
        if list_substring_idx is not None and len(list_substring_idx) > 1:
            bool_cont = dict_klifs[klifs_region_start]["contiguous"]
            # if contiguous with subsequent, concatenate with susbequent region
            if bool_cont:
                klifs_region_end = list_klifs[klifs_index + 1]
            # if not contiguous with subsequent, concatenate with previous region
            else:
                klifs_region_start = list_klifs[klifs_index - 1]
                # need for offset later
                len_klifs = len(remove_gaps_from_klifs(str_klifs))
            klifs_idx_start, klifs_idx_end = (
                dict_klifs[klifs_region_start]["start"] - 1,
                dict_klifs[klifs_region_end]["end"],
            )
            str_klifs, list_substring_idx = align_klifs_pocket_to_uniprot_seq(
                idx_start   = klifs_idx_start,
                idx_end     = klifs_idx_end,
                str_uniprot = string_uniprot,
                str_klifs   = string_klifs,
            )
            # if concat with previous, offset by length of preceding KLIFS region with gaps removed
            if not bool_cont and \
                list_substring_idx is not None and \
                    len(list_substring_idx) != 0:
                len_offset = len(remove_gaps_from_klifs(str_klifs)) - len_klifs
                list_substring_idx = [i + len_offset for i in list_substring_idx]

        list_klifs_region.append(klifs_region_start + ":" + klifs_region_end)
        list_klifs_substr_match.append(str_klifs)
        list_substring_idxs.append(list_substring_idx)

    #TODO: if no match, try to find match with gaps
    # list_idx_no_match = [idx for idx, i in enumerate(list_substring_idxs)\
    #                       if not i and i is None]
    # for idx in list_idx_no_match:
    #     # extract canonical UniProt region between start and end indices of previous and next regions
    #     try:
    #         idx_start = list_substring_idxs[idx-1] + len(remove_gaps_from_klifs(list_klifs_substr_actual[idx-1]))
    #     except:
    #         idx_start = 0
    #     idx_end = list_substring_idxs[idx+1] + len(remove_gaps_from_klifs(list_klifs_substr_actual[idx+1]))
    #     substr_uniprot = 
    #     substr_klifs = list_klifs_substr_actual[idx]

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


@dataclass
class KLIFSPocket:
    """Dataclass to hold KLIFS pocket alignment information per kinase.

    Attributes
    ----------
    uniprotID : str
        UniProt ID
    hgncName : str
        HGNC name
    uniprotSeq : str
        UniProt canonical sequence
    klifsSeq : str
        KLIFS pocket sequence
    list_klifs_region : list[str]
        List of start and end regions of KLIFS pocket separated by ":"; end region will be the
            same as start region if no concatenation necessary to find a single exact match
    list_klifs_substr_actual : list[str]
        List of substring of KLIFS pocket that maps to the *start region* of the KLIFS pocket
    list_klifs_substr_match : list[str]
        List of the actual substring used to match to the KLIFS pocket for the region(s) provided;
            will be the same as list_klifs_substr_actual if no concatenation necessary to find a single exact match
    list_substring_idxs : list[list[int] | None]
        List of indices in UniProt sequence where KLIFS substring match starts;
            offset by length of preceding KLIFS region with gaps removed

    """

    uniprotID: str
    hgncName: str
    uniprotSeq: str
    klifsSeq: str
    list_klifs_region: list[str]
    list_klifs_substr_actual: list[str]
    list_klifs_substr_match: list[str]
    list_substring_idxs: list[list[int] | None]

    # add post-init method
    def __post_init__(self):
        # self.list_substring_klifs_narm = [
        #     self.remove_gaps_from_klifs(substring_klifs)
        #     for substring_klifs in self.list_klifs_substr_actual
        # ]
        pass

    @staticmethod
    def remove_gaps_from_klifs(
        klifs_string: str
    ) -> str:
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

    @staticmethod
    def return_idx_of_substring_in_superstring(
        superstring: str, 
        substring: str
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
        self,
        idx_start: int,
        idx_end: int,
    ) -> list[int] | None:
        """Align KLIFS region to UniProt canonical Uniprot sequence.

        Parameters
        ----------
        idx_start : int
            Start index of KLIFS region
        idx_end : int
            End index of KLIFS region

        Returns
        -------
        substring_klifs : str
            Substring of KLIFS pocket that maps to indices for the region(s) provided
        list_idx : list[int] | None
            List of indices in UniProt sequence where KLIFS region starts

        """
        substring_klifs = self.klifsSeq[idx_start:idx_end]
        substring_klifs_narm = self.remove_gaps_from_klifs(substring_klifs)
        if len(substring_klifs_narm) == 0:
            list_idx = None
        else:
            list_idx = self.return_idx_of_substring_in_superstring(
                self.uniprotSeq, 
                substring_klifs_narm
            )
        return substring_klifs, list_idx
