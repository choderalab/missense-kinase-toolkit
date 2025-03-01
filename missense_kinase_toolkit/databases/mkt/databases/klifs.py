import logging
import re
from dataclasses import dataclass, field
from itertools import chain

import numpy as np
from Bio import Align
from bravado.client import SwaggerClient
from mkt.databases.aligners import BL2UniProtAligner, Kincore2UniProtAligner
from mkt.databases.api_schema import SwaggerAPIClient

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


LIST_KLIFS_REGION = list(
    chain(
        *[
            [f"{key}:{i}" for i in range(val["start"], val["end"] + 1)]
            for key, val in DICT_POCKET_KLIFS_REGIONS.items()
        ]
    )
)
"""list[str]: List of string of all KLIFS pocket regions in format region:idx."""


# αC:b.l - {'BUB1B': 'E'} - need to skip this, seems to be in b.l gap region
LIST_INTER_REGIONS = [
    "II:III",
    "III:αC",
    "IV:V",
    "hinge:linker",
    "αD:αE",
    "αE:VI",
    "VII:VIII",
]
"""list[str]: List of inter-region region gaps that exist given analysis."""


LIST_INTRA_REGIONS = [
    "b.l_intra",
    "linker_intra",
]
"""list[str]: List of intra-region region gaps that exist given analysis."""


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
        search_term: str,
        search_field: str | None = None,
        species: str = "Human",
    ) -> None:
        """Initialize KinaseInfo Class object.

        Upon initialization, KLIFS API is queried and kinase information for specificied kinase is retrieved.

        Parameters
        ----------
        search_term : str
            Search term used to query KLIFS API
        search_field : str | None
            Search field (optional; default: None);
            only used to post-hoc annotate column with search term in case of missing data
        species : str
            Species of the kinase; default "Human" but can also be "Mouse"

        Attributes
        ----------
        search_term : str
            Search term used to query KLIFS API
        search_field : str | None
            Search field (optional; default: None);
            only used to post-hoc annotate column with search term in case of missing data
        species : str
            Species of the kinase
        _kinase_info : dict[str, str | int | None]
            KLIFS API object for search term

        """
        super().__init__()
        self.search_term = search_term
        self.search_field = search_field
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
                    kinase_name=[self.search_term], species=self.species
                )
                .response()
                .result[0]
            )

            list_key = dir(kinase_info)
            list_val = [getattr(kinase_info, key) for key in list_key]

            dict_kinase_info = dict(zip(list_key, list_val))

        except Exception as e:
            print(
                f"Error in query_kinase_info for {self.search_term} (field: {self.search_field}):"
            )
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
            if self.search_field is not None:
                dict_kinase_info[self.search_field] = self.search_term

        return dict_kinase_info

    def get_search_term(self):
        """Get search term used for query."""
        return self.search_term

    def get_search_field(self):
        """Get search field used for query."""
        return self.search_field

    def get_species(self):
        """Get species used for query."""
        return self.species

    def get_kinase_info(self):
        """Get information about the kinase from KLIFS query."""
        return self._kinase_info


@dataclass
class KLIFSPocket:
    """Dataclass to hold KLIFS pocket alignment information per kinase.

    Attributes
    ----------
    uniprotSeq : str
        UniProt canonical sequence
    klifsSeq : str
        KLIFS pocket sequence
    idx_kd : tuple[int | None, int | None]
        Index of kinase domain in UniProt sequence (start, end)
    offset_bool : bool
        If True, offset indices by 1 to match 1-based indexing; default True
    list_klifs_region : list[str]
        List of start and end regions of KLIFS pocket separated by ":"; end region will be the
            same as start region if no concatenation necessary to find a single exact match
    list_klifs_substr_actual : list[str | None]
        List of substring of KLIFS pocket that maps to the *start region* of the KLIFS pocket
    list_klifs_substr_match : list[str | None]
        List of the actual substring used to match to the KLIFS pocket for the region(s) provided;
            will be the same as list_klifs_substr_actual if no concatenation necessary to find a single exact match
    list_substring_idxs : list[list[int] | None]
        List of indices in UniProt sequence where KLIFS substring match starts;
            offset by length of preceding KLIFS region with gaps removed
    list_align : list[str | None] | None
        List of final alignments to UniProt sequence

    """

    uniprotSeq: str
    klifsSeq: str
    idx_kd: tuple[int | None, int | None]
    offset_bool: bool = True
    list_klifs_region: list[str | None] = field(default_factory=list)
    list_klifs_substr_actual: list[str | None] = field(default_factory=list)
    list_klifs_substr_match: list[str | None] = field(default_factory=list)
    list_substring_idxs: list[list[int | None] | None] = field(default_factory=list)
    list_align: list[str | None] | None = None
    KLIFS2UniProtIdx: dict[str, int | None] = field(default_factory=dict)
    KLIFS2UniProtSeq: dict[str, str | None] = field(default_factory=dict)

    def __post_init__(self):
        self.iterate_klifs_alignment()
        self.generate_alignment_list(bool_offset=self.offset_bool)
        if self.list_align is not None:
            self.KLIFS2UniProtIdx.update(dict(zip(LIST_KLIFS_REGION, self.list_align)))
            self.generate_alignment_list_including_gaps()

    @staticmethod
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

    @staticmethod
    def return_idx_of_substring_in_superstring(
        superstring: str,
        substring: str,
    ) -> list[int] | None:
        """Returns the index where substring begins in superstring (does not require -1 offset).

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

    @staticmethod
    def return_idx_of_alignment_match(
        align: Align.PairwiseAlignments,
    ) -> list[int]:
        """Return indices of alignment match.

        Parameters
        ----------
        align : Align.PairwiseAlignments
            Pairwise alignments

        Returns
        -------
        list[int]
            List of indices for alignment match

        """
        # extract target (b.l) and query (UniProt) sequences
        target = align.indices[0]
        query = align.indices[1]

        # where target is aligned, set to 1; where target is not aligned, set to np.nan
        target[target >= 0] = 1
        target = np.where(target == -1, np.nan, target)

        # keep only indices where target is aligned to query
        output = target * query
        output = output[~np.isnan(output)]
        output = [int(i) for i in output.tolist()]

        return output

    def select_correct_alignment(
        self,
        alignments: Align.PairwiseAlignments,
        bool_bl: bool = True,
    ) -> list[int]:
        """Select correct alignment for b.l region.

        Parameters
        ----------
        alignments : Align.PairwiseAlignments
            Pairwise alignments
        bool_bl : bool
            If True, select correct alignment for b.l region; if False, select correct alignment for linker region

        Returns
        -------
        list[int]
            List of indices for correct alignment

        """
        list_alignments = [
            re.findall(r"[A-Z]+", alignment[0, :]) for alignment in alignments
        ]

        if bool_bl:
            # manual review showed 2 matches + gap + 5 matches in b.l region
            list_idx = [
                idx
                for idx, i in enumerate(list_alignments)
                if len(i) == 2 and len(i[0]) == 2
            ]
            region = "b.l"
        else:
            # manual review showed 1 matches + gap + 3 matches in linker region
            list_idx = [
                idx
                for idx, i in enumerate(list_alignments)
                if len(i) == 2 and len(i[0]) == 1
            ]
            region = "linker"

        if len(list_idx) > 1:
            logging.error(
                f"{len(list_idx)} correct alignments found for {region} region\n{list_alignments}"
            )
            return None
        # BUB1B and PIK3R4 have "-" in b.l region so will not obey heuristic in list_idx
        elif len(list_idx) == 0:
            if len(alignments) == 1:
                alignment = alignments[0]
            else:
                logging.error(
                    f"{len(alignments)} non-heuristic alignments found for {region} region\n"
                    f"{[print(i) for i in alignments]}"
                )
                return None
        else:
            alignment = alignments[list_idx[0]]

        return self.return_idx_of_alignment_match(alignment)

    def align_klifs_pocket_to_uniprot_seq(
        self,
        idx_start: int,
        idx_end: int,
    ) -> tuple[str, list[int] | None]:
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
                self.uniprotSeq, substring_klifs_narm
            )
        return substring_klifs, list_idx

    def find_start_or_end_idx_recursively(
        self,
        idx_in: int,
        bool_start: bool = True,
    ) -> int:
        """Find the start or end indices in UniProt canonical sequence of flanking KLIFS regions recursively.

        Parameters
        ----------
        idx_in : int
            Index of KLIFS region (e.g., I is 0, g.l is 1, etc.)
        bool_start : bool
            If True, find start index (default); if False, find end index
        """
        # if looking for preceding region, start at idx_in - 1
        if bool_start:
            # if first region
            if idx_in == 0:
                # if KD start is None, return 0
                if self.idx_kd[0] is None:
                    return 0
                # if KD start is provided, return KD start
                else:
                    return self.idx_kd[0]
            idx_temp = self.list_substring_idxs[idx_in - 1]
            str_temp = self.list_klifs_substr_actual[idx_in - 1]
            if idx_temp is not None and len(idx_temp) == 1:
                idx_out = idx_temp[0] + len(self.remove_gaps_from_klifs(str_temp))
            else:
                idx_out = self.find_start_or_end_idx_recursively(
                    idx_in - 1, bool_start=True
                )
        # if looking for subsequent region, start at idx_in + 1
        else:
            # if last region
            if idx_in == len(DICT_POCKET_KLIFS_REGIONS) - 1:
                # if KD end is None, return len(self.uniprotSeq) - 1
                if self.idx_kd[1] is None:
                    return len(self.uniprotSeq) - 1
                # if KD end is provided, return KD end
                else:
                    return self.idx_kd[1]
            idx_temp = self.list_substring_idxs[idx_in + 1]
            if idx_temp is not None and len(idx_temp) == 1:
                idx_out = idx_temp[0]
            else:
                idx_out = self.find_start_or_end_idx_recursively(
                    idx_in + 1, bool_start=False
                )

        return idx_out

    def return_partial_alignments(
        self,
        idx: int,
        align_fn: Align.PairwiseAligner | None = None,
    ) -> tuple[int, int, Align.PairwiseAlignments | list[int | None] | None]:
        """Return partial alignments for b.l region.

        Parameters
        ----------
        idx : int
            Index of region (e.g., I is 0, g.l is 1, etc.)
        align_fn : Align.PairwiseAligner | None
            Alignment function; if none provided will use exact match

        Returns
        -------
        tuple[int, int, Align.PairwiseAlignments | list[int | None] | None]
            Start, end, and alignments (either indices or alignments or None) for region

        """
        start_idx = self.find_start_or_end_idx_recursively(idx, bool_start=True)
        end_idx = self.find_start_or_end_idx_recursively(idx, bool_start=False)

        str_klifs = self.remove_gaps_from_klifs(self.list_klifs_substr_actual[idx])
        str_uniprot = self.uniprotSeq[start_idx:end_idx]

        if len(str_klifs) == 0:
            return start_idx, end_idx, None
        else:
            if align_fn is not None:
                aligned = align_fn.align(str_klifs, str_uniprot)
            else:
                aligned = self.return_idx_of_substring_in_superstring(
                    str_uniprot, str_klifs
                )

            return start_idx, end_idx, aligned

    def iterate_klifs_alignment(
        self,
    ) -> None:
        """Align KLIFS region to UniProt canonical Uniprot sequence."""
        dict_klifs = DICT_POCKET_KLIFS_REGIONS
        list_klifs = list(dict_klifs.keys())

        for klifs_index, klifs_region in enumerate(list_klifs):
            klifs_region_start, klifs_region_end = klifs_region, klifs_region
            klifs_idx_start, klifs_idx_end = (
                dict_klifs[klifs_region_start]["start"] - 1,
                dict_klifs[klifs_region_end]["end"],
            )

            str_klifs, list_substring_idx = self.align_klifs_pocket_to_uniprot_seq(
                idx_start=klifs_idx_start,
                idx_end=klifs_idx_end,
            )
            self.list_klifs_substr_actual.append(str_klifs)

            # if None KLIFS all "-" so disregard; if multiple idxs returned,
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
                    len_klifs = len(self.remove_gaps_from_klifs(str_klifs))
                klifs_idx_start, klifs_idx_end = (
                    dict_klifs[klifs_region_start]["start"] - 1,
                    dict_klifs[klifs_region_end]["end"],
                )
                str_klifs, list_substring_idx = self.align_klifs_pocket_to_uniprot_seq(
                    idx_start=klifs_idx_start,
                    idx_end=klifs_idx_end,
                )
                # if concat with previous, offset by length of preceding KLIFS region with gaps removed
                if (
                    not bool_cont
                    and list_substring_idx is not None
                    and len(list_substring_idx) != 0
                ):
                    len_offset = len(self.remove_gaps_from_klifs(str_klifs)) - len_klifs
                    list_substring_idx = [i + len_offset for i in list_substring_idx]

            self.list_klifs_region.append(klifs_region_start + ":" + klifs_region_end)
            self.list_klifs_substr_match.append(str_klifs)
            self.list_substring_idxs.append(list_substring_idx)

        # post-hoc adjustments

        # b.l region non-contiguous alignment
        idx_bl = [i for i, x in enumerate(list_klifs) if x == "b.l"][0]
        # STK40 has no b.l region, so skip entirely
        if self.list_substring_idxs[idx_bl] is None:
            pass
        else:
            start, _, bl_alignments = self.return_partial_alignments(
                idx=idx_bl,
                align_fn=BL2UniProtAligner(),
            )
            list_bl = self.select_correct_alignment(bl_alignments)
            self.list_substring_idxs[idx_bl] = [i + start for i in list_bl]

        # interpolate multi-matching using previous and subsequent regions
        for idx, substr_idx in enumerate(self.list_substring_idxs):
            if idx != idx_bl and substr_idx is not None and len(substr_idx) > 1:
                start = self.find_start_or_end_idx_recursively(idx, bool_start=True)
                end = self.find_start_or_end_idx_recursively(idx, bool_start=False)
                self.list_substring_idxs[idx] = [
                    i for i in substr_idx if i >= start and i <= end
                ]

        # final partial alignment algorithm
        for idx, substr_idx in enumerate(self.list_substring_idxs):
            if substr_idx == []:
                # check exact match
                start_exact, _, align_exact = self.return_partial_alignments(idx=idx)
                if align_exact != [] and len(align_exact) == 1:
                    self.list_substring_idxs[idx] = [
                        i + start_exact for i in align_exact
                    ]
                # if no exact match, try local alignment
                else:
                    start_local, _, align_local = self.return_partial_alignments(
                        idx=idx,
                        align_fn=Kincore2UniProtAligner(),
                    )
                    if len(align_local) == 1 and align_local[
                        0
                    ].target == self.remove_gaps_from_klifs(align_local[0][0, :]):
                        list_local = self.return_idx_of_alignment_match(align_local[0])
                        self.list_substring_idxs[idx] = [
                            i + start_local for i in list_local
                        ]
                    # if no exact match, try global alignment
                    else:
                        start_global, _, align_global = self.return_partial_alignments(
                            idx=idx,
                            align_fn=BL2UniProtAligner(),
                        )
                        # all that remains is linker region where some gaps (1 + 3) occur
                        list_global = self.select_correct_alignment(
                            align_global, bool_bl=False
                        )
                        self.list_substring_idxs[idx] = [
                            i + start_global for i in list_global
                        ]

    def generate_alignment_list(self, bool_offset: bool) -> list[str | None]:
        """Generate and validate alignment list.

        Parameters
        ----------
        bool_offset : bool
            If True, offset indices by 1 to match 1-based indexing

        Returns
        -------
        list[str | None]
            List of alignments
        """
        list_align = []
        for list_idx, list_seq in zip(
            self.list_substring_idxs, self.list_klifs_substr_actual
        ):
            if list_idx is not None and len(list_idx) == 1:
                if bool_offset:
                    temp_idx = list_idx[0] + 1
                else:
                    temp_idx = list_idx[0]
                for aa in list_seq:
                    if aa == "-":
                        list_align.append(None)
                    else:
                        list_align.append(temp_idx)
                        temp_idx += 1
            elif list_idx is None:
                list_align.extend([None] * len(list_seq))
            else:
                idx = 0
                for aa in list_seq:
                    if aa == "-":
                        list_align.append(None)
                    else:
                        if bool_offset:
                            list_align.append(list_idx[idx] + 1)
                        else:
                            list_align.append(list_idx[idx])
                        idx += 1

        for idx, i in enumerate(list_align):
            if i is not None:
                if bool_offset:
                    i = i - 1
                if self.uniprotSeq[i] != self.klifsSeq[idx]:
                    logging.error(
                        f"Alignment mismatch at {idx} position in KLIFS pocket and {i} of UniProt\n"
                        f"UniProt: {self.uniprotSeq[i]}\n"
                        f"KLIFS: {self.klifsSeq[idx]}\n"
                    )
                    return None
            else:
                if self.klifsSeq[idx] != "-":
                    logging.error(
                        f"Alignment mismatch at {idx + 1} position in KLIFS pocket and {i} of UniProt\n"
                        f"UniProt: -\n"
                        f"KLIFS: {self.klifsSeq[idx]}\n"
                    )
                    return None

        self.list_align = list_align

    def get_inter_region(self):
        """Get inter-region sequences."""

        list_region = list(DICT_POCKET_KLIFS_REGIONS.keys())
        dict_start_end = {
            list_region[i - 1]: list_region[i] for i in range(1, len(list_region) - 1)
        }
        dict_cols = {
            key: list(i for i in LIST_KLIFS_REGION if i.split(":")[0] == key)
            for key in list_region
        }

        list_inter = []
        for key1, val1 in dict_start_end.items():
            keys_start, keys_end = dict_cols[key1], dict_cols[val1]

            start = [
                val for key, val in self.KLIFS2UniProtIdx.items() if key in keys_start
            ]
            if all(v is None for v in start):
                max_start = None
            else:
                max_start = np.nanmax(np.array(start, dtype=float)) + 1

            end = [val for key, val in self.KLIFS2UniProtIdx.items() if key in keys_end]
            if all(v is None for v in end):
                min_end = None
            else:
                min_end = np.nanmin(np.array(end, dtype=float))

            list_inter.append((max_start, min_end))

        dict_inter = dict(
            zip([f"{key}:{val}" for key, val in dict_start_end.items()], list_inter)
        )

        dict_fasta = {i: {} for i in LIST_INTER_REGIONS}
        for region in LIST_INTER_REGIONS:
            start, end = dict_inter[region][0], dict_inter[region][1]
            if start is not None and end is not None:
                if end - start == 0:
                    dict_fasta[region] = None
                else:
                    dict_fasta[region] = self.uniprotSeq[int(start) - 1 : int(end) - 1]
            else:
                dict_fasta[region] = None

        return dict_fasta

    def recursive_idx_search(
        self,
        idx: int,
        in_dict: dict[str, int],
        decreasing: bool,
    ):
        """Recursively search for index in dictionary.

        Parameters
        ----------
        idx : int
            Index to start search
        in_dict : dict[str, int]
            Dictionary to search
        decreasing : bool
            If True, search in decreasing order; if False, search in increasing order

        Returns
        -------
        idx : int
            Index in dictionary

        """
        if idx == 0:
            return "NONE"
        list_keys = list(in_dict.keys())
        if in_dict[list_keys[idx]] is None:
            if decreasing:
                idx = self.recursive_idx_search(idx - 1, in_dict, True)
            else:
                idx = self.recursive_idx_search(idx + 1, in_dict, False)
        return idx

    def find_intra_gaps(
        self,
        dict_in: dict[str, int],
        bool_bl: bool = True,
    ) -> tuple[int, str] | None:
        """Find intra-pocket gaps in KLIFS pocket region.

        Parameters
        ----------
        dict_in : dict[str, int]
            Dictionary of KLIFS regions and their corresponding indices
        bool_bl : bool
            If True, find intra-region gaps for b.l region; if False, find intra-region gaps for linker region

        Returns
        -------
        tuple[str, str] | None
            Tuple of intra-region gaps

        """
        if bool_bl:
            region, idx_in, idx_out = "b.l", 1, 2
            region, idx_in, idx_out = "b.l", 1, 2
        else:
            region, idx_in, idx_out = "linker", 0, 1

        list_keys = list(dict_in.keys())
        list_idx = [idx for idx, i in enumerate(dict_in.keys()) if region in i]

        # TODO: ATR and CAMKK1 have inter hinge:linker region
        start = list_idx[idx_in]
        end = list_idx[idx_out]

        if dict_in[list_keys[start]] is None:
            start = self.recursive_idx_search(start - 1, dict_in, True)
        if dict_in[list_keys[end]] is None:
            end = self.recursive_idx_search(end + 1, dict_in, False)

        # STK40 has no b.l region or preceding
        if start == "NONE":
            return None

        return (dict_in[list_keys[start]], dict_in[list_keys[end]])

    def return_intra_gap_substr(self, bl_bool) -> str | None:
        """Return intra-region gap substring.

        Parameters
        ----------
        bl_bool : bool
            If True, find intra-region gaps for b.l region; if False, find intra-region gaps for linker region

        Returns
        -------
        str | None
            Intra-region gap substring

        """
        tuple_idx = self.find_intra_gaps(self.KLIFS2UniProtIdx, bl_bool)
        if tuple_idx is None:
            return None
        else:
            start, end = tuple_idx[0], tuple_idx[1]
            if end - start == 1:
                return None
            else:
                return self.uniprotSeq[start : end - 1]

    def get_intra_region(self):
        """Get intra-region sequences."""
        list_seq = []
        for region in LIST_INTRA_REGIONS:
            if region.split("_")[0] == "b.l":
                list_seq.append(self.return_intra_gap_substr(True))
            else:
                list_seq.append(self.return_intra_gap_substr(False))
        return dict(zip(LIST_INTRA_REGIONS, list_seq))

    def generate_alignment_list_including_gaps(self):
        """Return fully aligned KLIFS pocket."""
        list_region = list(DICT_POCKET_KLIFS_REGIONS.keys())

        # inter region
        dict_inter = self.get_inter_region()

        list_inter_regions = list(dict_inter.keys())
        list_idx_inter = list(
            chain(
                *[
                    list(
                        idx for idx, j in enumerate(list_region) if j == i.split(":")[0]
                    )
                    for i in list_inter_regions
                ]
            )
        )

        list_region_combo = list(list_region)
        i = 0
        for idx, val in zip(list_idx_inter, list_inter_regions):
            list_region_combo.insert(idx + i + 1, val)
            i += 1

        # intra region
        dict_intra = self.get_intra_region()

        idx = list_region_combo.index("b.l")
        list_region_combo[idx : idx + 1] = "b.l_1", "b.l_intra", "b.l_2"

        idx = list_region_combo.index("linker")
        list_region_combo[idx : idx + 1] = "linker_1", "linker_intra", "linker_2"

        dict_full_klifs_region = {region: None for region in list_region_combo}

        dict_actual = dict(zip(list_region, self.list_klifs_substr_actual))
        # for region in list_region_combo:KL
        for region, seq in dict_actual.items():
            if region == "b.l":
                dict_full_klifs_region["b.l_1"] = seq[0:2]
                dict_full_klifs_region["b.l_2"] = seq[2:]
                pass
            elif region == "linker":
                dict_full_klifs_region["linker_1"] = seq[0:1]
                dict_full_klifs_region["linker_2"] = seq[1:]
            else:
                dict_full_klifs_region[region] = seq

        for region, seq in dict_inter.items():
            dict_full_klifs_region[region] = seq

        for region, seq in dict_intra.items():
            dict_full_klifs_region[region] = seq

        self.KLIFS2UniProtSeq = dict_full_klifs_region


# # NOT IN USE - USE TO GENERATE ABOVE

# list_multi = [list(val.KLIFSPocket.list_klifs_region[idx] for idx, entry \
#                    in enumerate(val.KLIFSPocket.list_substring_idxs) if entry is not None and len(entry)>1)\
#               for key, val in dict_klifs.items()]
# set(chain(*list_multi)) # {'b.l:b.l', 'linker:linker'}

# df_klifs_idx = pd.DataFrame([list(j for i, j in val.KLIFS2UniProt.items()) for key, val in dict_klifs.items()],
#                             columns=klifs.LIST_KLIFS_REGION, index=dict_klifs.keys())

# # dict_temp = klifs.DICT_POCKET_KLIFS_REGIONS
# list_region = list(klifs.DICT_POCKET_KLIFS_REGIONS.keys())

# dict_start_end = {list_region[i-1]:list_region[i] for i in range(1, len(list_region)-1)}
# dict_cols = {key: list(i for i in df_klifs_idx.columns.tolist() \
#                        if i.split(":")[0] == key) for key in list_region}

# list_inter = []
# for key, val in dict_start_end.items():

#     list_temp = []
#     for idx, row in df_klifs_idx.iterrows():

#         cols_start, cols_end = dict_cols[key], dict_cols[val]

#         start = row.loc[cols_start].values
#         if np.all(np.isnan(start)):
#             max_start = None
#         else:
#             max_start = np.nanmax(start) + 1

#         end = row.loc[cols_end].values
#         if np.all(np.isnan(end)):
#             min_end = None
#         else:
#             min_end = np.nanmin(end)

#         list_temp.append((max_start, min_end))

#     list_inter.append(list_temp)

# df_inter = pd.DataFrame(list_inter,
#                         index=[f"{key}:{val}" for key, val in dict_start_end.items()],
#                         columns=df_klifs_idx.index).T
# df_length = df_inter.map(lambda x: try_except_substraction(x[0], x[1]))

# # df_one = df_length.loc[:, df_length.apply(lambda x: any(x == 1))]
# # df_multi = df_length.loc[:, df_length.apply(lambda x: any(x > 1))]
# df_multi = df_length.loc[:, df_length.apply(lambda x: any(x > 0))]
# # αC:b.l - {'BUB1B': 'E'} - need to skip this, seems to be in b.l gap region

# df_bl = pd.DataFrame([list(j for i, j in val.KLIFS2UniProt.items() if i in list_idx_dict[0]) \
#                       for key, val in dict_klifs.items()],
#                      columns=list_idx_dict[0], index=dict_klifs.keys())
# df_bl[df_bl.isnull().any(axis=1)]

# df_linker = pd.DataFrame([list(j for i, j in val.KLIFS2UniProt.items() if i in list_idx_dict[1]) \
#                           for key, val in dict_klifs.items()],
#                          columns=list_idx_dict[1], index=dict_klifs.keys())
# df_linker[df_linker.isnull().any(axis=1)]
