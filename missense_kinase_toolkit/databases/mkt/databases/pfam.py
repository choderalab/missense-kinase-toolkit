import json
import logging

import pandas as pd
from mkt.databases import requests_wrapper, utils_requests

logger = logging.getLogger(__name__)


class Pfam:
    """Class to interact with the Pfam API.

    Parameters
    ----------
    uniprot_id : str
        UniProt ID to query Pfam API for.

    Attributes
    ----------
    url : str
        Pfam API URL.
    uniprot_id : str
        UniProt ID to query Pfam API for.
    _pfam : pd.DataFrame | None
        DataFrame with Pfam domain information if request is successful, None if response is empty or
    """

    def __init__(
        self,
        uniprot_id: str,
    ) -> None:
        """Initialize Pfam Class object.

        Attributes
        ----------
        url : str
            Pfam API URL

        """
        self.url = "https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/UniProt/"
        self.uniprot_id = uniprot_id
        self._pfam = self.query_pfam_api()

    def query_pfam_api(self):
        """Queries Pfam API for UniProt ID as DataFrame object.

        Returns
        -------
        pd.DataFrame | str | None
            DataFrame with Pfam domain information if request is successful, None if response is empty or request fails

        """
        url = f"{self.url}{self.uniprot_id}"

        header = {"Accept": "application/json"}
        res = requests_wrapper.get_cached_session().get(url, headers=header)

        if res.ok:
            if len(res.text) == 0:
                logger.warning(f"No PFAM domains found: {self.uniprot_id}...")
                return None
            else:
                list_json = json.loads(res.text)["results"]

                # metadata for UniProt ID
                list_metadata = [entry["metadata"] for entry in list_json]
                list_metadata = [
                    {
                        "pfam_accession" if k == "accession" else k: v
                        for k, v in entry.items()
                    }
                    for entry in list_metadata
                ]

                # Pfam domains locations
                list_locations = [
                    entry["proteins"][0]["entry_protein_locations"][0]["fragments"][0]
                    for entry in list_json
                ]

                # model information
                list_model = [
                    entry["proteins"][0]["entry_protein_locations"][0]
                    for entry in list_json
                ]
                [entry.pop("fragments", None) for entry in list_model]

                # protein information
                # do last because pop is an in-place operation
                list_protein = [entry["proteins"][0] for entry in list_json]
                [entry.pop("entry_protein_locations", None) for entry in list_protein]
                list_protein = [
                    {"uniprot" if k == "accession" else k: v for k, v in entry.items()}
                    for entry in list_protein
                ]

                df_concat = pd.concat(
                    [
                        pd.DataFrame(list_protein),
                        pd.DataFrame(list_metadata),
                        pd.DataFrame(list_locations),
                        pd.DataFrame(list_model),
                    ],
                    axis=1,
                )

                return df_concat
        else:
            utils_requests.print_status_code_if_res_not_ok(res)
            return None


def find_pfam_domain(
    input_id: str,
    input_position: int,
    df_ref: pd.DataFrame,
    col_ref_id: str,
    col_ref_start: None | str = None,
    col_ref_end: None | str = None,
    col_ref_domain: None | str = None,
) -> str | None:
    """Find Pfam domain for a given HGNC symbol and position

    Parameters
    ----------
    input_id : str
        Input ID that matches
    input_position : int
        Codon position in UniProt canonical sequence
    df_ref : pd.DataFrame
        DataFrame with Pfam domain information
    col_ref_id : str
        Column that contains the IDs to match to in the df_ref dataframe
    col_ref_start : None | str
        Column containing the domain start position; if None defaults to "start" (Pfam API default)
    col_ref_end : None | str
        Column containing the domain end position; if None defaults to "end" (Pfam API default)
    col_ref_domain : None | str
        Column containing the domain name; if None defaults to "name" (Pfam API default)

    Returns
    -------
    str | None
        Pfam domain if found, None if not found
    """

    if col_ref_start is None:
        col_ref_start = "start"
    if col_ref_end is None:
        col_ref_end = "end"
    if col_ref_domain is None:
        col_ref_domain = "name"

    df_temp = df_ref.loc[df_ref[col_ref_id] == input_id].reset_index()
    try:
        domain = df_temp.loc[
            (
                (input_position >= df_temp[col_ref_start])
                & (input_position <= df_temp[col_ref_end])
            ),
            col_ref_domain,
        ].values[0]
        return domain
    except IndexError:
        return None
