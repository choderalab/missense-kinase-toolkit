import ast
from dataclasses import dataclass

from mkt.databases import requests_wrapper, utils_requests
from mkt.databases.api_schema import RESTAPIClient


@dataclass
class UniProt:
    """Class to interact with the UniProt API."""

    uniprot_id: str
    """UniProt ID."""
    url: str = "https://rest.uniprot.org/uniprotkb"
    """UniProt API URL."""


@dataclass
class UniProtFASTA(UniProt, RESTAPIClient):
    """Class to interact UniProt API for FASTA download."""

    _sequence: str | None = None

    def __post_init__(self):
        self._sequence = self.query_api()

    def query_api(
        self,
        bool_seq: bool = True,
    ) -> str | None:
        """Get FASTA sequence for UniProt ID.

        Parameters
        ----------
        bool_seq : bool
            If True, return sequence string only (i.e., no header or line breaks); otherwise return full FASTA string

        Returns
        -------
        str | None
            FASTA sequences for UniProt ID; None if request fails

        """
        self.url_fasta = f"{self.url}/{self.uniprot_id}.fasta"

        res = requests_wrapper.get_cached_session().get(self.url_fasta)
        if res.ok:
            str_fasta = res.text
            if bool_seq:
                str_fasta = self._convert_fasta2seq(str_fasta)
        else:
            str_fasta = None
            utils_requests.print_status_code_if_res_not_ok(res)
        return str_fasta

    @staticmethod
    def _convert_fasta2seq(str_fasta):
        """Convert FASTA sequence to string sequence (i.e., remove header line breaks).

        Parameters
        ----------
        str_fasta : str
            FASTA string (including header and line breaks)

        Returns
        -------
        str_seq : str
            Sequence string (excluding header and line breaks)

        """
        str_seq = [i.split("\n", 1)[1].replace("\n", "") for i in [str_fasta]][0]
        return str_seq


@dataclass
class UniProtJSON(UniProt, RESTAPIClient):
    """Class to interact UniProt API for JSON download."""

    headers: str = "{'Accept': 'application/json'}"
    """Header for the API request."""
    _json: dict | None = None

    def __post_init__(self):
        self._json = self.query_api()

    def query_api(self) -> dict | None:
        """Get JSON for UniProt ID.

        Returns
        -------
        dict | None
            JSON for UniProt ID; None if request fails

        """
        self.url_json = f"{self.url}/{self.uniprot_id}"

        res = requests_wrapper.get_cached_session().get(
            self.url_json,
            headers=ast.literal_eval(self.headers),
        )
        if res.ok:
            json = res.json()
        else:
            json = None
            utils_requests.print_status_code_if_res_not_ok(res)
        return json
