import ast
import logging
from dataclasses import dataclass

import pandas as pd
from mkt.databases import requests_wrapper, utils_requests
from mkt.databases.api_schema import RESTAPIClient
from mkt.schema.kinase_schema import SwissProtPattern, TrEMBLPattern

logger = logging.getLogger(__name__)


@dataclass
class UniProt:
    """Class to interact with the UniProt API."""

    uniprot_id: str
    """UniProt ID."""
    url: str = "https://rest.uniprot.org/uniprotkb"
    """URL for the UniProt API."""


@dataclass
class UniProtFASTA(UniProt, RESTAPIClient):
    """Class to interact UniProt API for FASTA download."""

    url_fasta: str | None = None
    """URL for the UniProt FASTA API including UniProt ID."""
    _header: str | None = None
    """FASTA header."""
    _sequence: str | None = None
    """FASTA sequence."""

    def __post_init__(self):
        self.url = self.set_url()
        self.url_fasta = self.url.replace("<ID>", self.uniprot_id)
        self._header, self._sequence = self.query_api()

    def set_url(self):
        """Set URL for UniProt API; UniProtKB if SwissProt or Unisave if TrEMBL."""
        import re

        if re.match(SwissProtPattern, self.uniprot_id):
            return "https://rest.uniprot.org/uniprotkb/<ID>.fasta"
        elif re.match(TrEMBLPattern, self.uniprot_id):
            return "https://rest.uniprot.org/unisave/<ID>?format=fasta&versions=67"
        else:
            logging.error(f"Invalid UniProt ID: {self.uniprot_id}")
            raise ValueError(f"Invalid UniProt ID: {self.uniprot_id}")

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
        res = requests_wrapper.get_cached_session().get(self.url_fasta)
        if res.ok:
            str_fasta = res.text
            if bool_seq:
                header, seq = self._convert_fasta2seq(str_fasta)
        else:
            utils_requests.print_status_code_if_res_not_ok(res)
            print(f"UniProt ID: {self.uniprot_id}\n")
            header, seq = None, None
        return header, seq

    @staticmethod
    def _convert_fasta2seq(str_fasta) -> tuple[str, str]:
        """Convert FASTA sequence to string sequence (i.e., remove header line breaks).

        Parameters
        ----------
        str_fasta : str
            FASTA string (including header and line breaks)

        Returns
        -------
        header, seq : tuple[str, str]
            FASTA header and sequence strings (excluding line breaks)

        """
        header = str_fasta.split("\n")[0]
        seq = [i.split("\n", 1)[1].replace("\n", "") for i in [str_fasta]][0]
        return header, seq


DICT_UNIPROT_EVIDENCE = {
    "ECO:0000250": "Computational, ISS",
    "ECO:0000255": "Computational, ISM",
    "ECO:0000269": "Manual, experimental",
    "ECO:0000305": "Inferred by curator",
    "ECO:0007744": "Manual, combo",
}
"""dict[str, str]: UniProt evidence codes mapped to descriptions."""


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

    def extract_modified_residues(self):
        """Extract modified residues from JSON.

        Returns
        -------
        list[str]
            List of modified residues

        """
        if self._json is not None:
            try:
                df_mod_res = pd.DataFrame(
                    [
                        entry
                        for entry in self.json["features"]
                        if entry["type"] == "Modified residue"
                    ]
                )
                df_mod_res["location"] = pd.json_normalize(df_mod_res["location"])
                df_mod_res["evidenceCode"] = df_mod_res["evidences"].apply(
                    lambda x: {entry["evidenceCode"] for entry in x}
                )
                return df_mod_res
            except KeyError as e:
                logger.error(f"Key error: {e}")
                return None
        else:
            None
