import ast
import logging

from mkt.databases import requests_wrapper
from mkt.databases.api_schema import RESTAPIClient
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AlphaFoldPrediction(RESTAPIClient):
    """Class to query the AlphaFold EBI prediction API for a given UniProt accession.

    Parameters:
    -----------
    uniprot_id : str
        UniProt accession (e.g. "P00533").
    """

    uniprot_id: str
    """UniProt accession to query."""
    url: str = "https://alphafold.ebi.ac.uk/api"
    """Base URL for the AlphaFold EBI API."""
    headers: str = "{'Accept': 'application/json'}"
    """Header for the API request."""
    _json: dict | None = None
    """JSON response from the AlphaFold API."""

    def __post_init__(self):
        self.query_api()

    def query_api(self) -> None:
        """Query the AlphaFold prediction endpoint and populate _json.

        Returns:
        --------
        None
        """
        url = f"{self.url}/prediction/{self.uniprot_id}"
        res = requests_wrapper.get_cached_session().get(
            url,
            headers=ast.literal_eval(self.headers),
        )

        if res.ok:
            results = res.json()
            # filter to canonical accession (exact match, excluding isoforms)
            canonical = [
                r for r in results if r.get("uniprotAccession") == self.uniprot_id
            ]
            if len(canonical) == 1:
                self._json = canonical[0]
            else:
                logger.warning(
                    "Expected 1 canonical result for %s, got %d",
                    self.uniprot_id,
                    len(canonical),
                )
                self._json = None
        else:
            logger.error(
                "AlphaFold API error for %s: %s", self.uniprot_id, res.status_code
            )
            self._json = None


@dataclass
class AlphaFoldStructure(AlphaFoldPrediction):
    """Class to download the mmCIF file from the AlphaFold EBI server.

    Inherits from AlphaFoldPrediction and uses the cifUrl from the prediction
    JSON to download and store the CIF content.

    Parameters:
    -----------
    uniprot_id : str
        UniProt accession (e.g. "P00533").
    """

    _cif: str | None = None
    """mmCIF file content downloaded from AlphaFold."""

    def __post_init__(self):
        super().__post_init__()
        self._download_cif()

    def _download_cif(self) -> None:
        """Download the mmCIF file from the cifUrl in the prediction JSON.

        Returns:
        --------
        None
        """
        if self._json is None:
            logger.error("No prediction JSON available for %s", self.uniprot_id)
            return

        cif_url = self._json.get("cifUrl")
        if cif_url is None:
            logger.error("No cifUrl in prediction JSON for %s", self.uniprot_id)
            return

        res = requests_wrapper.get_cached_session().get(cif_url)
        if res.ok:
            self._cif = res.text
        else:
            logger.error(
                "Failed to download CIF for %s: %s", self.uniprot_id, res.status_code
            )
            self._cif = None
