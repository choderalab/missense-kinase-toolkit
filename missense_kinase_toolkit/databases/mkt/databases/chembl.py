import logging
from dataclasses import dataclass

from mkt.databases.api_schema import RESTAPIClient
from mkt.databases.requests_wrapper import get_cached_session

logger = logging.getLogger(__name__)


@dataclass
class ChEMBL(RESTAPIClient):
    """ChEMBL API client."""

    id: str
    """ID for querying specific entities."""
    url_base: str = "https://www.ebi.ac.uk/chembl/api/data"
    """Base URL for the ChEMBL API."""
    url_suffix: str | None = None
    """URL suffix to update for specific queries."""
    url_query: str | None = None
    """URL query for specific queries."""
    params: dict | None = None
    """Parameters for the API query."""
    _json: dict | None = None

    def __post_init__(self):
        """Initialize the ChEMBL API client."""
        if self.url_suffix is not None:
            self.url_query = f"{self.url_base}{self.url_suffix}"
        else:
            self.url_query = f"{self.url_base}/molecule/search"
        if self.params is None:
            self.params = {"q": self.id, "format": "json"}
        self.query_api()

    def query_api(self):
        """Query the ChEMBL API for a given URL."""
        if self.url_query is None:
            logger.error("URL query is not set. Please update the URL before querying.")
            return
        if self.params is not None:
            res = get_cached_session().get(self.url_query, params=self.params)
        else:
            res = get_cached_session().get(self.url_query)
        if res.ok:
            self._json = res.json()

    def update_url(self, url: str) -> None:
        """Update the URL for querying the ChEMBL API."""
        self.url_query = f"{self.url_base}{self.url.suffix}"

    def update_params(self, **kwargs) -> None:
        """Update the parameters for the API query."""
        self.params.update(kwargs)


@dataclass
class ChEMBLMolecule(ChEMBL):
    """ChEMBL Molecule API client."""

    id: str
    """Molecule ID for querying specific molecules."""
    url_suffix: str = "/molecule/search"
    """URL suffix for querying molecules in ChEMBL."""

    def get_chembl_id(self) -> str | None:
        """Get the ChEMBL ID for the queried molecule."""
        if self._json is not None and "molecules" in self._json:
            if len(self._json["molecules"]) == 0:
                logger.error("No molecules found in the response.")
                return None
            else:
                return [i["molecule_chembl_id"] for i in self._json["molecules"]]
        else:
            logger.error("No ChEMBL ID found in the response.")
            return None
