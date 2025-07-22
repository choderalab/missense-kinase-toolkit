import logging
from dataclasses import dataclass, field

from mkt.databases.api_schema import RESTAPIClient
from mkt.databases.requests_wrapper import get_cached_session

logger = logging.getLogger(__name__)


@dataclass
class ChEMBL(RESTAPIClient):
    """ChEMBL API client."""

    id: str
    """ID for querying specific entities."""
    url_suffix: str
    """URL suffix to update for specific queries."""
    url_base: str = "https://www.ebi.ac.uk/chembl/api/data"
    """Base URL for the ChEMBL API."""
    url_query: str | None = None
    """URL query for specific queries."""
    params: dict = field(default_factory=lambda: {"q": "<ID>", "format": "json"})
    """Parameters for the API query."""
    _json: dict | None = None

    def __post_init__(self):
        """Initialize the ChEMBL API client."""
        self.url_query = f"{self.url_base}{self.url_suffix}"
        self.params = {k: v.replace("<ID>", self.id) for k, v in self.params.items()}
        self.query_api()

    def query_api(self):
        """Query the ChEMBL API for a given URL."""
        if self.url_query is None:
            logger.error("URL query is not set. Please update the URL before querying.")
            return
        if self.params:
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


@dataclass
class ChEMBLMoleculeSearch(ChEMBL):
    """ChEMBL molecule search API client."""

    url_suffix: str = "/molecule/search"
    """URL suffix for querying molecule search in ChEMBL."""


@dataclass
class ChEMBLMoleculeExact(ChEMBL):
    """ChEMBL molecule exact match API client."""

    url_suffix: str = "/molecule"
    """URL suffix for querying exact molecule match in ChEMBL."""
    params: dict = field(
        default_factory=lambda: {
            "molecule_synonyms__molecule_synonym__iexact": "<ID>",
            "format": "json",
        }
    )
    """Parameters for the molecule exact match API query."""


@dataclass
class ChEMBLMoleculePreferred(ChEMBL):
    """ChEMBL molecule preferred match API client."""

    url_suffix: str = "/molecule"
    """URL suffix for querying exact molecule match in ChEMBL."""
    params: dict = field(
        default_factory=lambda: {"pref_name__iexact": "<ID>", "format": "json"}
    )
    """Parameters for the molecule preferred match API query."""
