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
    verbose: bool = False
    """Flag to enable verbose logging."""
    _json: dict | None = None
    """JSON response from the API query."""

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
                if self.verbose:
                    logger.error(f"No molecules found in the response for {self.id}.")
                return None
            else:
                return [i["molecule_chembl_id"] for i in self._json["molecules"]]
        else:
            if self.verbose:
                logger.error(f"No ChEMBL ID found in the response for {self.id}.")
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


def return_chembl_id(drug: str):
    """Return the ChEMBL ID for a given drug name.

    Parameters
    ----------
    drug : str
        The name of the drug to search for in ChEMBL.

    Returns
    -------
    tuple
        A tuple containing the ChEMBL ID and the source of the ID (exact, preferred, or search);
            if no ID is found, returns (None, None).
    """
    chembl_id = ChEMBLMoleculeExact(id=drug).get_chembl_id()
    source = "exact"

    # if None try preferred match
    if chembl_id == [] or chembl_id is None:
        chembl_id = ChEMBLMoleculePreferred(id=drug).get_chembl_id()
        source = "preferred"

    # if still None, do search
    if chembl_id == [] or chembl_id is None:
        chembl_id = ChEMBLMoleculeSearch(id=drug, verbose=True).get_chembl_id()
        source = "search"

    # if still None, return None
    if chembl_id == [] or chembl_id is None:
        logger.error(f"No ChEMBL ID found for {drug}.")
        chembl_id, source = None, None

    return chembl_id, source


@dataclass
class ChEMBLMolecule(ChEMBL):
    """ChEMBL molecule API client."""

    url_suffix: str = "/molecule"
    """URL suffix for querying exact molecule match in ChEMBL."""
    params: dict = field(
        default_factory=lambda: {
            "molecule_synonyms__molecule_synonym__iexact": "<ID>",
            "format": "json",
        }
    )
    """Parameters for the molecule API query."""

    def check_molecules(self) -> str | None:
        """Return the SMILES string for the queried molecule."""
        if self._json is not None and "molecules" in self._json:
            if len(self._json["molecules"]) == 0:
                if self.verbose:
                    logger.error(f"No molecules found in the response for {self.id}.")
                return None
            if len(self._json["molecules"]) > 1:
                if self.verbose:
                    logger.warning(
                        f"Multiple molecules found for {self.id}. Returning the first one."
                    )
        else:
            if self.verbose:
                logger.error(f"No molecules found in the response for {self.id}.")
            return None

    def return_smiles(self) -> str | None:
        """Return the SMILES string for the queried molecule."""
        self.check_molecules()
        return (
            self._json["molecules"][0]
            .get("molecule_structures", {})
            .get("canonical_smiles", None)
        )

    def return_preferred_name(self) -> str | None:
        """Return the preferred name for the queried molecule."""
        self.check_molecules()
        return self._json["molecules"][0].get("pref_name", None)

    def adjudicate_preferred_name(self, str_in: str | None = None) -> str:
        """Return the adjudicated preferred name for the queried molecule.

        Parameters
        ----------
        str_in : str | None
            The input string to adjudicate the preferred name. If None, will not compare to input string.

        Returns
        -------
        str
            The adjudicated preferred name or the original ID if no preferred name is found or matches the input string.
        """
        preferred_name = self.return_preferred_name()
        if preferred_name is not None:
            # prefer INN to internal identifier
            if str_in is not None:
                if "-" in str_in and "-" not in preferred_name:
                    return preferred_name.title()
                elif "-" in preferred_name and "-" not in str_in:
                    return str_in
                else:
                    return preferred_name.title()
            return preferred_name.title()
        else:
            if str_in is not None:
                return str_in.title()
            else:
                logger.error(
                    f"No preferred name found for {self.id}. Returning original ID."
                )
                return self.id.upper()
