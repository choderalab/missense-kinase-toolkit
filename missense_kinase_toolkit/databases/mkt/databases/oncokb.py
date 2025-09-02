import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from mkt.databases import requests_wrapper
from mkt.databases.api_schema import APIKeyRESTAPIClient
from mkt.databases.config import maybe_get_oncokb_token

logger = logging.getLogger(__name__)


@dataclass
class OncoKB(APIKeyRESTAPIClient, ABC):
    """OncoKB API client."""

    url: str = "https://www.oncokb.org/api/v1"
    """Base URL for the OncoKB API."""
    header: dict = field(default_factory=dict)
    """OncoKB API token, if available."""
    url_query: str | None = field(init=False, default=None)
    """URL to update for specific queries."""
    _json: dict | None = field(init=False, default=None)

    def __post_init__(self):
        """Initialize the OncoKB API client."""
        self.header = self.set_api_key()
        try:
            "Authorization" in self.header
            self.header.update({"Accept": "application/json"})
        except KeyError:
            logger.error(
                "No OncoKB API token provided. Please set it in the environment."
            )
        self.update_url()
        self.query_api()

    def maybe_get_token(self) -> str | None:
        return maybe_get_oncokb_token()

    @abstractmethod
    def update_url(self): ...

    def query_api(self):
        """Query the OncoKB API for a given URL."""
        res = requests_wrapper.get_cached_session().get(
            self.url_query, headers=self.header
        )
        if res.ok:
            self._json = res.json()
        else:
            logger.error(f"Error querying OncoKB API: {res.status_code} - {res.text}")
            self._json = None

    def has_json(self) -> dict | None:
        """Get the JSON response from the OncoKB API query.

        Returns
        -------
        dict | None
            JSON response if available, otherwise None

        """
        if self._json is None:
            logger.error("No data available. Please query the API first or fix query.")
            return False
        return True

    @staticmethod
    def extract_level_as_int(str_in: str | None) -> int | None:
        """Extract the level from a string and convert it to an integer.

        Parameters
        ----------
        str_in : str | None
            String containing the level information

        Returns
        -------
        int | None
            Level as integer if found, otherwise None

        """
        import re

        if str_in is None:
            return None
        try:
            return int(re.search(r"\d+", str_in).group())
        except (ValueError, IndexError):
            logger.error(f"Could not extract level from string: {str_in}")
            return None


@dataclass
class OncoKBProteinChange(OncoKB):
    """OncoKB API client for protein changes."""

    gene_name: str | None = None
    """Gene associated with the protein change."""
    alteration: str | None = None
    """Alteration type (e.g., V600E if BRAF is gene)."""
    dict_highest_level: dict[str, int] = field(
        default_factory=lambda: {
            "Sensitive": None,
            "Resistance": None,
            "Diagnostic_Implication": None,
            "Prognostic_Implication": None,
            "FDA": None,
        }
    )
    """Dictionary to store the highest level of evidence for each alteration."""
    list_treatment: list[str] = field(default_factory=list)
    """List of treatments associated with the alteration."""
    oncogenic: str | None = None
    """Oncogenic status of the alteration."""
    vus: bool | None = None
    """Whether the alteration is a Variant of Uncertain Significance (VUS)."""
    known_effect: str | None = None
    """Effect of the mutation on the protein (e.g., missense, nonsense)."""
    verbose: bool = True
    """Whether to log warnings for missing data."""

    def __post_init__(self):
        """Initialize the OncoKBProteinChange client."""
        if self.gene_name is None or self.alteration is None:
            logger.error("Gene name and alteration must be provided.")
        else:
            super().__post_init__()
            if not self.has_json():
                return

            json_data = self._json
            gene_exists = json_data["geneExist"]
            variant_exists = json_data["variantExist"]

            if gene_exists and variant_exists:
                self.annotate_highest_level()
                self.get_treatments()
                self.oncogenic = json_data.get("oncogenic", None)
                self.vus = json_data.get("vus", None)
                if "mutationEffect" in json_data:
                    self.known_effect = json_data["mutationEffect"].get(
                        "knownEffect", None
                    )
            elif self.verbose:
                if gene_exists:
                    msg = f"Alteration {self.alteration} does not exist for gene {self.gene_name} in OncoKB."
                if variant_exists:
                    msg = f"Gene {self.gene_name} does not exist in OncoKB for alteration {self.alteration}."
                else:
                    msg = f"Gene {self.gene_name} and alteration {self.alteration} does not exist in OncoKB."
                logger.error(msg)

    def update_url(self):
        """Update the URL for the OncoKB API query based on the UniProt ID."""
        if self.gene_name is None or self.alteration is None:
            logger.error("Gene name and alteration must be provided.")
        else:
            self.url_query = (
                f"{self.url}/annotate/mutations/byProteinChange"
                f"?hugoSymbol={self.gene_name}&alteration={self.alteration}"
            )

    def annotate_highest_level(self):
        """Annotate the highest level of evidence for the protein change."""
        for key in self.dict_highest_level.keys():
            key_orig = (
                "highest" + "".join([i.title() for i in key.split("_")]) + "Level"
            )
            if key_orig in self._json:
                level = self.extract_level_as_int(self._json[key_orig])
                if level is not None:
                    self.dict_highest_level[key] = level
            else:
                if self.verbose:
                    logger.warning(
                        f"No '{key_orig}' found in response for "
                        f"{self.gene_names}_{self.alteration}."
                    )

    def get_treatments(self):
        """Get the list of treatments associated with the alteration."""
        if "treatments" in self._json:
            try:
                self.list_treatment = [
                    [j["drugName"] for j in i["drugs"]]
                    for i in self._json["treatments"]
                ]
            except Exception as e:
                if self.verbose:
                    logger.error(
                        f"Error extracting treatments for "
                        f"{self.gene_name}_{self.alteration}: {e}"
                    )
        else:
            if self.verbose:
                logger.warning(
                    f"No 'treatments' found in response for "
                    f"{self.gene_name}_{self.alteration}."
                )
