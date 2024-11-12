import logging
import json
from enum import Enum
from pydantic.dataclasses import dataclass

from missense_kinase_toolkit.databases import requests_wrapper
from missense_kinase_toolkit.databases.api_schema import RESTAPIClient

logger = logging.getLogger(__name__)


class ScoreDatabase(str, Enum):
    """Enum class to define the score database."""
    Conservation = "CONSERV"
    EVE = "EVE"
    ESM1b = "ESM"
    AlphaMissense = "AM"


@dataclass
class ProtvarScore(RESTAPIClient):
    """Class to interact with Protvar API."""
    
    database: ScoreDatabase
    """Database to query for score: Conservation (CONSERV), EVE (EVE), ESM1b (ESM) and AlphaMissense (AM) scores."""
    uniprot_id: str
    """Uniprot ID."""
    pos: int
    """Position in the protein where mutation resides."""
    mut: str | None = None
    """Mutant residue (1 or 3 letter code); disregarded for Conservation score and optional for the other scores;
        if None will provide all ."""

    def __post_init__(self):
        self.url = "https://www.ebi.ac.uk/ProtVar/api/score/<UNIPROT>/<POS>?mt=<MUT>&name=<DATABASE>"
        self.create_query_url()
        self.query_api()

    def create_query_url(self):
        """Create URL for Protvar score API query."""

        if self.mut is None:
            mut_old = "mt=<MUT>&"
            mut_new = ""
        else:
            mut_old = "<MUT>"
            mut_new = self.mut

        self.url_query = (
            self.url
                .replace("<UNIPROT>", self.uniprot_id)
                .replace("<POS>", str(self.pos))
                .replace(mut_old, mut_new)
                .replace("<DATABASE>", self.database)
            )

    def query_api(self) -> dict:
        header = {"Accept": "application/json"}
        res = requests_wrapper.get_cached_session().get(self.url_query, headers=header)

        if res.ok:
            self._protvar_score = json.loads(res.text)
        else:
            print(f"Error: {res.status_code}")
            self._protvar_scores = None