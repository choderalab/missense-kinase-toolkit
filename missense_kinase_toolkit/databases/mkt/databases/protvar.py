import json
import logging
from collections import defaultdict
from dataclasses import field
from enum import Enum

from mkt.databases import requests_wrapper
from mkt.databases.api_schema import RESTAPIClient
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


class ScoreDatabase(str, Enum):
    """Enum class to define the score databases returned by ProtVar."""

    Conservation = "CONSERV"
    EVE = "EVE"
    ESM1b = "ESM"
    AlphaMissense = "AM"
    popEVE = "POPEVE"


DICT_SCORE_KEY = {
    ScoreDatabase.Conservation: "score",
    ScoreDatabase.EVE: "score",
    ScoreDatabase.ESM1b: "score",
    ScoreDatabase.AlphaMissense: "amPathogenicity",
}
"""dict[ScoreDatabase, str]: Key holding the numeric score within each database's \
    score dict; databases absent from this mapping (e.g. popEVE) have no single scalar score."""

DICT_CLASS_KEY = {
    ScoreDatabase.EVE: "eveClass",
    ScoreDatabase.AlphaMissense: "amClass",
}
"""dict[ScoreDatabase, str]: Key holding the categorical classification; only EVE \
    and AlphaMissense have one."""

TUPLE_VARIANT_DATABASES = (
    ScoreDatabase.EVE,
    ScoreDatabase.ESM1b,
    ScoreDatabase.AlphaMissense,
    ScoreDatabase.popEVE,
)
"""tuple[ScoreDatabase, ...]: Variant-level databases (one score per substitution); \
    Conservation is excluded as it is residue- rather than variant-level."""


def _coerce_database(database: "ScoreDatabase | str | None") -> "ScoreDatabase | None":
    """Coerce a ScoreDatabase or its value (e.g. "AM") to a ScoreDatabase, or None."""
    if isinstance(database, ScoreDatabase):
        return database
    try:
        return ScoreDatabase(database)
    except ValueError:
        return None


@dataclass
class ProtvarVariant:
    """Scores for a single variant (one mutant residue) at a position.

    Built by :class:`ProtvarScoreQuery`; holds the raw per-database score dict for
    this variant and exposes scalar accessors. Conservation, which is residue-
    rather than variant-level, is shared across all variants at the position.
    """

    mt: str | None
    """Mutant residue (1-letter code) this variant represents; None if unknown."""
    scores: dict = field(default_factory=dict)
    """Mapping of ScoreDatabase to its raw score dict for this variant."""

    def get_score(self, database: "ScoreDatabase | str"):
        """Return the numeric score for a database, or None if not present.

        Parameters
        ----------
        database : ScoreDatabase | str
            Database to retrieve, as a ScoreDatabase member or its value (e.g. "AM").

        Returns
        -------
        float | None
            The score ("score" for Conservation/EVE/ESM1b, "amPathogenicity" for
            AlphaMissense); None for popEVE (use :meth:`get_popeve`) or when the
            database is absent for this variant.
        """
        db = _coerce_database(database)
        key = DICT_SCORE_KEY.get(db) if db is not None else None
        entry = self.scores.get(db)
        if key is None or entry is None:
            return None
        return entry.get(key)

    def get_classification(self, database: "ScoreDatabase | str"):
        """Return the categorical classification for a database, or None.

        Parameters
        ----------
        database : ScoreDatabase | str
            Database to retrieve, as a ScoreDatabase member or its value (e.g. "AM").

        Returns
        -------
        str | None
            The classification ("eveClass" for EVE, "amClass" for AlphaMissense);
            None for databases without one (Conservation, ESM1b, popEVE) or when
            the database is absent for this variant.
        """
        db = _coerce_database(database)
        key = DICT_CLASS_KEY.get(db) if db is not None else None
        entry = self.scores.get(db)
        if key is None or entry is None:
            return None
        return entry.get(key)

    def get_popeve(self) -> dict | None:
        """Return the standalone popEVE payload for this variant, or None.

        Returns
        -------
        dict | None
            The full popEVE score dict (multiple keys, retained verbatim), or None
            if popEVE was not returned for this variant.
        """
        return self.scores.get(ScoreDatabase.popEVE)


@dataclass
class ProtvarScoreQuery(RESTAPIClient):
    """Class to interact with the ProtVar score API.

    A single query returns scores for every available database
    (Conservation, EVE, ESM1b, AlphaMissense, popEVE). The response is parsed
    into :class:`ProtvarVariant` objects exposed via :attr:`variants`:

    - ``mut`` supplied -> a single variant keyed by that residue.
    - ``mut`` is None  -> one variant per possible substitution (keyed by residue).

    Residue identity for the variant-level databases is recovered from popEVE,
    the only database whose payload carries the mutant residue (``mt``).
    """

    uniprot_id: str
    """Uniprot ID."""
    pos: int
    """Position in the protein where mutation resides."""
    mut: str | None = None
    """Mutant residue (1 or 3 letter code); optional. If None, scores for every
        possible substitution at this position are returned."""
    url: str = "https://www.ebi.ac.uk/ProtVar/api/score/<UNIPROT>/<POS>?mt=<MUT>"
    """URL for Protvar score API query."""
    header: dict = field(default_factory=lambda: {"Accept": "application/json"})
    """Header for the API request."""

    def __post_init__(self):
        self.create_query_url()
        self.query_api()

    def create_query_url(self):
        """Create URL for Protvar score API query."""

        if self.mut is None:
            mut_old = "?mt=<MUT>"
            mut_new = ""
        else:
            mut_old = "<MUT>"
            mut_new = self.mut

        self.url_query = (
            self.url.replace("<UNIPROT>", self.uniprot_id)
            .replace("<POS>", str(self.pos))
            .replace(mut_old, mut_new)
        )

    def query_api(self) -> None:
        """Query the ProtVar score API and parse the response into variants."""
        res = requests_wrapper.get_cached_session().get(
            self.url_query,
            headers=self.header,
        )
        self._stamp_from_response(res)

        if res.ok:
            parsed_scores = json.loads(res.text)
            # the api returns a list of score dicts (one per database when a
            # mutant is supplied); normalize a lone dict into a list
            self._protvar_scores_query = (
                parsed_scores if isinstance(parsed_scores, list) else [parsed_scores]
            )
        else:
            logger.error("Error: %s", res.status_code)
            self._protvar_scores_query = None

        self._build_variants()

    def _build_variants(self) -> None:
        """Parse the flat score list into per-variant ProtvarVariant objects."""
        grouped = defaultdict(list)
        for score in self._protvar_scores_query or []:
            db = _coerce_database(score.get("type"))
            if db is not None:
                grouped[db].append(score)

        # conservation is residue-level: a single entry shared by every variant
        conserv = grouped.get(ScoreDatabase.Conservation)
        conserv_entry = conserv[0] if conserv else None

        # recover the residue order; popEVE is the only variant-level database
        # whose payload carries the mutant residue (mt)
        popeve = grouped.get(ScoreDatabase.popEVE)
        if popeve:
            order = [entry.get("mt") for entry in popeve]
        elif self.mut is not None:
            order = [self.mut]
        else:
            order = list(
                range(
                    max((len(grouped[db]) for db in TUPLE_VARIANT_DATABASES), default=0)
                )
            )
            if order:
                logger.warning(
                    "popEVE absent for %s pos %s; cannot label variants by residue, "
                    "falling back to positional indices",
                    self.uniprot_id,
                    self.pos,
                )

        variants = {}
        for idx, mt in enumerate(order):
            scores = {}
            if conserv_entry is not None:
                scores[ScoreDatabase.Conservation] = conserv_entry
            for db in TUPLE_VARIANT_DATABASES:
                entries = grouped.get(db, [])
                if idx < len(entries):
                    scores[db] = entries[idx]
            variants[mt] = ProtvarVariant(mt=mt, scores=scores)
        self.variants = variants

    def get_variant(self, mt: str | None = None) -> ProtvarVariant | None:
        """Return a single variant by mutant residue.

        Parameters
        ----------
        mt : str | None, optional
            Mutant residue to retrieve. If None and the query targeted a single
            variant, that sole variant is returned; otherwise None.

        Returns
        -------
        ProtvarVariant | None
            The requested variant, or None if absent.
        """
        if mt is None:
            return (
                next(iter(self.variants.values())) if len(self.variants) == 1 else None
            )
        return self.variants.get(mt)

    def get_scores(self) -> list[dict] | None:
        """Return all raw score dicts retained from the query.

        Returns
        -------
        list[dict] | None
            All score dicts returned by the api, or None if the query failed.
        """
        return self._protvar_scores_query
