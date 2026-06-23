import json
import logging
from enum import Enum

import pandas as pd
from mkt.databases import requests_wrapper
from mkt.databases.api_schema import RESTAPIClient
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


class HotspotVersion(str, Enum):
    """cancerhotspots.org publication tiers exposed by the API.

    Note: V1 and V2 return identical payloads from the API; V2 is a strict
    subset of V3 (V3 adds the 164 new Bandlamudi 2026 hotspots).
    """

    CHANG = "v2"
    """Chang 2016/2017 (the API returns identical payloads for v1 and v2)."""
    BANDLAMUDI = "v3"
    """Bandlamudi 2026 (== v2 plus the 164 newly called hotspots)."""


class HotspotTier(str, Enum):
    """Earliest publication that called a given hotspot."""

    CHANG = "Chang"
    """Present in the Chang tier (API ``version=v2``)."""
    BANDLAMUDI = "Bandlamudi 2026"
    """First called in the Bandlamudi 2026 tier (API ``version=v3`` only)."""


# record keys whose values are dicts (alt-AA counts, organ counts)
DICT_COLUMNS = ("variantAminoAcid", "tumorTypeComposition")
"""Record keys whose values are dicts; json-encoded when writing to CSV."""

TIER_COLUMN = "tier"
"""Name of the row-level annotation column added by :class:`CancerHotspots`."""


@dataclass
class CancerHotspotsQuery(RESTAPIClient):
    """Single-version query against the cancerhotspots.org single-residue API.

    One instance == one network call for one publication tier. There is no
    working per-gene endpoint upstream, so the full payload (~1300 records) is
    fetched once and filtered client-side via :meth:`get_gene`. The raw JSON is
    kept in ``_json`` and a tidy table in ``_df``.

    For the harmonized, tier-annotated table combining both versions, use
    :class:`CancerHotspots` instead.
    """

    version: HotspotVersion = HotspotVersion.BANDLAMUDI
    """Publication tier to query; defaults to the most inclusive (Bandlamudi 2026)."""
    url: str = "https://www.cancerhotspots.org/api/hotspots/single"
    """URL for the cancerhotspots.org single-residue hotspots API."""

    def __post_init__(self):
        self.query_api()

    def query_api(self) -> None:
        """Query the cancerhotspots.org API and populate ``_json`` and ``_df``."""
        res = requests_wrapper.get_cached_session().get(
            self.url,
            params={"version": self.version.value},
        )
        self._stamp_from_response(res)
        self.check_response(res)

        if res.ok:
            self._json = res.json()
            self._df = self._to_dataframe(self._json)
        else:
            logger.error("Error: %s", res.status_code)
            self._json = None
            self._df = None

    @staticmethod
    def _to_dataframe(records: list[dict]) -> pd.DataFrame:
        """Flatten raw hotspot records into a tidy DataFrame.

        Parameters
        ----------
        records : list[dict]
            Raw JSON records from the single-residue hotspots endpoint.

        Returns
        -------
        pd.DataFrame
            One row per record; ``aminoAcidPosition`` is flattened into
            ``positionStart``/``positionEnd`` integer columns and the dict-valued
            columns (variant amino acids, tumor type composition) are retained.
        """
        df = pd.DataFrame(records)
        if df.empty:
            return df

        position = pd.json_normalize(df["aminoAcidPosition"])
        df["positionStart"] = position["start"].astype("Int64")
        df["positionEnd"] = position["end"].astype("Int64")
        df = df.drop(columns=["aminoAcidPosition"])

        return df

    def get_gene(self, hugo_symbol: str) -> pd.DataFrame:
        """Return hotspot records for a single gene.

        Parameters
        ----------
        hugo_symbol : str
            HGNC gene symbol to filter on (e.g. ``"BRAF"``).

        Returns
        -------
        pd.DataFrame
            Rows of ``_df`` whose ``hugoSymbol`` matches; empty if none.
        """
        if self._df is None:
            return pd.DataFrame()
        return self._df[self._df["hugoSymbol"] == hugo_symbol].reset_index(drop=True)


@dataclass
class CancerHotspots:
    """Harmonize cancerhotspots.org publication tiers into one annotated table.

    Combines the Chang (``v2``) and Bandlamudi 2026 (``v3``) queries and, because
    ``v2`` is a strict subset of ``v3``, builds a single DataFrame (``_df``) from
    the ``v3`` superset with a row-level :data:`TIER_COLUMN` annotation: each
    record is labelled :attr:`HotspotTier.CHANG` if its ``(hugoSymbol, residue)``
    also appears in ``v2``, otherwise :attr:`HotspotTier.BANDLAMUDI`. This is the
    main entry point; the per-version :class:`CancerHotspotsQuery` objects remain
    available via :attr:`query_chang` / :attr:`query_bandlamudi` for provenance
    (``query_datetime`` / ``from_cache``).

    The row-level annotation is residue-keyed, so the Bandlamudi 2026 tier holds
    the 164 ``(gene, residue)`` pairs reported by cancerhotspots.org. For a
    position-keyed collapse (e.g. coloring a lollipop x-axis), use
    :meth:`first_occurrence_map`, which yields 161 — see its docstring.
    """

    def __post_init__(self):
        self.query_chang = CancerHotspotsQuery(version=HotspotVersion.CHANG)
        self.query_bandlamudi = CancerHotspotsQuery(version=HotspotVersion.BANDLAMUDI)
        self._df = self._annotate()

    def _annotate(self) -> pd.DataFrame | None:
        """Annotate the v3 superset with a row-level publication tier."""
        df_chang = self.query_chang._df
        df_bandlamudi = self.query_bandlamudi._df
        if df_chang is None or df_bandlamudi is None:
            logger.warning(
                "One or both tier queries returned no data; cannot annotate."
            )
            return None

        chang_residues = set(zip(df_chang["hugoSymbol"], df_chang["residue"]))
        df = df_bandlamudi.copy()
        df[TIER_COLUMN] = [
            (
                HotspotTier.CHANG.value
                if (gene, residue) in chang_residues
                else HotspotTier.BANDLAMUDI.value
            )
            for gene, residue in zip(df["hugoSymbol"], df["residue"])
        ]
        return df

    @property
    def df(self) -> pd.DataFrame | None:
        """Harmonized, tier-annotated DataFrame (one row per ``v3`` record)."""
        return self._df

    def get_gene(self, hugo_symbol: str) -> pd.DataFrame:
        """Return tier-annotated hotspot records for a single gene.

        Parameters
        ----------
        hugo_symbol : str
            HGNC gene symbol to filter on (e.g. ``"BRAF"``).

        Returns
        -------
        pd.DataFrame
            Rows of :attr:`df` whose ``hugoSymbol`` matches; empty if none.
        """
        if self._df is None:
            return pd.DataFrame()
        return self._df[self._df["hugoSymbol"] == hugo_symbol].reset_index(drop=True)

    def first_occurrence_map(
        self, single_residue_only: bool = False
    ) -> dict[tuple[str, int], str]:
        """Map ``(hugoSymbol, positionStart)`` to the earliest hotspot tier.

        Collapses the residue-level :attr:`df` to one label per residue
        *position*: :attr:`HotspotTier.CHANG` if any record at that position is in
        the Chang tier, otherwise :attr:`HotspotTier.BANDLAMUDI`. Positions absent
        from both tiers are simply not keyed (treat as "not a hotspot").

        This keys on position rather than the full ``residue`` string because the
        downstream lollipop colors residue positions on its x-axis. As a result
        the Bandlamudi 2026 tier holds 161 positions, not the 164 new
        ``(gene, residue)`` pairs in :attr:`df`: three v3-only residues (FOXA1
        D249, MTOR Y1450, TP53 E224) sit at positions already called in v2, so
        they collapse into existing Chang positions. Use :attr:`df` (residue
        level) if you need the headline 164.

        Parameters
        ----------
        single_residue_only : bool, optional
            If True, exclude in-frame indel records (``type != "single residue"``)
            before collapsing, so the map only covers single-residue hotspots.
            Default False (all records).

        Returns
        -------
        dict[tuple[str, int], str]
            Mapping of ``(hugoSymbol, positionStart)`` to a :class:`HotspotTier`
            value; empty if the harmonized table is unavailable.
        """
        if self._df is None:
            return {}

        df = self._df
        if single_residue_only:
            df = df[df["type"] == "single residue"]

        occurrence: dict[tuple[str, int], str] = {}
        for row in df.itertuples(index=False):
            key = (row.hugoSymbol, int(row.positionStart))
            # Chang wins at a position even if a v3-only residue shares it
            if occurrence.get(key) == HotspotTier.CHANG.value:
                continue
            occurrence[key] = row.tier
        return occurrence

    def to_csv(self, path: str) -> None:
        """Write the harmonized table to CSV, json-encoding dict-valued columns.

        Parameters
        ----------
        path : str
            Output CSV path.
        """
        if self._df is None:
            logger.warning("No data to write; query returned no records.")
            return
        df = self._df.copy()
        for col in DICT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )
        df.to_csv(path, index=False)

    @staticmethod
    def read_csv(path: str) -> pd.DataFrame:
        """Read a harmonized table written by :meth:`to_csv` back into a DataFrame.

        Inverts :meth:`to_csv`: json-decodes the dict-valued columns and restores
        the nullable integer position columns.

        Parameters
        ----------
        path : str
            Path to a CSV previously written by :meth:`to_csv`.

        Returns
        -------
        pd.DataFrame
            Tier-annotated table equivalent to :attr:`df`.
        """
        df = pd.read_csv(path)
        for col in DICT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )
        for col in ("positionStart", "positionEnd"):
            if col in df.columns:
                df[col] = df[col].astype("Int64")
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "CancerHotspots":
        """Build a client from an already-harmonized table without querying the API.

        Bypasses ``__post_init__`` (which would issue the per-tier network
        queries) via ``object.__new__`` and sets ``_df`` directly, so cached data
        can be reloaded offline. The per-tier provenance objects
        (``query_chang`` / ``query_bandlamudi``) are set to None.

        Parameters
        ----------
        df : pd.DataFrame
            Tier-annotated table (e.g. from :meth:`read_csv` or :attr:`df`).

        Returns
        -------
        CancerHotspots
            Instance backed by ``df``; :meth:`get_gene` and
            :meth:`first_occurrence_map` work as usual.
        """
        obj = object.__new__(cls)
        obj.query_chang = None
        obj.query_bandlamudi = None
        obj._df = df
        return obj

    @classmethod
    def from_csv(cls, path: str) -> "CancerHotspots":
        """Build a client from a CSV written by :meth:`to_csv`, without querying.

        Parameters
        ----------
        path : str
            Path to a CSV previously written by :meth:`to_csv`.

        Returns
        -------
        CancerHotspots
            Instance backed by the cached table.
        """
        return cls.from_dataframe(cls.read_csv(path))
