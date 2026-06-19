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


# columns flattened/derived from the raw JSON records
DICT_COLUMNS = ("variantAminoAcid", "tumorTypeComposition")
"""Record keys whose values are dicts (alt-AA counts, organ counts)."""


@dataclass
class CancerHotspots(RESTAPIClient):
    """Class to interact with the cancerhotspots.org single-residue hotspots API.

    Fetches the bulk single-residue hotspots payload for a given publication
    tier and exposes it as a tidy DataFrame (``_df``) alongside the raw JSON
    (``_json``). There is no working per-gene endpoint upstream, so the full
    payload is fetched once and filtered client-side via ``get_gene``.
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
            print(f"Error: {res.status_code}")
            self._json = None
            self._df = None

    @staticmethod
    def _to_dataframe(records: list[dict]) -> pd.DataFrame:
        """Flatten raw hotspot records into a tidy DataFrame.

        Parameters:
        -----------
        records : list[dict]
            Raw JSON records from the single-residue hotspots endpoint.

        Returns:
        --------
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

        Parameters:
        -----------
        hugo_symbol : str
            HGNC gene symbol to filter on (e.g. ``"BRAF"``).

        Returns:
        --------
        pd.DataFrame
            Rows of ``_df`` whose ``hugoSymbol`` matches; empty if none.
        """
        if self._df is None:
            return pd.DataFrame()
        return self._df[self._df["hugoSymbol"] == hugo_symbol].reset_index(drop=True)

    def to_csv(self, path: str) -> None:
        """Write ``_df`` to CSV, json-encoding the dict-valued columns.

        Parameters:
        -----------
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


def first_occurrence_map() -> dict[tuple[str, int], str]:
    """Map ``(hugoSymbol, positionStart)`` to the earliest hotspot tier.

    Builds the collapse logic described in the API notes: a residue position is
    labelled ``"Chang"`` if present in ``version=v2`` and ``"Bandlamudi 2026"``
    if it appears only in ``version=v3``. Positions absent from both are simply
    not keyed (the downstream consumer treats them as "Not a hotspot").

    Note: this keys on ``(hugoSymbol, positionStart)`` rather than the full
    ``residue`` string, because the downstream lollipop colors residue positions
    on its x-axis. As a result the "Bandlamudi 2026" tier holds 161 positions,
    not the 164 new ``(gene, residue)`` pairs reported by cancerhotspots.org:
    three v3-only residues (FOXA1 D249, MTOR Y1450, TP53 E224) sit at positions
    already called in v2, so they collapse into existing "Chang" positions. If
    you need the headline 164, key on ``residue`` instead — but then
    ``(gene, position)`` is no longer unique and a position can carry both tiers.

    Returns:
    --------
    dict[tuple[str, int], str]
        Mapping of ``(hugoSymbol, positionStart)`` to ``"Chang"`` or
        ``"Bandlamudi 2026"``.
    """
    chang = CancerHotspots(version=HotspotVersion.CHANG)
    bandlamudi = CancerHotspots(version=HotspotVersion.BANDLAMUDI)

    chang_keys = {
        (row.hugoSymbol, int(row.positionStart))
        for row in chang._df.itertuples(index=False)
    }

    occurrence: dict[tuple[str, int], str] = {}
    for row in bandlamudi._df.itertuples(index=False):
        key = (row.hugoSymbol, int(row.positionStart))
        occurrence[key] = "Chang" if key in chang_keys else "Bandlamudi 2026"

    return occurrence
