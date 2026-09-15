"""Loader and parser for the 3D Hotspots database.

Provides :class:`ThreeDHotspots` to load and parse the residue-level 3D hotspot
calls (``Table S2``) published by 3dhotspots.org, with the :class:`HotspotClass`
enumeration for the per-residue classification.

Unlike :mod:`mkt.databases.cancer_hotspots`, 3dhotspots.org exposes no API: the
data ship as a single XLSX workbook (``3d_hotspots.xls``). Following the pattern
in :mod:`mkt.databases.oncotree`, the loader reads a local copy under
``<repo_root>/data`` when present and otherwise downloads :data:`DEFAULT_URL`.
"""

import logging
import re
from enum import Enum
from io import BytesIO
from os import path

import pandas as pd
from mkt.databases import requests_wrapper
from mkt.schema.io_utils import get_repo_root
from pydantic import BaseModel, PrivateAttr

logger = logging.getLogger(__name__)


DEFAULT_FILENAME = "3d_hotspots.xls"
"""Default filename for the 3D Hotspots workbook, relative to ``<repo_root>/data``."""

DEFAULT_URL = "https://www.3dhotspots.org/files/3d_hotspots.xls"
"""Upstream URL for the 3D Hotspots workbook; used when no local file is found."""

RESIDUE_SHEET = "Table S2"
"""Workbook sheet holding the per-residue hotspot calls."""

LICENSE_COL_PREFIX = "Data available under ODC"
"""Prefix of the ODbL license-boilerplate column upstream appends to every sheet."""

COLUMN_RENAME = {
    "Gene": "hugoSymbol",
    "Residue": "residue",
    "#Mutations": "mutationCount",
    "p-value": "pValue",
    "Class": "hotspotClass",
}
"""Rename map from the raw ``Table S2`` headers to camelCase columns.

``hugoSymbol`` / ``residue`` mirror :mod:`mkt.databases.cancer_hotspots` so the
two hotspot resources share a column vocabulary.
"""

RESIDUE_RE = re.compile(r"^[A-Za-z]+(\d+)$")
"""Regex capturing the integer position from a ``"<ref-AA><position>"`` residue."""


class HotspotClass(str, Enum):
    """Per-residue 3D hotspot classification from ``Table S2``."""

    HOTSPOT = "Hotspot"
    """Recurrently mutated residue also called as a linear (single-residue) hotspot."""
    HOTSPOT_LINKED = "Hotspot-linked"
    """3D-cluster residue sharing a cluster with a linear hotspot residue."""
    CLUSTER_EXCLUSIVE = "Cluster-exclusive"
    """3D-cluster residue not otherwise called as a linear hotspot."""


class ThreeDHotspots(BaseModel):
    """Loader for the residue-level 3D hotspot calls from 3dhotspots.org.

    Source: https://www.3dhotspots.org/files/3d_hotspots.xls (``Table S2``)

    The workbook's ``Table S2`` sheet lists one row per hotspot residue with its
    gene, mutation count, p-value, and :class:`HotspotClass`. The loader drops the
    trailing ODbL license-boilerplate column, renames the headers to the camelCase
    vocabulary shared with :mod:`mkt.databases.cancer_hotspots`
    (``hugoSymbol`` / ``residue``), and derives a nullable-integer
    ``positionStart`` from each residue label. The cleaned table is exposed via the
    ``df`` property and filtered per gene with :meth:`get_gene`.

    Parameters
    ----------
    filepath : str | None
        Optional path to the 3D Hotspots workbook. If unset, the loader looks for
        ``<repo_root>/data/3d_hotspots.xls`` and falls back to downloading
        :data:`DEFAULT_URL` when no local file exists.
    """

    filepath: str | None = None

    _df: pd.DataFrame = PrivateAttr()

    def model_post_init(self, __context: any) -> None:
        """Load and clean ``Table S2`` into ``self._df``."""
        self._df = self._load()

    def _resolve_source(self) -> str:
        """Resolve the source path or URL to read.

        Returns
        -------
        str
            ``self.filepath`` if set; otherwise the repo-bundled default path if
            it exists on disk; otherwise :data:`DEFAULT_URL`.
        """
        if self.filepath is not None:
            return self.filepath
        local = path.join(get_repo_root(), "data", DEFAULT_FILENAME)
        if path.isfile(local):
            return local
        logger.info(
            "No local 3D Hotspots workbook at %s; downloading from %s",
            local,
            DEFAULT_URL,
        )
        return DEFAULT_URL

    @staticmethod
    def _read_bytes(source: str) -> bytes:
        """Return the raw workbook bytes from a local path or http(s) URL."""
        if source.startswith(("http://", "https://")):
            res = requests_wrapper.get_cached_session().get(source)
            res.raise_for_status()
            return res.content
        with open(source, "rb") as f:
            return f.read()

    def _load(self) -> pd.DataFrame:
        """Read ``Table S2`` and derive ``positionStart``.

        Returns
        -------
        pd.DataFrame
            Columns, in order: ``hugoSymbol``, ``residue``, ``positionStart``,
            ``mutationCount``, ``pValue``, ``hotspotClass``.
        """
        raw = pd.read_excel(
            BytesIO(self._read_bytes(self._resolve_source())), sheet_name=RESIDUE_SHEET
        )

        # drop the trailing ODbL license-boilerplate column
        raw = raw.loc[:, ~raw.columns.str.startswith(LICENSE_COL_PREFIX)]
        df = raw.rename(columns=COLUMN_RENAME)

        # position parsed from the "<ref-AA><position>" residue label, e.g. A34 -> 34
        df["positionStart"] = (
            df["residue"].astype(str).str.extract(RESIDUE_RE)[0].astype("Int64")
        )

        # p-value ships as a mix of floats and the string "<0.0001"; keep verbatim as text
        df["pValue"] = df["pValue"].astype(str)

        return df[
            [
                "hugoSymbol",
                "residue",
                "positionStart",
                "mutationCount",
                "pValue",
                "hotspotClass",
            ]
        ]

    @property
    def df(self) -> pd.DataFrame:
        """Cleaned residue-level 3D hotspot table populated at instantiation."""
        return self._df

    def get_gene(self, hugo_symbol: str) -> pd.DataFrame:
        """Return 3D hotspot residues for a single gene.

        Parameters
        ----------
        hugo_symbol : str
            HGNC gene symbol to filter on (e.g. ``"BRAF"``).

        Returns
        -------
        pd.DataFrame
            Rows of :attr:`df` whose ``hugoSymbol`` matches; empty if none.
        """
        return self._df[self._df["hugoSymbol"] == hugo_symbol].reset_index(drop=True)

    def to_csv(self, path: str) -> None:
        """Write the cleaned table to CSV.

        Parameters
        ----------
        path : str
            Output CSV path.
        """
        self._df.to_csv(path, index=False)

    @staticmethod
    def read_csv(path: str) -> pd.DataFrame:
        """Read a table written by :meth:`to_csv` back into a DataFrame.

        Restores the nullable-integer ``positionStart`` column and keeps
        ``pValue`` as text.

        Parameters
        ----------
        path : str
            Path to a CSV previously written by :meth:`to_csv`.

        Returns
        -------
        pd.DataFrame
            Cleaned table equivalent to :attr:`df`.
        """
        df = pd.read_csv(path, dtype={"pValue": str})
        if "positionStart" in df.columns:
            df["positionStart"] = df["positionStart"].astype("Int64")
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ThreeDHotspots":
        """Build a loader from an already-cleaned table without reading the workbook.

        Bypasses ``model_post_init`` (which would read/download the workbook) and
        sets ``_df`` directly, so cached data can be reloaded offline.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned table (e.g. from :meth:`read_csv` or :attr:`df`).

        Returns
        -------
        ThreeDHotspots
            Instance backed by ``df``; :meth:`get_gene` works as usual.
        """
        obj = cls.__new__(cls)
        BaseModel.__init__(obj)
        obj._df = df
        return obj

    @classmethod
    def from_csv(cls, path: str) -> "ThreeDHotspots":
        """Build a loader from a CSV written by :meth:`to_csv`, without downloading.

        Parameters
        ----------
        path : str
            Path to a CSV previously written by :meth:`to_csv`.

        Returns
        -------
        ThreeDHotspots
            Instance backed by the cached table.
        """
        return cls.from_dataframe(cls.read_csv(path))
