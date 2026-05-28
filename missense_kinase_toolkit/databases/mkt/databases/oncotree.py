import logging
import re
from os import path

import pandas as pd
import requests
from mkt.schema.io_utils import get_repo_root
from pydantic import BaseModel, PrivateAttr

logger = logging.getLogger(__name__)


DEFAULT_FILENAME = "tumor_types.txt"
"""Default filename for the OncoTree dump, relative to ``<repo_root>/data``."""

DEFAULT_URL = "https://oncotree.mskcc.org/api/tumor_types.txt"
"""Upstream URL for the OncoTree dump; used when no local file is found."""

LEVEL_PREFIX = "level_"
"""Prefix used by OncoTree's hierarchy columns; count varies by snapshot."""

META_COLS = ["metamaintype", "metacolor", "metanci", "metaumls", "history"]
"""Metadata columns in the raw OncoTree TSV."""

CODE_RE = re.compile(r"\s*\(([^()]+)\)\s*$")
"""Regex capturing the trailing ``(CODE)`` suffix on every OncoTree label."""


class OncoTree(BaseModel):
    """Loader for the OncoTree tumor-type hierarchy.

    Source: https://oncotree.mskcc.org/api/tumor_types.txt

    The raw TSV ships hierarchy columns (``level_1`` through ``level_N``;
    the upstream API currently emits six, older local snapshots emit seven),
    five metadata columns, and labels in the form ``"Display Name (CODE)"``.
    Upstream rows are *ragged*: shallow entries only emit tabs up to their
    deepest populated level, so a depth-1 row has six fields rather than
    eleven. The loader pads those rows so the metadata columns realign,
    then exposes a cleaned DataFrame via the ``df`` property with four
    derived columns added between the levels and metadata: ``depth``
    (count of populated level cells), ``name`` (the deepest label with
    its ``(CODE)`` suffix stripped), ``code`` (the OncoTree code parsed
    from that suffix), and ``parent`` (the immediately shallower level
    label, empty at depth 1).

    Parameters
    ----------
    filepath : str | None
        Optional path to the OncoTree TSV. If unset, the loader looks for
        ``<repo_root>/data/tumor_types.txt`` and falls back to fetching
        ``DEFAULT_URL`` when no local file exists.
    """

    filepath: str | None = None

    _df: pd.DataFrame = PrivateAttr()

    def model_post_init(self, __context: any) -> None:
        """Load and clean the OncoTree TSV into ``self._df``."""
        self._df = self._load()

    def _resolve_source(self) -> str:
        """Resolve the source path or URL to read.

        Returns
        -------
        str
            ``self.filepath`` if set; otherwise the repo-bundled default
            path if it exists on disk; otherwise ``DEFAULT_URL``.
        """
        if self.filepath is not None:
            return self.filepath
        local = path.join(get_repo_root(), "data", DEFAULT_FILENAME)
        if path.isfile(local):
            return local
        logger.info(
            "No local OncoTree dump at %s; fetching from %s",
            local,
            DEFAULT_URL,
        )
        return DEFAULT_URL

    @staticmethod
    def _read_text(source: str) -> str:
        """Return the raw TSV text from a local path or http(s) URL."""
        if source.startswith(("http://", "https://")):
            res = requests.get(source)
            res.raise_for_status()
            return res.text
        with open(source) as f:
            return f.read()

    @staticmethod
    def _parse_ragged_tsv(text: str) -> pd.DataFrame:
        """Parse the ragged OncoTree TSV into a rectangular DataFrame.

        Shallow rows omit trailing empty level cells, so the populated
        fields are ``[*levels, *metadata]`` with no padding between.
        Padding is inserted between the levels and the trailing metadata
        block so every row has ``len(header)`` cells.

        Parameters
        ----------
        text : str
            Raw TSV contents, including the header line.

        Returns
        -------
        pd.DataFrame
            String-typed DataFrame matching the TSV header.
        """
        n_meta = len(META_COLS)
        lines = [
            line for line in text.splitlines() if line and not line.startswith("#")
        ]
        header = lines[0].split("\t")
        n_total = len(header)
        n_levels = n_total - n_meta

        rows = []
        for line in lines[1:]:
            fields = line.split("\t")
            level_part = fields[:-n_meta]
            meta_part = fields[-n_meta:]
            level_part += [""] * (n_levels - len(level_part))
            rows.append(level_part + meta_part)

        return pd.DataFrame(rows, columns=header, dtype=str)

    def _load(self) -> pd.DataFrame:
        """Read the OncoTree TSV and derive ``depth``, ``name``, ``code``, ``parent``.

        Returns
        -------
        pd.DataFrame
            Columns, in order: ``level_1`` ... ``level_N``, ``depth``,
            ``name``, ``code``, ``parent``, then the five metadata columns.
            ``N`` is the number of ``level_*`` columns in the source file.
        """
        df = self._parse_ragged_tsv(self._read_text(self._resolve_source()))

        level_cols = [c for c in df.columns if c.startswith(LEVEL_PREFIX)]

        # count of populated level cells per row, e.g. 4 for a level_4 leaf
        df["depth"] = df[level_cols].ne("").sum(axis=1)

        # deepest populated label, e.g. "Adenosquamous Carcinoma of the Gallbladder (GBASC)"
        leaf = df.apply(lambda r: r[level_cols[r["depth"] - 1]], axis=1)

        # name = leaf with trailing "(CODE)" stripped
        df["name"] = leaf.str.replace(CODE_RE, "", regex=True).str.strip()
        # code = the captured contents of "(CODE)"
        df["code"] = leaf.str.extract(CODE_RE)

        # parent = the level just shallower than leaf; empty string at depth 1
        df["parent"] = df.apply(
            lambda r: r[level_cols[r["depth"] - 2]] if r["depth"] > 1 else "",
            axis=1,
        )

        return df[level_cols + ["depth", "name", "code", "parent"] + META_COLS]

    @property
    def df(self) -> pd.DataFrame:
        """Cleaned OncoTree DataFrame populated at instantiation."""
        return self._df
