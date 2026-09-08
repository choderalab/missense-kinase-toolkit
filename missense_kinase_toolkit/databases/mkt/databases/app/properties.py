"""Kinase-level property tables backing the Streamlit app.

Provides :class:`PropertyTables`, which assembles the property summary tables rendered
in the Streamlit app.
"""

import logging
from dataclasses import dataclass

import pandas as pd
from mkt.schema.constants import (
    DICT_MOLECULAR_BRAKE,
    LIST_KLIFS_DFG_MOTIF,
    LIST_KLIFS_HRD_MOTIF,
    STR_KLIFS_BETA3_LYSINE,
)
from mkt.schema.kinase_schema import KinaseInfo, Provenance
from mkt.schema.utils import rgetattr

logger = logging.getLogger(__name__)


@dataclass
class PropertyTables:
    """Class to hold the property tables."""

    obj_kinase: KinaseInfo
    """KinaseInfo object from which to extract properties."""
    df_kinhub: pd.DataFrame | None = None
    """Dataframe containing the KinHub information."""
    df_klifs: pd.DataFrame | None = None
    """Dataframe containing the KLIFS information."""
    df_kincore: pd.DataFrame | None = None
    """Dataframe containing the KinCoRe information."""
    df_computed: pd.DataFrame | None = None
    """Dataframe containing the adjudicated/computed properties."""

    def __post_init__(self):
        """Post-initialization method to extract properties."""
        self.assign_properties()

    def convert_property2dataframe(
        self,
        str_attr: str,
        list_drop: list[str] | None = None,
        list_keep: list[str] | None = None,
    ) -> pd.DataFrame:
        """Convert the properties of the KinaseInfo object to a dataframe.

        Parameters
        ----------
        str_attr : str
            The attribute of the KinaseInfo object to convert to dataframe.
        list_drop : list[str], optional
            The list of attributes to drop from the dataframe, by default None.
            If provided, these attributes will be dropped from the dataframe.
        list_keep : list[str], optional
            The list of attributes to keep in the dataframe, by default None.
            If provided, only these attributes will be kept in the dataframe.

        Returns
        -------
        pd.DataFrame
            The dataframe containing the properties of the KinaseInfo object.
        """
        try:
            obj_temp = rgetattr(self.obj_kinase, str_attr)

            # copy so list_drop's `del` does not mutate the cached KinaseInfo object
            dict_temp = dict(obj_temp.__dict__)

            # drop or keep specified attributes
            if list_drop is not None:
                for str_drop in list_drop:
                    del dict_temp[str_drop]

            # keep only specified attributes
            if list_keep is not None:
                dict_temp = {k: dict_temp[k] for k in list_keep if k in dict_temp}

            # convert to dataframe
            df_temp = pd.DataFrame.from_dict(dict_temp, orient="index")
            df_temp.index = df_temp.index.map(lambda x: x.replace("_", " ").upper())
            df_temp.columns = ["Property"]

            return df_temp

        except Exception as e:
            logger.error(f"Error converting properties to dataframe: {e}")
            return None

    def assign_properties(self):
        """Extract the properties from the KinaseInfo object.

        Returns
        -------
        None
            The properties are extracted and stored in the class.
        """
        self.df_kinhub = self.convert_property2dataframe("kinhub")

        self.df_klifs = self.convert_property2dataframe(
            "klifs", list_drop=["pocket_seq"]
        )

        self.df_kincore = self.convert_property2dataframe(
            "kincore.fasta",
            list_keep=["group", "hgnc", "swissprot", "uniprot", "source"],
        )

        self.df_computed = self.build_computed_table()

        self.format_property_columns()

    def _residue_index(self, label: str) -> str | None:
        """Return the residue + UniProt index at a KLIFS ``region:idx`` label (e.g. "D855").

        Handles a trailing signed offset on the label (e.g. "VIII:79-1"). Returns None when
        no KLIFS mapping is available or the position is unmapped.
        """
        k2u = self.obj_kinase.KLIFS2UniProtIdx
        if not k2u:
            return None
        base, offset = label, 0
        for sign, mult in (("-", -1), ("+", 1)):
            head, sep, num = label.rpartition(sign)
            if sep and num.isdigit():
                base, offset = head, mult * int(num)
                break
        idx = k2u.get(base)
        if idx is None:
            return None
        idx += offset
        return f"{self.obj_kinase.uniprot.canonical_seq[idx - 1]}{idx}"

    def _motif(self, labels) -> str | None:
        """Join per-position residue+index into a motif string (e.g. "H835-R836-D837")."""
        parts = [self._residue_index(label) for label in labels]
        return "-".join(p or "-" for p in parts) if any(parts) else None

    def build_computed_table(self) -> pd.DataFrame | None:
        """Assemble the adjudicated/computed-property table for the kinase.

        Surfaces the classification flags (``is_pseudokinase``/``is_pseudogene``/
        ``is_lipid_kinase``) and the pocket/activation-loop motifs -- the catalytic Lys, HRD,
        DFG (KLIFS pocket), APE (Dunbrack MSA), and molecular-brake positions -- as
        ``residue+UniProt index`` strings (e.g. "A227-P228-E229"). The canonical identity is
        implied by the motif name; only the molecular-brake triad states its canonical (N-E-K)
        in the label.

        Returns
        -------
        pd.DataFrame | None
            A single-column ("Property") table indexed by property label, or None on error.
        """
        try:
            obj = self.obj_kinase
            dict_computed: dict[str, str] = {
                "is_pseudokinase": str(obj.is_pseudokinase()),
                "is_pseudogene": str(obj.is_pseudogene()),
                "is_lipid_kinase": str(obj.is_lipid_kinase()),
            }

            for label, motif in [
                ("catalytic Lys", self._residue_index(STR_KLIFS_BETA3_LYSINE)),
                ("HRD motif", self._motif(LIST_KLIFS_HRD_MOTIF)),
                ("DFG motif", self._motif(LIST_KLIFS_DFG_MOTIF)),
            ]:
                if motif is not None:
                    dict_computed[label] = motif

            list_ape = obj.adjudicate_ape()
            seq = obj.uniprot.canonical_seq
            dict_computed["APE motif"] = (
                "-".join("-" if i is None else f"{seq[i - 1]}{i}" for i in list_ape)
                if list_ape is not None
                else "None"
            )

            # molecular brake states its canonical triad (N-E-K) in the label
            brake_canonical = "-".join(DICT_MOLECULAR_BRAKE.values())
            dict_computed[f"molecular brake ({brake_canonical})"] = (
                self._motif(DICT_MOLECULAR_BRAKE.keys()) or "None"
            )

            df_temp = pd.DataFrame.from_dict(
                dict_computed, orient="index", columns=["Property"]
            )
            df_temp.index = df_temp.index.map(lambda x: x.replace("_", " ").upper())
            return df_temp

        except Exception as e:
            logger.error(f"Error building computed property table: {e}")
            return None

    @staticmethod
    def _format_property_value(value) -> str:
        """Render a single property value as a display string.

        Parameters
        ----------
        value : Any
            The raw attribute value from the KinaseInfo sub-object.

        Returns
        -------
        str
            String representation; iterables are comma-joined and None becomes "".
        """
        if value is None:
            return ""
        if isinstance(value, Provenance):
            # short citation, linked to the DOI when present (rendered via the Styler HTML)
            head = value.citation or value.name
            if value.doi:
                return f'<a href="{value.doi}" target="_blank">{head}</a>'
            return head
        if isinstance(value, (list, tuple, set, frozenset)):
            return ", ".join(str(v) for v in value)
        return str(value)

    def format_property_columns(self) -> None:
        """Stringify the ``Property`` column of each table for ``st.table``.

        The property tables collapse a kinase's heterogeneous attributes (str,
        int, list, set, ...) into a single column, which pyarrow cannot serialize
        to an Arrow table. Coercing every value to a string yields a uniform,
        Arrow-compatible column.

        Returns
        -------
        None
            The ``Property`` column of each populated table is modified in place.
        """
        for df_temp in (self.df_kinhub, self.df_klifs, self.df_kincore):
            if df_temp is not None and "Property" in df_temp.columns:
                df_temp["Property"] = df_temp["Property"].map(
                    self._format_property_value
                )
