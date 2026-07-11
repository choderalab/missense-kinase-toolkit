"""Structure visualization backing the Streamlit app.

Provides :class:`StructureVisualizer`, which builds the interactive structure views
rendered in the Streamlit app.
"""

import logging
from typing import TYPE_CHECKING, Any

from Bio.PDB.Structure import Structure
from mkt.databases.colors import map_aa_to_single_letter_code
from mkt.databases.utils import convert_mmcifdict2structure, convert_structure2string

if TYPE_CHECKING:
    from mkt.databases.app.schema import StructureConfig

logger = logging.getLogger(__name__)


class StructureVisualizer:
    """Load and process kinase structures for visualization.

    This class handles structure loading from KinaseInfo CIF data and provides
    highlight data for visualization. Style/color logic is delegated to
    StructureConfig objects.

    Parameters
    ----------
    config : StructureConfig
        Configuration object containing seq_align (with kinase info) and
        pre-computed list_idx, list_color, list_style for highlighting.

    Attributes
    ----------
    config : StructureConfig
        The configuration object.
    obj_kinase : KinaseInfo
        KinaseInfo object from config.seq_align.obj_kinase.
    structure : Structure
        Bio.PDB Structure object loaded from CIF.
    pdb_text : str
        PDB-formatted string of the structure.
    residues : list
        List of residues from the structure.
    """

    def __init__(self, config: "StructureConfig"):
        self.config = config
        self.obj_kinase = config.seq_align.obj_kinase
        self.structure = self._convert_mmcifdict2structure()
        self.pdb_text = self._convert_structure2string()
        self.residues = list(self.structure.get_residues())

    @staticmethod
    def parse_pdb_line(line: str) -> dict[str, Any] | None:
        """Parse a line from a PDB file and extract relevant information.

        Parameters
        ----------
        line : str
            Line from a PDB file.

        Returns
        -------
        dict[str, Any] | None
            Dictionary containing extracted information or None if the line
            does not match the criteria (ATOM line with CA atom).
        """
        match = line.startswith("ATOM") and (line[13:15] == "CA")
        if match:
            list_line = [i for i in line.split(" ") if i != ""]
            dict_out = {
                "res_no": list_line[5],
                "res_name": map_aa_to_single_letter_code(list_line[3]),
                "coords": (
                    float(list_line[6]),
                    float(list_line[7]),
                    float(list_line[8]),
                ),
            }
            return dict_out
        else:
            return None

    def _convert_mmcifdict2structure(self) -> Structure:
        """Convert this kinase's MMCIF2Dict to a Bio.PDB Structure.

        Returns
        -------
        Structure
            Bio.PDB Structure object.
        """
        return convert_mmcifdict2structure(
            self.obj_kinase.kincore.cif.cif,
            structure_id=self.obj_kinase.hgnc_name,
        )

    def _convert_structure2string(self) -> str:
        """Convert this Bio.PDB Structure object to a PDB format string.

        Returns
        -------
        str
            Structure in PDB string format.
        """
        return convert_structure2string(self.structure)

    def get_highlight_data(
        self,
    ) -> tuple[list[int], dict[int, str], dict[int, str], dict[int, str | None]]:
        """Get highlight indices and color/style/label dictionaries for visualization.

        The config provides list_idx (1-indexed), list_color, list_style, and list_label.
        This method converts them to the dict format expected by consumers.

        Returns
        -------
        tuple[list[int], dict[int, str], dict[int, str], dict[int, str | None]]
            - list_highlight: List of 1-indexed residue positions to highlight.
            - dict_color: Mapping from residue position to color.
            - dict_style: Mapping from residue position to style.
            - dict_label: Mapping from residue position to label (None for no label).
        """
        list_highlight = self.config.list_idx
        dict_color = dict(zip(self.config.list_idx, self.config.list_color))
        dict_style = dict(zip(self.config.list_idx, self.config.list_style))
        dict_label = dict(zip(self.config.list_idx, self.config.list_label))

        return list_highlight, dict_color, dict_style, dict_label

    # Keep old method name as alias for backwards compatibility during transition
    def _generate_highlight_idx(
        self,
    ) -> tuple[list[int], dict[int, str], dict[int, str], dict[int, str | None]]:
        """Alias for get_highlight_data() for backwards compatibility.

        .. deprecated::
            Use get_highlight_data() instead.
        """
        return self.get_highlight_data()
