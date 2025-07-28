import logging
import os
from dataclasses import dataclass, field
from io import StringIO
from tempfile import NamedTemporaryFile
from typing import Any

import py3Dmol
from Bio.PDB import MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from mkt.databases.colors import map_aa_to_single_letter_code
from mkt.databases.kinase_schema import KinaseInfo

logger = logging.getLogger(__name__)


LIST_KLIFS_STICK = [
    45,
    46,
    47,  # hinge region
    67,
    68,
    69,  # HRD motif
    79,
    80,
    81,
    82,  # xDFG motif
]
"""list[int]: Zero indexed location of residues to highlight as stick"""


@dataclass
class StructureVisualizer:
    """Class to visualize structures using py3Dmol."""

    obj_kinase: KinaseInfo
    """KinaseInfo object from which to extract sequences."""
    dict_align: dict[str, str | list[str]] = field(default_factory=dict)
    """Dict with keys DICT_ALIGNMENT containing seq and a list of colors per residue."""
    str_attr: str | None = None
    """Attribute to be highlighted in the structure."""
    bool_show: bool = False
    """Whether to show the structure in the viewer or return HTML."""
    dict_dims: dict[str, int] = field(
        default_factory=lambda: {"width": 600, "height": 600}
    )
    """Dimensions for the py3Dmol viewer."""
    dict_style: dict[str, str] = field(
        default_factory=lambda: {
            "None": "cartoon",
            "KLIFS": "cartoon",
            "Phosphosites": "stick",
            "lowlight": "cartoon",
        }
    )
    """Style for the py3Dmol viewer."""
    dict_color: dict[str, str] = field(
        default_factory=lambda: {
            "None": "spectrum",
            "Phosphosites": "red",
            "lowlight": "gray",
        }
    )
    """Color scheme for the py3Dmol viewer."""
    dict_opacity: dict[str, float] = field(
        default_factory=lambda: {
            "None": 1.0,
            "KLIFS": 1.0,
            "Phosphosites": 1.0,
            "lowlight": 0.5,
        }
    )
    """Opacity for the py3Dmol viewer."""

    def __post_init__(self):
        if self.str_attr == "None":
            self.str_attr = None
        self.structure = self.convert_mmcifdict2structure()
        self.pdb_text = self.convert_structure2string()
        self.residues = self.structure.get_residues()
        self.html = self.visualize_structure()

    @staticmethod
    def _parse_pdb_line(line: str) -> dict[str, Any] | None:
        """Parse a line from a PDB file and extract relevant information.

        Parameters
        ----------
        line : str
            Line from a PDB file.

        Returns
        -------
        dict[str, Any] | None
            Dictionary containing extracted information or None if the line does not match the criteria.

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

    def convert_mmcifdict2structure(self) -> Structure:
        """Convert MMCIF2Dict object back to mmCIF text format.

        Returns
        -------
        Structure
            mmCIF dictionary formatted as a MMCIFParser object

        """
        mmcif_io = MMCIFIO()
        mmcif_io.set_dict(self.obj_kinase.kincore.cif.cif)

        temp_string = StringIO()
        mmcif_io.save(temp_string)

        with NamedTemporaryFile(mode="w+", suffix=".cif", delete=False) as temp_file:
            temp_file.write(temp_string.getvalue())
            temp_file_name = temp_file.name

        parser = MMCIFParser()
        structure = parser.get_structure(self.obj_kinase.hgnc_name, temp_file_name)

        os.remove(temp_file_name)

        return structure

    def convert_structure2string(self) -> str:
        """Convert Bio.PDB.Structure.Structure object to pymol compatible

        Returns
        -------
        str
            Structure in string format.

        """
        pdb_io = PDBIO()
        pdb_io.set_structure(self.structure)
        pdb_string = StringIO()
        pdb_io.save(pdb_string)
        pdb_text = pdb_string.getvalue()

        return pdb_text

    def _index_stick_region(self) -> list[int]:
        """Extract the indices of the residues in the KLIFS pocket region by removing `-`.

        Returns
        -------
        list[int]
            List of indices for the residues in the KLIFS pocket region.

        """
        i = 0
        list_idx_stick = []
        for idx, res in enumerate(self.obj_kinase.klifs.pocket_seq):
            if idx in LIST_KLIFS_STICK and res != "-":
                list_idx_stick.append(idx - i)
            if res == "-":
                i += 1
        return list_idx_stick

    def _generate_highlight_idx(
        self,
    ) -> tuple[list[int], dict[str, str], dict[str, str]]:
        """Generate a list of indices for the residues to be highlighted.

        Returns
        -------
        list[int]
            List of indices for the residues to be highlighted.
        dict[str, str]
            Dictionary of colors for the residues to be highlighted.
        dict[str, str]
            Dictionary of styles for the residues to be highlighted.

        """
        str_seq_cif = self.dict_align["KinCore, CIF"]["str_seq"]
        list_cif_idx = [idx for idx, i in enumerate(str_seq_cif) if i != "-"]

        try:
            str_seq_attr = self.dict_align[self.str_attr]["str_seq"]
            list_attr_idx = [idx for idx, i in enumerate(str_seq_attr) if i != "-"]
        except KeyError:
            logger.error(f"{self.str_attr} not found in {self.dict_align.keys()}")

        list_intersect = list(set(list_cif_idx).intersection(set(list_attr_idx)))

        if self.str_attr == "KLIFS":
            # all KLIFS regions should fall within KinCore CIF region, but future-proofing
            list_idx_keep = [
                idx for idx, i in enumerate(list_intersect) if i in list_attr_idx
            ]
            list_idx_stick = self._index_stick_region()
            list_style = [
                "stick" if idx in list_idx_stick and idx in list_idx_keep else "cartoon"
                for idx, i in enumerate(list_attr_idx)
            ]
            list_color = [
                self.dict_align[self.str_attr]["list_colors"][i] for i in list_intersect
            ]
        else:
            list_style = [self.dict_style[self.str_attr] for i in list_intersect]
            list_color = [self.dict_color[self.str_attr] for i in list_intersect]

        # +1 for 1-based indexing
        list_intersect = [i + 1 for i in list_intersect]
        dict_color = dict(zip(list_intersect, list_color))
        dict_style = dict(zip(list_intersect, list_style))

        return list_intersect, dict_color, dict_style

    def _return_style_dict(
        self,
        str_key,
        str_color: str | None = None,
        str_style: str | None = None,
        float_opacity: float | None = None,
    ) -> dict[str, Any]:
        """Return the style dictionary for the given key.

        Parameters
        ----------
        str_key : str
            Key for the style dictionary.
        str_color : str, optional
            Color for the style dictionary, by default None.
        str_style : str, optional
            Style for the style dictionary, by default None.
        float_opacity : float, optional
            Opacity for the style dictionary, by default None.

        Returns
        -------
        dict[str, Any]
            Style dictionary for the given key.

        """
        if str_color is None:
            str_color = self.dict_color[str_key]
        if str_style is None:
            str_style = self.dict_style[str_key]
        if float_opacity is None:
            float_opacity = self.dict_opacity[str_key]

        dict_style = {
            str_style: {
                "color": str_color,
                "opacity": float_opacity,
            }
        }

        return dict_style

    def visualize_structure(self) -> str | None:
        """Visualize the structure using py3Dmol.

        Returns
        -------
        str
            HTML representation of the py3Dmol viewer or None if self.bool_show=True.

        """
        view = py3Dmol.view(**self.dict_dims)

        view.addModel(self.pdb_text, "pdb")

        if self.str_attr is None:
            str_attr = str(self.str_attr)
            view.setStyle(self._return_style_dict(str_attr))
        else:
            list_highlight, dict_color, dict_style = self._generate_highlight_idx()
            for i in self.residues:
                res_no = i.get_id()[1]
                # set lowlight background
                view.setStyle(
                    {"resi": str(res_no)},
                    self._return_style_dict("lowlight"),
                )
                # add highlights for the selected attribute
                if res_no in list_highlight:
                    # KLIFS uses KLIFS pocket colors
                    view.addStyle(
                        {"resi": str(res_no)},
                        self._return_style_dict(
                            str_key=self.str_attr,
                            str_color=dict_color[res_no],
                            str_style=dict_style[res_no],
                        ),
                    )

        view.zoomTo()
        if self.bool_show:
            view.show()
        else:
            return view._make_html()
