import logging
import os
from dataclasses import dataclass, field
from io import StringIO
from tempfile import NamedTemporaryFile
from typing import Any

import webcolors
from Bio.PDB import MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from mkt.databases.colors import map_aa_to_single_letter_code
from mkt.databases.kinase_schema import KinaseInfo

logger = logging.getLogger(__name__)


LIST_KLIFS_STICK_MANUAL = [
    45,  # hinge region
    46,
    47,
    67,  # HRD motif
    68,
    69,
    79,  # xDFG motif
    80,
    81,
    82,
]
"""list[int]: Zero indexed location of residues to highlight as stick; manually curated."""

LIST_KLIFS_STICK_CONSERVED = [
    3,  # g.l:4
    5,  # g.l:6
    8,  # g.l:9
    68,  # c.l:69 (R in HRD)
    80,  # xDFG:81 (D in xDFG)
]
"""list[int]: Zero indexed location of residues to highlight as stick; conserved residues."""

DICT_VIZ_STYLE = {
    "None": "cartoon",
    "Phosphosites": "stick",
    "lowlight": "cartoon",
}
"""dict[str, str]: Style for the py3Dmol viewer."""

DICT_VIZ_COLOR = {
    "None": "spectrum",
    "Phosphosites": "red",
    "Mutations": "red",
    "lowlight": "gray",
}
"""dict[str, str]: Color scheme for the py3Dmol viewer."""


@dataclass
class StructureVisualizer:
    """Class to visualize structures using py3Dmol."""

    obj_kinase: KinaseInfo
    """KinaseInfo object from which to extract sequences."""
    dict_align: dict[str, str | list[str]] = field(default_factory=dict)
    """Dict with keys DICT_ALIGNMENT containing seq and a list of colors per residue."""
    str_attr: str | None = None
    """Attribute to be highlighted in the structure."""
    bool_klifs_manual: bool = True
    """Whether to use manually curated KLIFS stick residues or the conserved ones."""
    dict_style: dict[str, str] = field(default_factory=lambda: DICT_VIZ_STYLE)
    """Style for the py3Dmol viewer."""
    dict_color: dict[str, str] = field(default_factory=lambda: DICT_VIZ_COLOR)
    """Color scheme for the py3Dmol viewer."""
    dict_mutations: dict[int, float] | None = None
    """Dictionary of mutations indices with corresponding normalized counts."""

    def __post_init__(self):
        if self.str_attr == "None":
            self.str_attr = None
        self.structure = self.convert_mmcifdict2structure()
        self.pdb_text = self.convert_structure2string()
        self.residues = self.structure.get_residues()

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
        list_klifs_sticks = (
            LIST_KLIFS_STICK_MANUAL
            if self.bool_klifs_manual
            else LIST_KLIFS_STICK_CONSERVED
        )
        i = 0
        list_idx_stick = []
        for idx, res in enumerate(self.obj_kinase.klifs.pocket_seq):
            if idx in list_klifs_sticks and res != "-":
                list_idx_stick.append(idx - i)
            if res == "-":
                i += 1
        return list_idx_stick

    def _index_mutation_region(self) -> list[int]:
        """Extract the indices of the residues in the mutation region.

        Returns
        -------
        list[int]
            List of indices for the residues in the mutation region.

        """
        if self.dict_mutations is None:
            logger.warning(
                "dict_mutations is None, cannot extract mutation region indices."
            )
            return []

        # extract indices of top 10 mutations (allowing for more in case of ties)
        dict_temp = dict(
            sorted(self.dict_mutations.items(), key=lambda item: item[1], reverse=True)
        )
        values = list(dict_temp.values())
        threshold = values[min(9, len(values) - 1)] if values else float("-inf")
        list_idx_mutation = [k - 1 for k, v in dict_temp.items() if v >= threshold]

        return list_idx_mutation

    @staticmethod
    def _convert_color_to_hex(color: str) -> str:
        """Convert named color to hex.

        Parameters
        ----------
        color : str
            Color name or hex string.

        Returns
        -------
        str
            Hex color string.
        """
        if color.startswith("#"):
            return color

        try:
            return webcolors.name_to_hex(color)
        except ValueError:
            logger.warning(f"Color '{color}' not recognized, defaulting to gray")
            return "#808080"

    @staticmethod
    def _interpolate_color(
        norm_value: float, start_color_hex: str, end_color_hex: str
    ) -> str:
        """Interpolate between two colors based on normalized value.

        Parameters
        ----------
        norm_value : float
            Normalized value between 0 and 1.
        start_color_hex : str
            Starting color in hex format (e.g., "#FFFFFF").
        end_color_hex : str
            Ending color in hex format (e.g., "#FF0000").

        Returns
        -------
        str
            Interpolated color in hex format.
        """
        # convert hex to RGB
        start_r = int(start_color_hex[1:3], 16)
        start_g = int(start_color_hex[3:5], 16)
        start_b = int(start_color_hex[5:7], 16)

        end_r = int(end_color_hex[1:3], 16)
        end_g = int(end_color_hex[3:5], 16)
        end_b = int(end_color_hex[5:7], 16)

        # interpolate
        interp_r = int(start_r + (end_r - start_r) * norm_value)
        interp_g = int(start_g + (end_g - start_g) * norm_value)
        interp_b = int(start_b + (end_b - start_b) * norm_value)

        return f"#{interp_r:02x}{interp_g:02x}{interp_b:02x}"

    def _generate_klifs_style_color_lists(
        self, list_intersect, list_attr_idx
    ) -> tuple[list[str], list[str]]:
        """Generate a list of indices for the residues to be highlighted in KLIFS.

        Parameters
        ----------
        list_intersect : list[int]
            List of intersecting indices between CIF and attribute sequences.
        list_attr_idx : list[int]
            List of indices for the residues in the attribute sequence.

        Returns
        -------
        tuple[list[str], list[str]]
            List of styles for the residues to be highlighted,
            List of colors for the residues to be highlighted.
        """
        # all KLIFS regions should fall within KinCore CIF region, but future-proofing
        list_idx_keep = [
            idx for idx, i in enumerate(list_intersect) if i in list_attr_idx
        ]
        list_idx_stick = self._index_stick_region()
        list_style = [
            "stick" if idx in list_idx_stick and idx in list_idx_keep else "cartoon"
            for idx, _ in enumerate(list_attr_idx)
        ]
        list_color = [
            self.dict_align[self.str_attr]["list_colors"][i] for i in list_intersect
        ]

        return list_style, list_color

    def _generate_mutations_style_color_lists(
        self,
        list_intersect,
        list_norm_count,
    ) -> tuple[list[str], list[str]]:
        """Generate a list of indices for the residues to be highlighted for Mutations.

        Parameters
        ----------
        list_intersect : list[int]
            List of intersecting indices between CIF and mutations (should be same).
        list_norm_count : list[float]
            List of normalized counts for the mutations.

        Returns
        -------
        tuple[list[str], list[str]]
            List of styles for the residues to be highlighted,
            List of colors for the residues to be highlighted.
        """
        assert len(list_intersect) == len(
            list_norm_count
        ), "Length of intersecting indices and normalized counts must be the same."

        list_idx_stick = self._index_mutation_region()
        list_style = [
            "stick" if i in list_idx_stick else "cartoon" for i in list_intersect
        ]

        # get target color from dict_color and convert to hex
        target_color = self.dict_color[self.str_attr]
        target_color_hex = self._convert_color_to_hex(target_color)

        # calculate start color (x% between white and target)
        start_color_hex = self._interpolate_color(0.25, "#FFFFFF", target_color_hex)

        # threshold for piecewise scaling
        threshold = 0.2

        # gradient from start to target color; lightgray for 0
        list_color = []
        for norm_count in list_norm_count:
            if norm_count == 0:
                list_color.append("lightgray")
            else:
                # piecewise scaling: compress values below threshold, expand above
                if norm_count < threshold:
                    # map 0-0.2 to 0-0.3 (compressed, stays lighter)
                    scaled_count = (norm_count / threshold) * 0.3
                else:
                    # map 0.2-1.0 to 0.3-1.0 (expanded, gets darker faster)
                    scaled_count = (
                        0.3 + ((norm_count - threshold) / (1.0 - threshold)) * 0.7
                    )

                # interpolate between start and target color
                color_hex = self._interpolate_color(
                    scaled_count, start_color_hex, target_color_hex
                )
                list_color.append(color_hex)

        return list_style, list_color

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

        if self.str_attr != "Mutations":
            try:
                str_seq_attr = self.dict_align[self.str_attr]["str_seq"]
                list_attr_idx = [idx for idx, i in enumerate(str_seq_attr) if i != "-"]
            except KeyError:
                logger.error(f"{self.str_attr} not found in {self.dict_align.keys()}")
        else:
            list_attr_idx = [i - 1 for i in self.dict_mutations.keys()]
            assert all(
                i in list_cif_idx for i in list_attr_idx
            ), "Some mutation indices are not present in the KinCore CIF sequence."
            list_norm_count = list(self.dict_mutations.values())

        list_intersect = sorted(
            list(set(list_cif_idx).intersection(set(list_attr_idx)))
        )

        if self.str_attr == "KLIFS":
            list_style, list_color = self._generate_klifs_style_color_lists(
                list_intersect, list_attr_idx
            )
        elif self.str_attr == "Mutations":
            list_style, list_color = self._generate_mutations_style_color_lists(
                list_intersect,
                list_norm_count,
            )
        else:
            list_style = [self.dict_style[self.str_attr] for _ in list_intersect]
            list_color = [self.dict_color[self.str_attr] for _ in list_intersect]

        # +1 for 1-based indexing
        list_intersect = [i + 1 for i in list_intersect]
        dict_color = dict(zip(list_intersect, list_color))
        dict_style = dict(zip(list_intersect, list_style))

        return list_intersect, dict_color, dict_style
