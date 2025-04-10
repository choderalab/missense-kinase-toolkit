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


@dataclass
class StructureVisualizer:
    """Class to visualize structures using py3Dmol."""

    str_template: str = "1gag_template.pdb"
    """Path to the template structure."""
    list_rotate: list[float] = field(lambda: [-40, 0, 240])
    """List of rotation angles for the py3Dmol viewer."""
    dict_color: dict[str, str] = field(
        default_factory=lambda: {
            "normal": "spectrum",
            "highlight": "red",
            "lowlight": "gray",
        }
    )
    field(default_factory=dict)
    """Color dictionary for the py3Dmol viewer."""
    dict_dims: dict[str, int] = field(
        default_factory=lambda: {"width": 500, "height": 600}
    )
    """Dimensions for the py3Dmol viewer."""
    style: str = "cartoon"
    """Style for the py3Dmol viewer."""
    dict_opacity: dict[str, int] = field(
        default_factory=lambda: {"normal": 0.95, "highlight": 0.95, "lowlight": 0.2}
    )
    """Opacity for the py3Dmol viewer."""

    @staticmethod
    def _convert_mmcifdict2structure(
        mmcif_dict: dict[str, Any], str_id: str
    ) -> Structure:
        """Convert MMCIF2Dict object back to mmCIF text format

        Parameters
        ----------
        mmcif_dict : dict[str, str]
            Dictionary containing mmCIF data.

        Returns
        -------
        Structure
            mmCIF dictionary formatted as a

        """
        mmcif_io = MMCIFIO()
        mmcif_io.set_dict(mmcif_dict)

        temp_string = StringIO()
        mmcif_io.save(temp_string)

        with NamedTemporaryFile(mode="w+", suffix=".cif", delete=False) as temp_file:
            temp_file.write(temp_string.getvalue())
            temp_file_name = temp_file.name

        parser = MMCIFParser()
        structure = parser.get_structure(str_id, temp_file_name)

        os.remove(temp_file_name)

        return structure

    @staticmethod
    def _convert_structure2string(structure: Structure) -> str:
        """Convert Bio.PDB.Structure.Structure object to pymol compatible

        Parameters
        ----------
        structure : Structure
            Bio.PDB.Structure.Structure object.

        Returns
        -------
        str
            Structure in string format.

        """
        pdb_io = PDBIO()
        pdb_io.set_structure(structure)
        pdb_string = StringIO()
        pdb_io.save(pdb_string)
        pdb_text = pdb_string.getvalue()

        return pdb_text

    def visualize_structure(
        self,
        mmcif_dict: dict[str, Any],
        str_id: str,
        # list_res: list[str] = None,
        bool_show: bool = False,
    ) -> str:
        """Visualize the structure using py3Dmol.

        Parameters
        ----------
        mmcif_dict : dict[str, Any]
            Dictionary containing mmCIF data.

        str_id : str
            Identifier for the structure.

        Returns
        -------
        str
            HTML representation of the py3Dmol viewer.

        """
        structure = self._convert_mmcifdict2structure(mmcif_dict, str_id)
        pdb_text = self._convert_structure2string(structure)

        view = py3Dmol.view(
            width=self.dict_dims["width"], height=self.dict_dims["height"]
        )

        view.addModel(pdb_text, "pdb")

        # TODO: align to template
        # rotate the view so N-term up and C-helix to the right
        view.rotate(self.list_rotate[0], "x")
        view.rotate(self.list_rotate[1], "y")
        view.rotate(self.list_rotate[2], "z")

        # TODO: Add color and opacity via list_res
        view.setStyle({self.style: {"color": "spectrum"}})
        view.zoomTo()

        if bool_show:
            view.show()
        else:
            return view._make_html()
