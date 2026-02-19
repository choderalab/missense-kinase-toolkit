import logging
import os
import textwrap
from dataclasses import dataclass, field

import webcolors
from mkt.databases.app.structures import StructureVisualizer

logger = logging.getLogger(__name__)


DICT_FILENAME_DEFAULTS = {
    "file_pdb": "structure.pdb",
    "file_script": "pymol_script.py",
    "file_txt": "instructions.txt",
}
"""dict: Default filenames for PDB, script, and instructions."""

DICT_COLOR_MAP = {
    "cyan": "#00FFFF",
    "magenta": "#FF00FF",
    "yellow": "#FFFF00",
    "red": "#FF0000",
    "green": "#008000",
    "blue": "#0000FF",
    "orange": "#FFA500",
    "purple": "#800080",
    "pink": "#FFC0CB",
    "brown": "#A52A2A",
    "gray": "#808080",
    "grey": "#808080",
    "darkred": "#8B0000",
    "darkgreen": "#006400",
    "darkblue": "#00008B",
    "darkorange": "#FF8C00",
    "darkviolet": "#9400D3",
    "white": "#FFFFFF",
    "black": "#000000",
    "lightblue": "#ADD8E6",
    "lightgreen": "#90EE90",
    "khaki": "#F0E68C",  # CSS3 standard khaki
    "cornflowerblue": "#6495ED",  # CSS3 standard cornflowerblue
}
"""dict: Fallback color name to hex mapping (CSS3 standard colors)."""


@dataclass
class PyMOLGenerator:
    """Generate PDB file with embedded color/style info and standalone PyMOL script."""

    viz: StructureVisualizer
    """StructureVisualizer object with loaded structure and config."""
    gene_name: str = field(init=False)
    """Gene name of the structure."""
    dict_filenames: dict = field(default_factory=lambda: dict)
    """Dictionary of filenames for PDB, script, and instructions."""

    def __post_init__(self):
        self.gene_name = self.viz.obj_kinase.hgnc_name

        # rename files with gene name and attribute
        str_attr = self.viz.config.str_attr
        self.dict_filenames = {
            k: f"{self.gene_name}_{str_attr.lower()}_{v}"
            for k, v in DICT_FILENAME_DEFAULTS.items()
        }

    def _convert_color_to_hex(self, color: str) -> str:
        """Convert named color to hex, with fallback options.

        Parameters
        ----------
        color : str
            Color name or hex string.

        Returns
        -------
        str
            Hex color string.
        """
        # if already hex, return as-is
        if color.startswith("#"):
            return color

        # try webcolors library first (uses CSS3 standard colors)
        try:
            return webcolors.name_to_hex(color)
        except ValueError:
            # fallback to custom mapping if webcolors doesn't recognize it
            return DICT_COLOR_MAP.get(color.lower(), "#808080")

    def _get_color_and_style_mapping(
        self,
    ) -> tuple[dict[int, str], list[int], dict[int, str]]:
        """Generate residue-to-color mapping, stick residue list, and label mapping.

        Uses the get_highlight_data() from StructureVisualizer which gets
        data from the config.

        Returns
        -------
        tuple[dict[int, str], list[int], dict[int, str]]
            Dictionary mapping residue numbers to hex colors,
            list of stick residue numbers,
            dictionary mapping residue numbers to label strings.
        """
        color_mapping = {}
        stick_residues = []
        label_mapping = {}

        # Get highlight data from visualizer (which gets it from config)
        # list_highlight is already 1-indexed
        list_highlight_align, dict_color_align, dict_style_align, dict_label_align = (
            self.viz.get_highlight_data()
        )

        # Access dict_align through the config's seq_align
        str_seq_cif = self.viz.config.seq_align.dict_align["KinCore, CIF"]["str_seq"]

        # create mapping: alignment_index (1-based) -> PDB_residue_number (1-based sequential)
        # The PDB is renumbered sequentially (1, 2, 3, ...) counting only non-gap CIF residues
        alignment_to_pdb = {}
        pdb_residue_count = 0
        for align_idx, cif_res in enumerate(str_seq_cif):
            if cif_res != "-":
                pdb_residue_count += 1
                # list_highlight from config is already 1-indexed
                alignment_to_pdb[align_idx + 1] = pdb_residue_count

        # convert colors to hex and identify stick residues
        for align_idx in list_highlight_align:
            if align_idx in alignment_to_pdb:
                pdb_res_num = alignment_to_pdb[align_idx]
                color = dict_color_align[align_idx]
                hex_color = self._convert_color_to_hex(color)
                color_mapping[pdb_res_num] = hex_color

                # Check if this should be a stick residue
                style = dict_style_align[align_idx]
                if style == "stick":
                    stick_residues.append(pdb_res_num)

                # Check for label
                label = dict_label_align.get(align_idx)
                if label is not None:
                    label_mapping[pdb_res_num] = label

        return color_mapping, stick_residues, label_mapping

    def return_filepath_dict(self, output_dir: str) -> dict[str, str]:
        """Return dictionary of filenames with paths.

        Parameters
        ----------
        output_dir : str
            Directory where files are saved.

        Returns
        -------
        dict[str, str]
            Dictionary mapping file types to full file paths.
        """
        return {k: os.path.join(output_dir, v) for k, v in self.dict_filenames.items()}

    def generate_annotated_pdb(self, output_path: str) -> str:
        """Generate PDB file with renumbered residues and color/style annotations.

        Parameters
        ----------
        output_path : str
            Path to save the annotated PDB file.

        Returns
        -------
        str
            Path to the saved annotated PDB file.
        """
        color_mapping, stick_residues, label_mapping = (
            self._get_color_and_style_mapping()
        )

        pdb_lines = self.viz.pdb_text.split("\n")

        # first, find the original residue numbers and create a mapping
        original_residues = []
        for line in pdb_lines:
            if line.startswith(("ATOM", "HETATM")) and len(line) > 22:
                try:
                    res_num = int(line[22:26].strip())
                    if res_num not in original_residues:
                        original_residues.append(res_num)
                except ValueError:
                    continue

        original_residues.sort()
        logger.debug(
            f"Found original residue range: {min(original_residues)} to {max(original_residues)}"
        )
        logger.debug(f"Total residues: {len(original_residues)}")

        # create mapping from original residue numbers to sequential (1, 2, 3...)
        old_to_new = {old_res: idx + 1 for idx, old_res in enumerate(original_residues)}

        # renumber the PDB content
        renumbered_lines = []
        for line in pdb_lines:
            if line.startswith(("ATOM", "HETATM")) and len(line) > 26:
                try:
                    old_res_num = int(line[22:26].strip())
                    new_res_num = old_to_new.get(old_res_num, old_res_num)
                    # Replace residue number in the line (columns 22-26)
                    new_line = line[:22] + f"{new_res_num:4d}" + line[26:]
                    renumbered_lines.append(new_line)
                except ValueError:
                    renumbered_lines.append(line)
            else:
                renumbered_lines.append(line)

        # prepare annotated lines with header
        str_attr = self.viz.config.str_attr
        annotated_lines = [
            "REMARK   1 GENERATED FOR PYMOL VISUALIZATION",
            f"REMARK   1 GENE: {self.gene_name}",
            f"REMARK   1 ATTRIBUTE: {str_attr}",
            f"REMARK   1 RESIDUES RENUMBERED: {min(original_residues)}-{max(original_residues)} -> 1-{len(original_residues)}",
            "REMARK   1 ",
            "REMARK   2 COLOR MAPPING (residue_number:hex_color):",
        ]

        # add color mapping as remarks (these should now be 1-based)
        for res_num, hex_color in color_mapping.items():
            annotated_lines.append(f"REMARK   2 {res_num}:{hex_color}")

        annotated_lines.extend(["REMARK   2 ", "REMARK   3 STICK RESIDUES:"])

        # add stick residues as remarks
        if stick_residues:
            stick_str = ",".join(map(str, stick_residues))
            annotated_lines.append(f"REMARK   3 {stick_str}")
        else:
            annotated_lines.append("REMARK   3 NONE")

        annotated_lines.extend(
            ["REMARK   3 ", "REMARK   4 ORIGINAL TO NEW RESIDUE MAPPING:"]
        )

        # add the mapping as remarks for reference
        for old_res, new_res in old_to_new.items():
            annotated_lines.append(f"REMARK   4 {old_res}->{new_res}")

        annotated_lines.append("REMARK   4 ")

        # add label mapping as remarks
        annotated_lines.append("REMARK   5 RESIDUE LABELS:")
        if label_mapping:
            for res_num, label_text in label_mapping.items():
                annotated_lines.append(f"REMARK   5 {res_num}:{label_text}")
        else:
            annotated_lines.append("REMARK   5 NONE")
        annotated_lines.append("REMARK   5 ")

        # add renumbered PDB content
        annotated_lines.extend(renumbered_lines)

        # write to file
        with open(output_path, "w") as f:
            f.write("\n".join(annotated_lines))

        logger.debug(f"Color mapping contains {len(color_mapping)} residues")
        logger.debug(f"Stick residues: {stick_residues}")

        return output_path

    def generate_pymol_script(self, pdb_path: str, output_path: str) -> str:
        """Generate PyMOL script that reads annotations from PDB and applies styling.

        Parameters
        ----------
        pdb_path : str
            Path to the annotated PDB file.
        output_path : str
            Path to save the PyMOL script.

        Returns
        -------
        str
            Path to the saved PyMOL script.
        """

        script_lines = [
            f"# PyMOL script for {self.gene_name} structure visualization",
            "from pymol import cmd",
            "import re",
            "",
            "def parse_pdb_remarks(pdb_file):",
            "    color_mapping = {}",
            "    stick_residues = []",
            "    label_mapping = {}",
            "    with open(pdb_file, 'r') as f:",
            "        for line in f:",
            "            if line.startswith('REMARK   2 ') and ':' in line:",
            "                match = re.search(r'(\\d+):(#[0-9A-Fa-f]{6})', line)",
            "                if match:",
            "                    res_num = int(match.group(1))",
            "                    hex_color = match.group(2)",
            "                    color_mapping[res_num] = hex_color",
            "            elif line.startswith('REMARK   3 ') and ',' in line:",
            "                residues_str = line.replace('REMARK   3 ', '').strip()",
            "                if residues_str and residues_str != 'NONE':",
            "                    try:",
            "                        stick_residues = [int(x.strip()) for x in residues_str.split(',')]",
            "                    except ValueError:",
            "                        pass",
            "            elif line.startswith('REMARK   5 ') and ':' in line:",
            "                content = line.replace('REMARK   5 ', '').strip()",
            "                if content and content != 'NONE' and content != 'RESIDUE LABELS:':",
            "                    # format: res_num:label_text",
            "                    parts = content.split(':', 1)",
            "                    if len(parts) == 2:",
            "                        try:",
            "                            res_num = int(parts[0])",
            "                            label_mapping[res_num] = parts[1]",
            "                        except ValueError:",
            "                            pass",
            "    return color_mapping, stick_residues, label_mapping",
            "",
            "# Load structure",
            f"cmd.load('{os.path.basename(pdb_path)}', '{self.gene_name}')",
            "",
            "# Parse color data from PDB remarks",
            f"color_mapping, stick_residues, label_mapping = parse_pdb_remarks('{os.path.basename(pdb_path)}')",
            "",
            "print(f'Found {len(color_mapping)} residues with colors')",
            "print(f'Found {len(stick_residues)} stick residues')",
            "print(f'Found {len(label_mapping)} residue labels')",
            "print('Color mapping (first 5 residues):', dict(list(color_mapping.items())[:5]))  # Show first 5",
            "print('Stick residues:', stick_residues)",
            "print('Labels:', label_mapping)",
            "",
            "# Set initial cartoon style with gray background",
            f"cmd.show_as('cartoon', '{self.gene_name}')",
            f"cmd.color('gray70', '{self.gene_name}')",
            f"cmd.set('cartoon_transparency', 0.5, '{self.gene_name}')",
            "",
            "# Apply custom colors",
            "color_counter = 0",
            "for res_num, hex_color in color_mapping.items():",
            "    color_name = f'custom_{color_counter}'",
            "    ",
            "    # Convert hex to RGB",
            "    hex_clean = hex_color.lstrip('#')",
            "    r = int(hex_clean[0:2], 16) / 255.0",
            "    g = int(hex_clean[2:4], 16) / 255.0",
            "    b = int(hex_clean[4:6], 16) / 255.0",
            "    ",
            "    # Define and apply color",
            "    cmd.set_color(color_name, [r, g, b])",
            "    cmd.color(color_name, f'resi {res_num}')",
            "    cmd.set('cartoon_transparency', 0.0, f'resi {res_num}')",
            "    ",
            "    color_counter += 1",
            "",
            "# Apply stick representation",
            "if stick_residues:",
            "    stick_selection = '+'.join(map(str, stick_residues))",
            "    cmd.show('sticks', f'resi {stick_selection}')",
            "    cmd.set('stick_radius', 0.3, f'resi {stick_selection}')",
            "    print(f'Applied sticks to: resi {stick_selection}')",
            "",
            "# Apply residue labels",
            "if label_mapping:",
            "    cmd.set('label_color', 'black')",
            "    cmd.set('label_size', 14)",
            "    for res_num, label_text in label_mapping.items():",
            "        cmd.label(f'resi {res_num} and name CA', f'\"{label_text}\"')",
            "    print(f'Applied labels to {len(label_mapping)} residues')",
            "",
            "# Final setup",
            "cmd.bg_color('white')",
            "",
            "print('Structure styling complete!')",
            "print('To save as EPS: set ray_trace_mode, 3; png filename.eps, ray=1')",
            "print('To save as PNG: set ray_trace_mode, 1; png filename.png, ray=1, dpi=300')",
        ]

        script_content = "\n".join(script_lines)

        # write script to file
        with open(output_path, "w") as f:
            f.write(script_content)

        return output_path

    def generate_instructions(self, output_dir: str) -> str:
        """Generate instructions for manual PyMOL execution.

        Parameters
        ----------
        output_dir : str
            Directory where files are saved

        Returns
        -------
        str
            Instructions text
        """
        dict_filepaths = self.return_filepath_dict(output_dir)

        instructions = f"""\
        Files generated:
        PDB: {dict_filepaths["file_pdb"]}
        Script: {dict_filepaths["file_script"]}
        Instructions: {dict_filepaths["file_txt"]}

        {"=" * 60}
        MANUAL PYMOL INSTRUCTIONS:
        1. Open PyMOL GUI or command line
        2. Change to the output directory
           cd {os.path.abspath(output_dir)}
        3. Run the script:
           run {os.path.basename(dict_filepaths["file_script"])}
        4. To save as high-res PNG:
           set ray_trace_mode, <mode>
           png <your_filename>.png, ray=1, dpi=300
        """
        instructions = textwrap.dedent(instructions)

        with open(dict_filepaths["file_txt"], "w") as f:
            f.write(instructions)

        return instructions

    def save_pymol_files(self, output_dir: str):
        """Generate PDB file and PyMOL script for manual PyMOL execution.

        Parameters
        ----------
        output_dir : str
            Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)

        dict_filepaths = self.return_filepath_dict(output_dir)

        path_pdb, path_script = (
            dict_filepaths["file_pdb"],
            dict_filepaths["file_script"],
        )

        self.generate_annotated_pdb(path_pdb)
        self.generate_pymol_script(path_pdb, path_script)
        str_out = self.generate_instructions(output_dir)

        logger.info(str_out)
