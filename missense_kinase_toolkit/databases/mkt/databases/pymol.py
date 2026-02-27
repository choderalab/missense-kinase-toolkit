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
    str_attr: str
    """Granular attribute name of the structure visualization (e.g., KLIFS_IMPORTANT not just KLIFS)."""
    gene_name: str = field(init=False)
    """Gene name of the structure."""
    dict_filenames: dict = field(default_factory=lambda: dict)
    """Dictionary of filenames for PDB, script, and instructions."""

    def __post_init__(self):
        self.gene_name = self.viz.obj_kinase.hgnc_name

        # rename files with gene name and attribute
        # str_attr = self.viz.config.str_attr
        self.dict_filenames = {
            k: f"{self.gene_name}_{self.str_attr.lower()}_{v}"
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
        # str_attr = self.viz.config.str_attr
        annotated_lines = [
            "REMARK   1 GENERATED FOR PYMOL VISUALIZATION",
            f"REMARK   1 GENE: {self.gene_name}",
            f"REMARK   1 ATTRIBUTE: {self.str_attr}",
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

        # derive object name from PDB filename stem (e.g., "ABL1_group_structure")
        pdb_basename = os.path.basename(pdb_path)
        obj_name = os.path.splitext(pdb_basename)[0]

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
            f"cmd.load('{pdb_basename}', '{obj_name}')",
            "",
            "# Parse color data from PDB remarks",
            f"color_mapping, stick_residues, label_mapping = parse_pdb_remarks('{pdb_basename}')",
            "",
            "print(f'Found {len(color_mapping)} residues with colors')",
            "print(f'Found {len(stick_residues)} stick residues')",
            "print(f'Found {len(label_mapping)} residue labels')",
            "print('Color mapping (first 5 residues):', dict(list(color_mapping.items())[:5]))  # Show first 5",
            "print('Stick residues:', stick_residues)",
            "print('Labels:', label_mapping)",
            "",
            "# Set initial cartoon style with light gray background",
            f"cmd.show_as('cartoon', '{obj_name}')",
            "cmd.set_color('lightgray', [0.827, 0.827, 0.827])",  # #D3D3D3
            f"cmd.color('lightgray', '{obj_name}')",
            f"cmd.set('cartoon_transparency', 0.5, '{obj_name}')",
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
            f"    cmd.color(color_name, f'{obj_name} and resi {{res_num}}')",
            f"    cmd.set('cartoon_transparency', 0.0, f'{obj_name} and resi {{res_num}}')",
            "    ",
            "    color_counter += 1",
            "",
            "# Apply stick representation",
            "if stick_residues:",
            "    stick_selection = '+'.join(map(str, stick_residues))",
            f"    cmd.show('sticks', f'{obj_name} and resi {{stick_selection}}')",
            f"    cmd.set('stick_radius', 0.3, f'{obj_name} and resi {{stick_selection}}')",
            "    print(f'Applied sticks to: resi {stick_selection}')",
            "",
            "# Disable fog/depth cueing so back-plane labels render crisp",
            "cmd.set('depth_cue', 0)",
            "cmd.set('fog', 0)",
            "cmd.set('ray_trace_fog', 0)",
            "",
            "# Apply residue labels using pseudoatoms offset from CA with connector lines",
            "if label_mapping:",
            "    import numpy as np",
            "    label_offset = 20.0  # angstroms offset from CA",
            "    min_label_dist = 6.0  # minimum distance between labels (angstroms)",
            "    cmd.set('label_color', 'black')",
            "    cmd.set('label_size', 14)",
            "    cmd.set('label_font_id', 7)",  # bold font
            "    cmd.set('label_connector', 1)",
            "    cmd.set('label_connector_color', 'black')",
            "    cmd.set('label_connector_width', 1.5)",
            "",
            "    # pass 1: compute initial label positions",
            f"    com = np.array(cmd.centerofmass('{obj_name}'))",
            "    label_data = {}  # res_num -> (ca_pos, offset_pos, label_text)",
            "    for res_num, label_text in label_mapping.items():",
            f"        ca_sel = f'{obj_name} and resi {{res_num}} and name CA'",
            "        coords = cmd.get_coords(ca_sel)",
            "        if coords is not None and len(coords) > 0:",
            "            ca_pos = np.array(coords[0])",
            "            direction = ca_pos - com",
            "            norm = np.linalg.norm(direction)",
            "            if norm > 0:",
            "                direction = direction / norm",
            "            else:",
            "                direction = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)",
            "            offset_pos = ca_pos + direction * label_offset",
            "            label_data[res_num] = (ca_pos, offset_pos, label_text)",
            "",
            "    # pass 2: resolve label collisions by iterative repulsion",
            "    res_nums = list(label_data.keys())",
            "    positions = {rn: label_data[rn][1].copy() for rn in res_nums}",
            "    for _ in range(50):  # iterate to convergence",
            "        moved = False",
            "        for i, rn_i in enumerate(res_nums):",
            "            for rn_j in res_nums[i+1:]:",
            "                diff = positions[rn_i] - positions[rn_j]",
            "                dist = np.linalg.norm(diff)",
            "                if dist < min_label_dist and dist > 0:",
            "                    # push apart along their difference vector",
            "                    push = (min_label_dist - dist) / 2.0 * (diff / dist)",
            "                    positions[rn_i] += push",
            "                    positions[rn_j] -= push",
            "                    moved = True",
            "        if not moved:",
            "            break",
            "",
            "    # pass 3: place labels and connector lines",
            "    for res_num in res_nums:",
            "        ca_pos, _, label_text = label_data[res_num]",
            "        offset_pos = positions[res_num]",
            f"        ca_sel = f'{obj_name} and resi {{res_num}} and name CA'",
            "        # create pseudoatom at offset position for label",
            "        pseudo_name = f'label_pt_{res_num}'",
            "        cmd.pseudoatom(pseudo_name, pos=offset_pos.tolist())",
            "        cmd.label(pseudo_name, f'\"{label_text}\"')",
            "        # create a second pseudoatom for the line endpoint, stopping short of label",
            "        direction_to_label = offset_pos - ca_pos",
            "        label_dist = np.linalg.norm(direction_to_label)",
            "        if label_dist > 3.0:",
            "            line_end_pos = ca_pos + direction_to_label * ((label_dist - 3.0) / label_dist)",
            "        else:",
            "            line_end_pos = ca_pos",
            "        line_end_name = f'label_end_{res_num}'",
            "        cmd.pseudoatom(line_end_name, pos=line_end_pos.tolist())",
            "        # draw connector line from CA to shortened endpoint",
            "        line_name = f'label_line_{res_num}'",
            "        cmd.distance(line_name, ca_sel, line_end_name)",
            "        cmd.hide('labels', line_name)  # hide distance measurement",
            "        cmd.set('dash_gap', 0.0, line_name)  # solid line",
            "        cmd.set('dash_color', 'black', line_name)",
            "        cmd.set('dash_width', 1.5, line_name)",
            "        cmd.hide('nonbonded', pseudo_name)  # hide pseudoatom marker",
            "        cmd.hide('nonbonded', line_end_name)  # hide line endpoint marker",
            "        # group label objects for easy toggling",
            "        cmd.group('labels', f'{pseudo_name} {line_end_name} {line_name}')",
            "    print(f'Applied labels to {len(label_mapping)} residues')",
            "",
            "# Final setup",
            "cmd.bg_color('white')",
            "",
            "# Define save_image function for publication-quality rendering",
            f"def save_image(output_filename='{self.gene_name}_{self.str_attr.lower()}_structure.png'):",
            '    """Render and save a publication-quality PNG image."""',
            "    cmd.set('ray_trace_mode', 0)",  # standard ray tracing (no black outlines)
            "    cmd.set('ray_trace_gain', 0.0)",  # no edge darkening
            "    cmd.set('ray_shadows', 0)",  # no shadows to preserve colormap fidelity
            "    cmd.set('specular', 0)",  # no specular highlights
            "    cmd.set('ambient', 0.6)",  # higher ambient light to reduce directional shading
            "    cmd.set('direct', 0.4)",  # lower direct light to flatten shading
            "    cmd.set('cartoon_sampling', 14)",
            "    cmd.set('antialias', 2)",
            "    cmd.png(output_filename, dpi=300, ray=1)",
            "    import os",
            "    print(f'Rendered publication-quality image to: {os.path.abspath(output_filename)}')",
            "",
            "cmd.extend('save_image', save_image)",
            "",
            "print('Structure styling complete!')",
            "print('Adjust the view as needed, then run: save_image()')",
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

        png_filename = f"{self.gene_name}_{self.str_attr.lower()}_structure.png"
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
        4. Adjust the view so all labels are visible
        5. Save as high-res PNG:
           save_image()
           (saves to {png_filename} by default, or pass a custom filename: save_image("custom.png"))
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
