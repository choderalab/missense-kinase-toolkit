import os

try:
    import webcolors

    WEBCOLORS_AVAILABLE = True
except ImportError:
    WEBCOLORS_AVAILABLE = False


class SimplePyMOLGenerator:
    """Generate PDB file with embedded color/style info and standalone PyMOL script."""

    def __init__(self, structure_visualizer):
        """Initialize with existing StructureVisualizer object."""
        self.viz = structure_visualizer
        self.gene_name = self.viz.obj_kinase.hgnc_name

        # KLIFS stick residues (zero-indexed positions in KLIFS sequence)
        self.KLIFS_STICK_POSITIONS = [3, 5, 8, 68, 80]

    def _convert_color_to_hex(self, color: str) -> str:
        """Convert named color to hex, with fallback options."""
        # If already hex, return as-is
        if color.startswith("#"):
            return color

        # Try webcolors first
        if WEBCOLORS_AVAILABLE:
            try:
                return webcolors.name_to_hex(color)
            except ValueError:
                pass

        # Fallback color mapping
        color_map = {
            "cyan": "#00FFFF",
            "magenta": "#FF00FF",
            "yellow": "#FFFF00",
            "red": "#FF0000",
            "green": "#00FF00",
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
        }

        return color_map.get(color.lower(), "#808080")  # Default to gray

    def _get_color_and_style_mapping(self) -> tuple[dict[int, str], list[int]]:
        """Generate residue-to-color mapping and stick residue list."""
        color_mapping = {}
        stick_residues = []

        if not (self.viz.str_attr and self.viz.str_attr in self.viz.dict_align):
            return color_mapping, stick_residues

        # Get sequences and colors
        str_seq_cif = self.viz.dict_align["KinCore, CIF"]["str_seq"]
        str_seq_attr = self.viz.dict_align[self.viz.str_attr]["str_seq"]
        list_colors_attr = self.viz.dict_align[self.viz.str_attr]["list_colors"]

        # Map colors to residue positions
        cif_residue_count = 0
        for idx, (cif_res, attr_res) in enumerate(zip(str_seq_cif, str_seq_attr)):
            if cif_res != "-":
                cif_residue_count += 1
                if attr_res != "-":
                    color = list_colors_attr[idx]
                    hex_color = self._convert_color_to_hex(color)
                    color_mapping[cif_residue_count] = hex_color

                    # Check if this should be a stick residue (for KLIFS)
                    if self.viz.str_attr == "KLIFS":
                        # Find position in KLIFS sequence (excluding gaps)
                        klifs_pos = 0
                        for i in range(idx + 1):
                            if (
                                self.viz.dict_align[self.viz.str_attr]["str_seq"][i]
                                != "-"
                            ):
                                klifs_pos += 1
                        klifs_pos -= 1  # Convert to zero-indexed

                        if klifs_pos in self.KLIFS_STICK_POSITIONS:
                            stick_residues.append(cif_residue_count)

        return color_mapping, stick_residues

    def generate_annotated_pdb(self, output_path: str) -> str:
        """Generate PDB file with renumbered residues and color/style annotations."""
        color_mapping, stick_residues = self._get_color_and_style_mapping()

        pdb_lines = self.viz.pdb_text.split("\n")

        # First, find the original residue numbers and create a mapping
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
        print(
            f"Debug: Found original residue range: {min(original_residues)} to {max(original_residues)}"
        )
        print(f"Debug: Total residues: {len(original_residues)}")

        # Create mapping from original residue numbers to sequential (1, 2, 3...)
        old_to_new = {old_res: idx + 1 for idx, old_res in enumerate(original_residues)}

        # Renumber the PDB content
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

        # Prepare annotated lines with header
        annotated_lines = [
            "REMARK   1 GENERATED FOR PYMOL VISUALIZATION",
            f"REMARK   1 GENE: {self.gene_name}",
            f"REMARK   1 ATTRIBUTE: {self.viz.str_attr or 'None'}",
            f"REMARK   1 RESIDUES RENUMBERED: {min(original_residues)}-{max(original_residues)} -> 1-{len(original_residues)}",
            "REMARK   1 ",
            "REMARK   2 COLOR MAPPING (residue_number:hex_color):",
        ]

        # Add color mapping as remarks (these should now be 1-based)
        for res_num, hex_color in color_mapping.items():
            annotated_lines.append(f"REMARK   2 {res_num}:{hex_color}")

        annotated_lines.extend(["REMARK   2 ", "REMARK   3 STICK RESIDUES:"])

        # Add stick residues as remarks
        if stick_residues:
            stick_str = ",".join(map(str, stick_residues))
            annotated_lines.append(f"REMARK   3 {stick_str}")
        else:
            annotated_lines.append("REMARK   3 NONE")

        annotated_lines.extend(
            ["REMARK   3 ", "REMARK   4 ORIGINAL TO NEW RESIDUE MAPPING:"]
        )

        # Add the mapping as remarks for reference
        for old_res, new_res in old_to_new.items():
            annotated_lines.append(f"REMARK   4 {old_res}->{new_res}")

        annotated_lines.append("REMARK   4 ")

        # Add renumbered PDB content
        annotated_lines.extend(renumbered_lines)

        # Write to file
        with open(output_path, "w") as f:
            f.write("\n".join(annotated_lines))

        print(f"Debug: Color mapping contains {len(color_mapping)} residues")
        print(f"Debug: Stick residues: {stick_residues}")

        return output_path

    def generate_pymol_script(self, pdb_path: str, output_path: str) -> str:
        """Generate PyMOL script that reads annotations from PDB and applies styling."""

        script_lines = [
            f"# PyMOL script for {self.gene_name} structure visualization",
            "from pymol import cmd",
            "import re",
            "",
            "def parse_pdb_remarks(pdb_file):",
            "    color_mapping = {}",
            "    stick_residues = []",
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
            "    return color_mapping, stick_residues",
            "",
            "# Load structure",
            f"cmd.load('{os.path.basename(pdb_path)}', '{self.gene_name}')",
            "",
            "# Parse color data from PDB remarks",
            f"color_mapping, stick_residues = parse_pdb_remarks('{os.path.basename(pdb_path)}')",
            "",
            "print(f'Found {len(color_mapping)} residues with colors')",
            "print(f'Found {len(stick_residues)} stick residues')",
            "print('Color mapping:', dict(list(color_mapping.items())[:5]))  # Show first 5",
            "print('Stick residues:', stick_residues)",
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
            "# Final setup",
            "cmd.bg_color('white')",
            "",
            "print('Structure styling complete!')",
            "print('To save as EPS: set ray_trace_mode, 3; png filename.eps, ray=1')",
            "print('To save as PNG: set ray_trace_mode, 1; png filename.png, ray=1, dpi=300')",
        ]

        script_content = "\n".join(script_lines)

        # Write script to file
        with open(output_path, "w") as f:
            f.write(script_content)

        return output_path


def generate_simple_pymol_files(viz_obj, output_dir: str, base_name: str = None):
    """
    Generate PDB file and PyMOL script for manual PyMOL execution.

    Parameters
    ----------
    viz_obj : StructureVisualizer
        Existing StructureVisualizer object
    output_dir : str
        Directory to save files
    base_name : str, optional
        Base name for files (default: gene name)

    Returns
    -------
    tuple
        Paths to (pdb_file, pymol_script)
    """
    if base_name is None:
        base_name = viz_obj.obj_kinase.hgnc_name

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate files
    generator = SimplePyMOLGenerator(viz_obj)

    pdb_path = os.path.join(output_dir, f"{base_name}_structure.pdb")
    script_path = os.path.join(output_dir, f"{base_name}_pymol_script.py")

    generator.generate_annotated_pdb(pdb_path)
    generator.generate_pymol_script(pdb_path, script_path)

    print("Files generated:")
    print(f"  PDB: {pdb_path}")
    print(f"  Script: {script_path}")
    print("")
    print("To run in PyMOL:")
    print("  1. Open PyMOL")
    print(f"  2. Navigate to {output_dir}")
    print(f"  3. Run: run {os.path.basename(script_path)}")
    print("  4. Save PNG: set ray_trace_mode, 3; png your_filename.png, ray=1, dpi=300")

    return pdb_path, script_path
