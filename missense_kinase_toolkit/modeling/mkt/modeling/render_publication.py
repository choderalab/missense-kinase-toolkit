#!/usr/bin/env python
"""
Module for rendering publication-quality images in PyMOL.
"""

import os
import sys

from pymol import cmd


class PyMolImageGenerator:
    """Class for generating publication-quality images in PyMOL."""

    def __init__(
        self,
        output_filename: str,
        color: str = None,
        str_idx: str = None,
        highlight_color: str = None,
        bg_color: str = "grey70",
        bg_transparency: float = 0.7,
    ):
        """
        Initializes the PyMolImageGenerator with rendering parameters.

        Parameters:
        -----------
        output_filename: str
            Output PNG filename
        color: str, optional
            Optional color name for cartoon ribbons (legacy behavior - colors entire structure)
        str_idx: str, optional
            Optional residue range to highlight in format "start-end" (e.g., "245-315")
        highlight_color: str, optional
            Color for highlighted region (used when str_idx is provided)
        bg_color: str, optional
            Background color for non-highlighted regions (default: 'grey70')
        bg_transparency: float, optional
            Transparency for non-highlighted regions, 0.0-1.0 (default: 0.7)
        """
        self.output_filename = output_filename
        self.color = color
        self.str_idx = str_idx
        self.highlight_color = highlight_color
        self.bg_color = bg_color
        self.bg_transparency = bg_transparency

    def run(self):
        """Executes the image generation process."""
        self.generate_pymol_figure()
        self.render_publication_quality_image(self.output_filename)

    @staticmethod
    def render_publication_quality_image(output_filename: str):
        """
        Render a publication-quality image using PyMOL's ray tracing.

        Parameters:
        -----------
        output_filename: str
            Output PNG filename
        """
        # publication-quality ray trace settings
        cmd.set("ray_trace_mode", 1)
        cmd.set("ray_trace_gain", 0.1)
        cmd.set("cartoon_sampling", 14)
        cmd.set("antialias", 2)
        cmd.set("ray_shadows", 1)
        cmd.set("depth_cue", 0)

        # render and save
        cmd.png(output_filename, dpi=300, ray=1)

        # get and print absolute path
        abs_path = os.path.abspath(output_filename)
        print(f"Rendered publication-quality image to: {abs_path}")

    def generate_pymol_figure(self):
        """Render current PyMOL scene with publication-quality settings.

        Notes:
        ------
        Usage in PyMOL:
            run render_publication.py
            generate_pymol_figure output.png
            generate_pymol_figure output.png, marine
            generate_pymol_figure output.png, str_idx='245-315', highlight_color='red'
            generate_pymol_figure output.png, str_idx='245-315', highlight_color='red', bg_color='grey80', bg_transparency=0.5
        Usage from terminal:
            pymol pse/abl1_uniprot.pse -d "run render_publication.py; generate_pymol_figure images/abl1_uniprot.png; quit"
        """
        # region highlighting mode
        if self.str_idx:
            # parse as user specified: "start-end" -> idx_start, idx_end
            idx_start = int(self.str_idx.split("-")[0])
            idx_end = int(self.str_idx.split("-")[1])
            # create PyMOL selection string for residue range
            region_selection = f"resi {idx_start}-{idx_end}"

            try:
                # step 1: make entire structure light grey and transparent
                cmd.color(self.bg_color, "all")
                cmd.set("cartoon_transparency", self.bg_transparency, "all")
                print(
                    f"Applied background color '{self.bg_color}' with transparency {self.bg_transparency}"
                )

                # step 2: highlight specified region
                final_color = (
                    self.highlight_color
                    if self.highlight_color
                    else (self.color if self.color else "red")
                )
                cmd.color(final_color, region_selection)
                cmd.set("cartoon_transparency", 0.0, region_selection)  # opaque
                print(
                    f"Highlighted region {idx_start}-{idx_end} with color '{final_color}'"
                )
            except Exception as e:
                print(f"Warning: Could not apply region highlighting: {e}")
                print("Using default coloring")
        else:
            # apply color if specified
            if self.color:
                try:
                    cmd.color(self.color, "all")
                    print(f"Applied color: {self.color}")
                except Exception as e:
                    print(f"Warning: Could not apply color '{self.color}': {e}")
                    print("Using default coloring")


# wrapper function for PyMOL command registration
def generate_pymol_figure(
    output_filename: str,
    color: str = None,
    str_idx: str = None,
    highlight_color: str = None,
    bg_color: str = "grey70",
    bg_transparency: float = 0.7,
):
    """
    Wrapper function to make PyMolImageGenerator accessible as a PyMOL command.

    Parameters:
    -----------
    output_filename: str
        Output PNG filename
    color: str, optional
        Optional color name for cartoon ribbons (legacy behavior - colors entire structure)
    str_idx: str, optional
        Optional residue range to highlight in format "start-end" (e.g., "245-315")
    highlight_color: str, optional
        Color for highlighted region (used when str_idx is provided)
    bg_color: str, optional
        Background color for non-highlighted regions (default: 'grey70')
    bg_transparency: float, optional
        Transparency for non-highlighted regions, 0.0-1.0 (default: 0.7)
    """
    pymol_generator = PyMolImageGenerator(
        output_filename=output_filename,
        color=color,
        str_idx=str_idx,
        highlight_color=highlight_color,
        bg_color=bg_color,
        bg_transparency=bg_transparency,
    )
    pymol_generator.run()


# register as PyMOL command
cmd.extend("generate_pymol_figure", generate_pymol_figure)

# allow command-line usage
if __name__ == "__main__":
    if len(sys.argv) > 1:
        output = sys.argv[1]
        color = sys.argv[2] if len(sys.argv) > 2 else None
        generate_pymol_figure(output_filename=output, color=color)
        sys.exit(0)
