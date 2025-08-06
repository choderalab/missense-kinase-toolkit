from os import path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.glyphs import Rect, Text
from bokeh.plotting import figure
from pydantic.dataclasses import dataclass
from upsetplot import from_contents, plot


def generate_kinase_info_plot(
    dict_in: dict[str, Any],
    path_save: str,
) -> None:
    """Plot KinaseInfo upset plots from final harmonzied objects.

    Parameters
    ----------
    dict_in : dict[str, Any]
        Dictionary of KinaseInfo objects.
    path_save : str
        Path to save the plot.

    Returns
    -------
    None
        None
    """
    # generate data
    list_attr = ["uniprot", "pfam", "kinhub", "klifs", "kincore"]
    list_proper = ["UniProt", "Pfam", "KinHub", "KLIFS", "KinCore"]
    list_contents = [
        [k for k, v in dict_in.items() if getattr(v, attr) is not None]
        for attr in list_attr
    ]
    dict_contents = dict(zip(list_proper, list_contents))

    # generate figure
    contents = from_contents(dict_contents)
    fig = plt.figure(figsize=(12, 6))
    plot(contents, fig=fig, element_size=None)
    plt.savefig(path.join(path_save, "upset_plot.pdf"), bbox_inches="tight")


@dataclass
class SequenceAlignment:
    """Class for sequence alignment plot."""

    # required parameters
    list_sequences: list[str]
    """List of sequences to show in aligner."""
    list_ids: list[str]
    """List of sequence IDs."""
    dict_colors: dict[str, str]
    """Dictionary of colors for each sequence."""

    # optional formatting parameters
    font_size: int = 9
    """Font size for alignment."""
    plot_width: int = 800
    """Width of the plot."""
    plot_height: int = None
    """Height of the plot; default None and will full-page heuristic."""
    bool_top: bool = True
    """Show entire sequence view on top or not (no text, with zoom)."""
    bool_reverse: bool = True
    """Reverse the sequence or not."""
    bool_gridplot: bool = True
    """Use gridplot or not."""

    def __post_init__(self):
        # reverse input sequences and ids if bool_reverse is True
        if self.bool_reverse:
            self.list_sequences = self.list_sequences[::-1]
            self.list_ids = self.list_ids[::-1]

        # set up variables
        self.list_text = [i for s in self.list_sequences for i in s]
        self.colors = self.get_colors(self.list_text, self.dict_colors)
        self.N = len(self.list_sequences[0])  # number of residues
        self.S = len(self.list_sequences)  # number of sequences

        # bokeh plots
        self.source = self.create_coldatasource()
        self.plot_bottom = self.generate_bottom_plot()
        if self.bool_top:
            self.plot_top = self.generate_top_plot()
        else:
            self.plot_top = None
        if self.bool_gridplot:
            self.gridplot = self.generate_gridplot()
        else:
            self.gridplot = None

    @staticmethod
    def get_colors(
        list_str: str,
        dict_colors: dict[str, str],
    ) -> list[str]:
        """Get colors for residue in a given sequence.

        Parameters
        ----------
        list_str : str
            List of residues in a sequence.
        dict_colors : dict[str, str]
            Dictionary of colors for each residue.

        Returns
        -------
        list[str]
            List of colors for each residue.
        """
        list_colors = [dict_colors[i] for i in list_str]
        return list_colors

    def create_coldatasource(self):
        x = np.arange(1, self.N + 1)
        y = np.arange(0, self.S, 1)
        # creates a 2D grid of coords from the 1D arrays
        xx, yy = np.meshgrid(x, y)
        # flattens the arrays
        gx = xx.ravel()
        gy = yy.flatten()
        # use recty for rect coords with an offset
        recty = gy + 0.5
        # now we can create the ColumnDataSource with all the arrays
        source = ColumnDataSource(
            dict(
                x=gx,
                y=gy,
                recty=recty,
                text=self.list_text,
                colors=self.colors,
            )
        )
        return source

    def generate_top_plot(self) -> None:
        """Generate sequence alignment plot adapted from https://dmnfarrell.github.io/bioinformatics/bokeh-sequence-aligner."""

        x_range = Range1d(0, self.N + 1, bounds="auto")

        # entire sequence view (no text, with zoom)
        p_top = figure(
            title=None,
            frame_width=self.plot_width,
            frame_height=50,
            x_range=x_range,
            y_range=(0, self.S),
            tools="xpan, xwheel_zoom, reset, save",
            min_border=0,
            toolbar_location="below",
        )
        rects = Rect(
            x="x",
            y="recty",
            width=1,
            height=1,
            fill_color="colors",
            line_color=None,
            fill_alpha=0.6,
        )
        p_top.add_glyph(self.source, rects)
        p_top.yaxis.visible = False
        p_top.grid.visible = False

        return p_top

    def generate_bottom_plot(self) -> None:
        """Generate sequence alignment plot adapted from https://dmnfarrell.github.io/bioinformatics/bokeh-sequence-aligner."""

        if self.N > 100:
            viewlen = 100
        else:
            viewlen = self.N

        # sequence text view with ability to scroll along x axis
        # view_range is for the close up view
        view_range = (0, viewlen)
        if self.plot_height is None:
            self.plot_height = self.S * 15 + 50
        p_bottom = figure(
            title=None,
            frame_width=self.plot_width,
            frame_height=self.plot_height,
            x_range=view_range,
            y_range=self.list_ids,
            tools="xpan, xwheel_zoom, reset, save",
            min_border=0,
            toolbar_location="below",
            background_fill_color="white",
            border_fill_color="white",
        )
        glyph = Text(
            x="x",
            y="y",
            text="text",
            text_align="center",
            text_color="black",
            text_font_size=f"{str(self.font_size)}pt",
        )
        rects = Rect(
            x="x",
            y="recty",
            width=1,
            height=1,
            fill_color="colors",
            line_color=None,
            fill_alpha=0.4,
        )
        p_bottom.add_glyph(self.source, glyph)
        p_bottom.add_glyph(self.source, rects)
        p_bottom.grid.visible = False
        p_bottom.xaxis.major_label_text_font_style = "bold"
        p_bottom.yaxis.minor_tick_line_width = 0
        p_bottom.yaxis.major_tick_line_width = 0
        p_bottom.yaxis.axis_label_text_color = "black"
        p_bottom.yaxis.major_label_text_color = "black"
        p_bottom.xaxis.axis_label_text_color = "black"
        p_bottom.xaxis.major_label_text_color = "black"

        return p_bottom

    def generate_gridplot(self) -> None:
        """Generate sequence alignment gridplot."""
        if self.bool_top:
            return gridplot(
                [[self.plot_top], [self.plot_bottom]], toolbar_location="below"
            )
        else:
            return gridplot([[self.plot_bottom]], toolbar_location="below")

    def show_plot(self) -> None:
        """Show sequence alignment plot via Bokeh in separate window."""
        from bokeh.plotting import show

        show(self.gridplot)
