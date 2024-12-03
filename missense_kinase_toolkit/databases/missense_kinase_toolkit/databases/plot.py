import numpy as np
from bokeh.layouts import gridplot

# from bokeh.models import ColumnDataSource, Plot, Grid, Range1d
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.glyphs import Rect, Text
from bokeh.plotting import figure
from pydantic.dataclasses import dataclass


@dataclass
class SequenceAlignment:
    list_sequences: list[str]
    """List of sequences to show in aligner."""
    list_ids: list[str]
    """List of sequence IDs."""
    dict_colors: dict[str, str]
    """Dictionary of colors for each sequence."""
    font_size: int = 9
    """Font size for alignment."""
    plot_width: int = 800
    """Width of the plot."""

    def __post_init__(self):
        self.generate_alignment()

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

    def generate_alignment(self) -> None:
        """Generate sequence alignment plot adapted from https://dmnfarrell.github.io/bioinformatics/bokeh-sequence-aligner."""

        # reverse text and colors so A-Z is top-bottom not bottom-top
        list_text = [i for s in self.list_sequences[::-1] for i in s]
        colors = self.get_colors(list_text, self.dict_colors)

        N = len(self.list_sequences[0])
        S = len(self.list_sequences)

        x = np.arange(1, N + 1)
        y = np.arange(0, S, 1)
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
                text=list_text,
                colors=colors,
            )
        )
        x_range = Range1d(0, N + 1, bounds="auto")
        if N > 100:
            viewlen = 100
        else:
            viewlen = N

        # entire sequence view (no text, with zoom)
        p = figure(
            title=None,
            frame_width=self.plot_width,
            frame_height=50,
            x_range=x_range,
            y_range=(0, S),
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
        p.add_glyph(source, rects)
        p.yaxis.visible = False
        p.grid.visible = False

        # sequence text view with ability to scroll along x axis
        # view_range is for the close up view
        view_range = (0, viewlen)
        plot_height = S * 15 + 50
        p1 = figure(
            title=None,
            frame_width=self.plot_width,
            frame_height=plot_height,
            x_range=view_range,
            y_range=self.list_ids[::-1],
            tools="xpan,reset",
            min_border=0,
            toolbar_location="below",
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
        p1.add_glyph(source, glyph)
        p1.add_glyph(source, rects)
        p1.grid.visible = False
        p1.xaxis.major_label_text_font_style = "bold"
        p1.yaxis.minor_tick_line_width = 0
        p1.yaxis.major_tick_line_width = 0

        self.plot = gridplot([[p], [p1]], toolbar_location="below")

    def show_plot(self) -> None:
        """Show sequence alignment plot via Bokeh."""
        from bokeh.plotting import show

        # show in separate window
        show(self.plot)

        # notebook alternative
        # import panel as pn
        # pn.extension()
        # pn.pane.Bokeh(alignment_klifs_min.plot)
