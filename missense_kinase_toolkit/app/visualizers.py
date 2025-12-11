import logging
from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import py3Dmol
from bokeh.models import (
    ColumnDataSource,
    CustomJSTickFormatter,
    FixedTicker,
    Label,
)
from bokeh.models.glyphs import Rect, Text
from bokeh.plotting import figure
from mkt.databases.app.sequences import SequenceAlignment
from mkt.databases.app.structures import StructureVisualizer

logger = logging.getLogger(__name__)


@dataclass
class SequenceAlignmentVisualizer(SequenceAlignment):
    """Class to generate sequence alignment plots using Bokeh."""

    font_size: int = 9
    """Font size for alignment."""
    plot_width: int = 1200
    """Width of the plot."""

    def generate_plot(self) -> None:
        """Generate sequence alignment plot adapted from https://dmnfarrell.github.io/bioinformatics/bokeh-sequence-aligner."""

        list_text = [i for s in self.list_sequences for i in s]
        colors = list(chain(*self.list_colors))

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

        if N > 80:
            viewlen = 80
        else:
            viewlen = N

        # Determine which sequences consist of only '-' characters
        empty_sequences = [all(c == "-" for c in seq) for seq in self.list_sequences]

        # Create a dictionary to map y-axis labels to their colors
        y_label_colors = {}
        for i, (id_label, is_empty) in enumerate(zip(self.list_ids, empty_sequences)):
            y_label_colors[i] = "#DC143C" if is_empty else "black"

        # sequence text view with ability to scroll along x axis
        # view_range is for the close up view
        view_range = (-8, viewlen)
        plot_height = 100
        p1 = figure(
            title=None,
            frame_width=self.plot_width,
            frame_height=plot_height,
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
            text_baseline="bottom",
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

        p1.xaxis.ticker = FixedTicker(ticks=list(range(1, N + 1)))

        # Add KLIFS pocket labels as custom labels
        # Create KLIFS position mapping for the formatter
        klifs_mapping = {}
        if (
            hasattr(self.obj_kinase, "KLIFS2UniProtIdx")
            and self.obj_kinase.KLIFS2UniProtIdx
        ):
            klifs_mapping = {
                pos: label
                for label, pos in self.obj_kinase.KLIFS2UniProtIdx.items()
                if pos is not None
            }

        # Custom formatter that includes KLIFS labels
        formatter_code = f"""
        var klifs_map = {klifs_mapping};
        var base_label = String(tick);
        if (klifs_map[tick]) {{
            return klifs_map[tick] + "\t\t" + base_label;
        }}
        return base_label;
        """

        p1.xaxis.formatter = CustomJSTickFormatter(code=formatter_code)

        p1.xaxis.major_label_orientation = np.pi / 2  # Rotate labels 90 degrees
        p1.xaxis.major_label_standoff = 2  # Add some space between axis and labels
        p1.xaxis.axis_label = "Residue Position"

        # Remove default y-axis labels
        p1.yaxis.major_label_text_font_size = "0pt"  # Hide the default labels

        # Add custom colored labels
        for i, label in enumerate(self.list_ids):
            color = "#DC143C" if empty_sequences[i] else "black"
            custom_label = Label(
                x=0,  # Position at the y-axis
                y=i,  # The y position corresponds to the sequence index
                text=label,
                text_color=color,
                text_font_size="9pt",
                text_font_style="bold",
                text_align="right",
                x_offset=-10,  # Offset to position it near the axis
            )
            p1.add_layout(custom_label)

        p1.yaxis.minor_tick_line_width = 0
        p1.yaxis.major_tick_line_width = 0
        p1.yaxis.axis_label_text_color = "black"
        p1.yaxis.major_label_text_color = "black"
        p1.xaxis.axis_label_text_color = "black"
        p1.xaxis.major_label_text_color = "black"

        return p1


@dataclass
class StructureVisualizerVisualizer(StructureVisualizer):
    """Class to generate structure visualizations for kinase structures."""

    bool_show: bool = False
    """Whether to show the structure in the viewer or return HTML."""
    dict_dims: dict[str, int] = field(
        default_factory=lambda: {"width": 600, "height": 600}
    )
    """Dimensions for the py3Dmol viewer."""

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
