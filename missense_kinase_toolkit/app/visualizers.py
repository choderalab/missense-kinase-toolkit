import logging
from dataclasses import dataclass
from itertools import chain
from typing import Any

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
class SequenceAlignmentGenerator(SequenceAlignment):
    """Class to generate sequence alignment plots using Bokeh."""

    font_size: int = 9
    """Font size for alignment."""
    plot_width: int = 1200
    """Width of the plot."""

    def __post_init__(self):
        super().__post_init__()
        self.plot = self.generate_plot()

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

        # determine which sequences consist of only '-' characters
        empty_sequences = [all(c == "-" for c in seq) for seq in self.list_sequences]

        # create a dictionary to map y-axis labels to their colors
        y_label_colors = {}
        for i, (_, is_empty) in enumerate(zip(self.list_ids, empty_sequences)):
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

        # add KLIFS pocket labels as custom labels
        # create KLIFS position mapping for the formatter
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

        # custom formatter that includes KLIFS labels
        formatter_code = f"""
        var klifs_map = {klifs_mapping};
        var base_label = String(tick);
        if (klifs_map[tick]) {{
            return klifs_map[tick] + "\t\t" + base_label;
        }}
        return base_label;
        """

        p1.xaxis.formatter = CustomJSTickFormatter(code=formatter_code)

        # rotate labels 90 degrees
        p1.xaxis.major_label_orientation = np.pi / 2
        # add some space between axis and labels
        p1.xaxis.major_label_standoff = 2
        p1.xaxis.axis_label = "Residue Position"

        # hide the default y-axis labels
        p1.yaxis.major_label_text_font_size = "0pt"

        # add custom colored labels
        for i, label in enumerate(self.list_ids):
            color = "#DC143C" if empty_sequences[i] else "black"
            custom_label = Label(
                x=0,  # position at the y-axis
                y=i,  # y position corresponds to the sequence index
                text=label,
                text_color=color,
                text_font_size="9pt",
                text_font_style="bold",
                text_align="right",
                x_offset=-10,  # offset to position it near the axis
            )
            p1.add_layout(custom_label)

        p1.yaxis.minor_tick_line_width = 0
        p1.yaxis.major_tick_line_width = 0
        p1.yaxis.axis_label_text_color = "black"
        p1.yaxis.major_label_text_color = "black"
        p1.xaxis.axis_label_text_color = "black"
        p1.xaxis.major_label_text_color = "black"

        return p1


DICT_VIZ_OPACITY = {
    "None": 1.0,
    "KLIFS": 1.0,
    "Phosphosites": 1.0,
    "Mutations": 0.9,
    "lowlight": 0.5,
}
"""dict[str, float]: Opacity for the py3Dmol viewer."""

DICT_VIZ_STYLE = {
    "None": "cartoon",
    "lowlight": "cartoon",
}
"""dict[str, str]: Default styles for py3Dmol viewer."""

DICT_VIZ_COLOR = {
    "None": "spectrum",
    "lowlight": "gray",
}
"""dict[str, str]: Default colors for py3Dmol viewer."""


class StructureVisualizerGenerator:
    """Generate py3Dmol structure visualizations for kinase structures.

    Uses composition with StructureVisualizer to access structure data and
    highlight information from the config.

    Parameters
    ----------
    viz : StructureVisualizer
        StructureVisualizer instance with loaded structure and config.
    dict_opacity : dict[str, float], optional
        Opacity settings for different highlight types.
    bool_show : bool, optional
        Whether to show the structure in the viewer or return HTML.
    dict_dims : dict[str, int], optional
        Dimensions for the py3Dmol viewer.

    Attributes
    ----------
    viz : StructureVisualizer
        The underlying structure visualizer.
    html : str | None
        HTML representation of the py3Dmol viewer.
    """

    def __init__(
        self,
        viz: StructureVisualizer,
        dict_opacity: dict[str, float] | None = None,
        bool_show: bool = False,
        dict_dims: dict[str, int] | None = None,
    ):
        self.viz = viz
        self.dict_opacity = dict_opacity or DICT_VIZ_OPACITY.copy()
        self.bool_show = bool_show
        self.dict_dims = dict_dims or {"width": 600, "height": 600}

        # Generate visualization on init
        self.html = self.visualize_structure()

    @property
    def str_attr(self) -> str:
        """Get the attribute being highlighted from the config."""
        return self.viz.config.str_attr

    def _return_style_dict(
        self,
        str_key: str,
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
            str_color = DICT_VIZ_COLOR.get(str_key, "gray")
        if str_style is None:
            str_style = DICT_VIZ_STYLE.get(str_key, "cartoon")
        if float_opacity is None:
            float_opacity = self.dict_opacity.get(str_key, 1.0)

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
        str | None
            HTML representation of the py3Dmol viewer or None if bool_show=True.
        """
        view = py3Dmol.view(**self.dict_dims)

        view.addModel(self.viz.pdb_text, "pdb")

        # Get highlight data from the visualizer (which gets it from config)
        list_highlight, dict_color, dict_style, dict_label = (
            self.viz.get_highlight_data()
        )

        # Get opacity for this attribute type
        attr_opacity = self.dict_opacity.get(self.str_attr, 1.0)

        for residue in self.viz.residues:
            res_no = residue.get_id()[1]
            # set lowlight background
            view.setStyle(
                {"resi": str(res_no)},
                self._return_style_dict("lowlight"),
            )
            # add highlights for the selected attribute
            if res_no in list_highlight:
                view.addStyle(
                    {"resi": str(res_no)},
                    self._return_style_dict(
                        str_key=self.str_attr,
                        str_color=dict_color[res_no],
                        str_style=dict_style[res_no],
                        float_opacity=attr_opacity,
                    ),
                )
                # add label if present
                label_text = dict_label.get(res_no)
                if label_text is not None:
                    # addLabel signature: (text, options, sel)
                    # options = label styling, sel = atom selector for positioning
                    view.addLabel(
                        label_text,
                        {
                            "backgroundColor": "white",
                            "fontColor": "black",
                            "fontSize": 10,
                            "showBackground": True,
                            "backgroundOpacity": 0.7,
                        },
                        {"resi": str(res_no), "atom": "CA"},
                    )

        view.zoomTo()
        if self.bool_show:
            view.show()
            return None
        else:
            return view._make_html()
