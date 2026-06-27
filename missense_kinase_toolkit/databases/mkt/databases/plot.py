"""Plotting functions for kinase data.

Includes dynamic-range, ridgeline, stacked-bar, Venn-diagram, metrics-boxplot, and
sequence-input schematic plots, plus :class:`SequenceAlignment` rendering and
Matplotlib RC configuration helpers.
"""

import logging
import os
from os import path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mkt.databases.plot_config import (
    ColKinaseColorConfig,
    DynamicRangePlotConfig,
    FamilyColorConfig,
    MatplotlibRCConfig,
    MetricsBoxplotConfig,
    RegionGapViolinConfig,
    RidgelinePlotConfig,
    SequenceSchematicConfig,
    StackedBarchartConfig,
    UpsetPlotConfig,
    VennDiagramConfig,
)
from mkt.schema.io_utils import save_plot
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


# from mkt.schema.io_utils import deserialize_kinase_dict
# from mkt.databases.plot import generate_kinase_info_plot
# DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE")
# generate_kinase_info_plot(DICT_KINASE, "/Users/jessicawhite/Library/CloudStorage/OneDrive-Personal/PhD/Chodera/missense-kinase-toolkit/images")
def generate_kinase_info_plot(
    dict_in: dict[str, Any],
    path_save: str,
    cfg: UpsetPlotConfig | None = None,
) -> None:
    """Plot KinaseInfo upset plots from final harmonzied objects.

    Parameters
    ----------
    dict_in : dict[str, Any]
        Dictionary of KinaseInfo objects.
    path_save : str
        Path to save the plot.
    cfg : UpsetPlotConfig | None
        Plot aesthetics config; uses defaults (original size) when None.

    Returns
    -------
    None
        None
    """
    if cfg is None:
        cfg = UpsetPlotConfig()

    # generate data
    list_attr = ["uniprot", "pfam", "kinhub", "klifs", "kincore"]
    list_proper = ["UniProt", "Pfam", "KinHub", "KLIFS", "KinCore"]
    list_contents = [
        [k for k, v in dict_in.items() if getattr(v, attr) is not None]
        for attr in list_attr
    ]
    dict_contents = dict(zip(list_proper, list_contents))

    # define colors for each database
    dict_colors = cfg.dict_colors

    # generate figure
    from upsetplot import from_contents, plot

    contents = from_contents(dict_contents)
    fig = plt.figure(figsize=tuple(cfg.figsize))
    upset_plot = plot(
        contents,
        fig=fig,
        element_size=cfg.element_size,
        intersection_plot_elements=cfg.intersection_plot_elements,
        totals_plot_elements=cfg.totals_plot_elements,
        sort_by="cardinality",
        sort_categories_by=None,
    )

    # add percentage labels to intersection bars
    total = len(dict_in)
    ax_intersections = upset_plot["intersections"]
    # set y-axis to log scale
    ax_intersections.set_yscale("log")
    # the axis (not the data) is log-scaled, so the label stays plain
    ax_intersections.set_ylabel("Intersection Size")
    # remove gridlines from intersection plot
    ax_intersections.grid(False)
    # remove y-axis spine
    ax_intersections.spines["left"].set_visible(False)
    # cap the y-axis at the tallest bar so the log minor ticks stop at the data
    # and don't crowd the headroom where the percentage labels float
    if cfg.cap_intersection_ylim:
        max_height = max(p.get_height() for p in ax_intersections.patches)
        ax_intersections.set_ylim(top=max_height)
    # get the bar heights from the plot patches and apply colors; clip_on=False
    # lets the tallest bar's label render just above the capped axis
    for patch in ax_intersections.patches:
        height = patch.get_height()
        percentage = (height / total) * 100
        # for log scale, multiply by a factor instead of adding
        label_y_pos = height * 1.15
        ax_intersections.text(
            patch.get_x() + patch.get_width() / 2,
            label_y_pos,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=cfg.pct_label_fontsize,
            clip_on=False,
        )

    # add total counts to category bars on the left and apply colors
    ax_totals = upset_plot["totals"]
    # remove gridlines from totals plot
    ax_totals.grid(False)
    y_tick_labels = [label.get_text() for label in ax_totals.get_yticklabels()]
    for i, patch in enumerate(ax_totals.patches):
        width = patch.get_width()
        # center the text vertically within the bar
        y_pos = patch.get_y() + patch.get_height() * 0.5
        # apply color to bar
        label = y_tick_labels[i]
        if label in dict_colors:
            patch.set_facecolor(dict_colors[label])
        # add count label with white color for KinHub (black bar)
        label_color = "#FFFFFF" if label == "KinHub" else "#000000"
        ax_totals.text(
            width,
            y_pos,
            f" {int(width)}",
            ha="left",
            va="center_baseline",
            fontsize=cfg.count_label_fontsize,
            color=label_color,
        )
        # apply color to tick label
        ax_totals.get_yticklabels()[i].set_color(dict_colors.get(label, "#000000"))

    # close the dead whitespace between the totals bars and the category labels:
    # slide the matrix/intersections left (carrying the labels) to abut the
    # totals and widen them so the freed space is filled by the bars.
    if cfg.tighten_totals_gap:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        ax_matrix = upset_plot["matrix"]
        label_left = min(
            lbl.get_window_extent(renderer).transformed(inv).x0
            for lbl in ax_matrix.get_yticklabels()
            if lbl.get_text()
        )
        totals_x1 = ax_totals.get_position().x1
        shift = label_left - (totals_x1 + cfg.totals_gap_margin)
        if shift > 0:
            # widen only the matrix + intersection bars to fill the freed space;
            # the shading already spans the full width, so leave it in place
            for key in ("matrix", "intersections"):
                ax = upset_plot.get(key)
                if ax is None:
                    continue
                p = ax.get_position()
                # keep the right edge fixed; extend left by ``shift``
                ax.set_position([p.x0 - shift, p.y0, p.width + shift, p.height])

    # tight-crop so the saved file tracks the (element_size-driven) plot area
    plt.savefig(path.join(path_save, f"{cfg.filename}.pdf"), bbox_inches="tight")


def _collect_region_gap_specs(
    dict_in: dict[str, Any],
) -> tuple[list[dict], list[dict]]:
    """Collect inter- and intra-region gap specs with per-kinase gap lengths.

    Inter-region gaps are derived from ``DICT_POCKET_KLIFS_REGIONS`` entries where
    ``contiguous`` is ``False`` (the gap spans from that region to the next one).
    Intra-region gaps are the ``KLIFS2UniProtSeq`` keys ending in ``"_intra"``.
    Gap length is the length of the stored sequence; ``None`` counts as 0.

    Parameters
    ----------
    dict_in : dict[str, Any]
        Dictionary of KinaseInfo objects.

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(inter, intra)`` lists of spec dicts, each with keys ``key``, ``label``,
        ``color_left``, ``color_right``, ``lengths`` (np.ndarray) and ``median``.
    """
    from mkt.databases.klifs import DICT_POCKET_KLIFS_REGIONS
    from mkt.schema.utils import rgetattr

    regions = list(DICT_POCKET_KLIFS_REGIONS.items())
    region_color = {r: info["color"] for r, info in regions}

    inter, intra = [], []
    for i, (region, info) in enumerate(regions):
        if not info["contiguous"] and i + 1 < len(regions):
            nxt, nxt_info = regions[i + 1]
            inter.append(
                {
                    "key": f"{region}:{nxt}",
                    "label": f"{region}–{nxt}",
                    "color_left": info["color"],
                    "color_right": nxt_info["color"],
                }
            )

    # intra keys live in KLIFS2UniProtSeq; take a representative sample's keys
    sample = next(
        (
            rgetattr(v, "KLIFS2UniProtSeq")
            for v in dict_in.values()
            if rgetattr(v, "KLIFS2UniProtSeq")
        ),
        {},
    )
    for key in sample:
        if key.endswith("_intra"):
            base = key[: -len("_intra")]
            color = region_color.get(base, "grey")
            intra.append(
                {
                    "key": key,
                    "label": base,
                    "color_left": color,
                    "color_right": color,
                }
            )

    for spec in inter + intra:
        lengths = []
        for v in dict_in.values():
            seq_dict = rgetattr(v, "KLIFS2UniProtSeq")
            seq = seq_dict.get(spec["key"]) if seq_dict else None
            lengths.append(len(seq) if seq else 0)
        spec["lengths"] = np.array(lengths)
        spec["median"] = float(np.median(lengths))

    # sort within group by median so violins shift monotonically (less overlap)
    inter = sorted(inter, key=lambda s: s["median"], reverse=True)
    intra = sorted(intra, key=lambda s: s["median"], reverse=True)
    return inter, intra


def plot_region_gap_violin(
    dict_in: dict[str, Any],
    path_save: str,
    cfg: RegionGapViolinConfig | None = None,
    rc: MatplotlibRCConfig | None = None,
) -> None:
    """Plot inter-/intra-region gap-length distributions as split-color violins.

    Each region gap gets a violin (log-scaled y-axis) over the per-kinase gap
    lengths in ``DICT_KINASE``, overlaid with jittered raw points and a
    Max/Median/Min summary table. Inter-region gaps (derived from
    ``DICT_POCKET_KLIFS_REGIONS`` ``contiguous=False`` entries) and intra-region
    gaps (``_intra`` keys) are shown as two grouped sections. When a gap's two
    flanking regions differ in color the violin is split-colored and the jitter
    uses ``cfg.jitter_mixed_color``.

    Parameters
    ----------
    dict_in : dict[str, Any]
        Dictionary of KinaseInfo objects.
    path_save : str
        Directory in which to save the plot.
    cfg : RegionGapViolinConfig | None
        Plot aesthetics config; uses defaults when None.
    rc : MatplotlibRCConfig | None
        Matplotlib rcParams config; uses defaults when None.

    Returns
    -------
    None
    """
    import matplotlib.colors as mcolors

    if cfg is None:
        cfg = RegionGapViolinConfig()
    if rc is None:
        rc = MatplotlibRCConfig()

    apply_matplotlib_rc(rc)
    plt.rcParams["font.family"] = "Arial"

    inter, intra = _collect_region_gap_specs(dict_in)
    fs = cfg.base_fontsize

    def to_log(arr):
        return np.log10(np.asarray(arr, dtype=float) + 1.0)

    def style_violin(parts, color, zorder=2):
        rgb = mcolors.to_rgb(color)
        for body in parts["bodies"]:
            # transparent fill but fully-opaque matching outline
            body.set_facecolor((*rgb, cfg.fill_alpha))
            body.set_edgecolor((*rgb, 1.0))
            body.set_linewidth(cfg.violin_linewidth)
            body.set_zorder(zorder)

    # x positions: inter group, small gap, intra group
    specs, positions = [], []
    pos = 0.0
    for s in inter:
        specs.append(s)
        positions.append(pos)
        pos += 1.0
    last_inter_x = pos - 1.0
    group_break_x = last_inter_x + cfg.break_offset
    pos = last_inter_x + cfg.group_gap
    for s in intra:
        specs.append(s)
        positions.append(pos)
        pos += 1.0

    fig, ax = plt.subplots(figsize=tuple(cfg.figsize))

    for x, s in zip(positions, specs):
        data = to_log(s["lengths"])
        vp = ax.violinplot(
            [data],
            positions=[x],
            widths=cfg.violin_width,
            showmedians=False,
            showextrema=False,
        )
        # flanking regions that differ in color use the combined color
        if s["color_left"] == s["color_right"]:
            violin_color = s["color_left"]
        else:
            violin_color = cfg.jitter_mixed_color
        style_violin(vp, violin_color)

        jitter_x = x + np.random.normal(0, cfg.jitter_std, size=len(data))
        ax.scatter(
            jitter_x,
            data,
            s=cfg.jitter_size,
            color=violin_color,
            alpha=cfg.jitter_alpha,
            edgecolors=cfg.jitter_edgecolor,
            linewidths=cfg.jitter_linewidth,
            zorder=2.5,
        )

    # pseudo summary table above the plot (Max / Median / Min per column)
    table_rows = [
        ("Max", lambda a: f"{a.max():d}", cfg.table_row_y[0]),
        ("Median", lambda a: f"{np.median(a):.0f}", cfg.table_row_y[1]),
        ("Min", lambda a: f"{a.min():d}", cfg.table_row_y[2]),
    ]
    xtrans = ax.get_xaxis_transform()
    for label, fn, yf in table_rows:
        ax.text(
            cfg.table_label_x,
            yf,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=fs,
            fontweight="bold",
            color=cfg.text_color,
            clip_on=False,
        )
        for x, s in zip(positions, specs):
            ax.text(
                x,
                yf,
                fn(s["lengths"]),
                transform=xtrans,
                ha="center",
                va="center",
                fontsize=fs,
                color=cfg.text_color,
                clip_on=False,
            )

    # group headers and divider
    inter_xs = positions[: len(inter)]
    intra_xs = positions[len(inter) :]
    ax.text(
        float(np.mean(inter_xs)),
        cfg.header_y,
        "Inter-region",
        ha="center",
        va="top",
        fontsize=fs,
        fontweight="bold",
        color=cfg.text_color,
    )
    ax.text(
        float(np.mean(intra_xs)),
        cfg.header_y,
        "Intra-region",
        ha="center",
        va="top",
        fontsize=fs,
        fontweight="bold",
        color=cfg.text_color,
    )
    ax.axvline(
        group_break_x,
        color=cfg.divider_color,
        linestyle=cfg.divider_linestyle,
        linewidth=cfg.divider_linewidth,
        zorder=0,
    )

    # log-scaled y ticks in residue units
    y_max_lin = int(max(s["lengths"].max() for s in specs))
    tick_lengths = [
        t for t in [0, 1, 2, 5, 10, 20, 50, 100, 200, 300] if t <= y_max_lin
    ]
    ax.set_yticks(to_log(tick_lengths))
    ax.set_yticklabels([str(t) for t in tick_lengths])
    ax.set_ylim(-0.15, to_log(y_max_lin) + 0.4)

    ax.set_xticks(positions)
    ax.set_xticklabels([s["label"] for s in specs], fontsize=fs)
    ax.set_ylabel(cfg.ylabel_text, fontsize=fs)
    ax.tick_params(axis="y", labelsize=fs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=cfg.grid_alpha)

    # fixed margins (not tight_layout) so the saved canvas equals figsize
    plt.subplots_adjust(
        left=cfg.left_adjust,
        right=cfg.right_adjust,
        top=cfg.top_adjust,
        bottom=cfg.bottom_adjust,
    )

    # bbox_inches=None keeps the output exactly figsize (not content-cropped)
    save_plot(
        fig,
        cfg.filename,
        "Region gap violin",
        bool_force_local=False,
        bool_image_subdir=False,
        output_path=path_save,
        bbox_inches=None,
    )


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
        from bokeh.models import ColumnDataSource

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
        from bokeh.models import Range1d
        from bokeh.models.glyphs import Rect
        from bokeh.plotting import figure

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
        from bokeh.models.glyphs import Rect, Text
        from bokeh.plotting import figure

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
        from bokeh.layouts import gridplot

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


# ---------------------------------------------------------------------------
# Dataset plotting functions (config-driven)
# ---------------------------------------------------------------------------


def apply_matplotlib_rc(rc: MatplotlibRCConfig) -> None:
    """Apply global matplotlib rcParams from config."""
    import matplotlib

    matplotlib.rcParams["svg.fonttype"] = rc.svg_fonttype
    matplotlib.rcParams["pdf.fonttype"] = rc.pdf_fonttype
    matplotlib.rcParams["text.usetex"] = rc.text_usetex


def generate_venn_diagram_dict(df_in: pd.DataFrame) -> dict:
    """Generate a dictionary for Venn diagram plotting.

    Parameters:
    -----------
    df_in : pd.DataFrame
        DataFrame with columns: kinase_name, seq_construct_unaligned, seq_klifs_region_aligned, seq_klifs_residues_only

    Returns:
    --------
    dict
        Dictionary with keys as sequence types and values as lists of kinase names.
    """
    dict_out = {
        "Construct Unaligned": df_in.loc[
            ~df_in["seq_construct_unaligned"].isna(), "kinase_name"
        ]
        .unique()
        .tolist(),
        "KLIFS Region Aligned": df_in.loc[
            ~df_in["seq_klifs_region_aligned"].isna(), "kinase_name"
        ]
        .unique()
        .tolist(),
        "Klifs Residues Only": df_in.loc[
            ~df_in["seq_klifs_residues_only"].isna(), "kinase_name"
        ]
        .unique()
        .tolist(),
    }

    return dict_out


def convert_to_percentile(input, orig_max=10) -> float:
    """Convert Kd values to percentile scale."""
    return (input / orig_max) * 100


def convert_from_percentile(input, orig_max=10, precision=3) -> float:
    """Convert percentile values back to Kd scale."""
    if precision is not None:
        try:
            return round((input / 100) * orig_max, precision)
        except TypeError:
            return [np.round(i, precision) for i in (input / 100) * orig_max]
    else:
        return (input / 100) * max(orig_max)


def plot_dynamic_range(
    df_davis: pd.DataFrame,
    df_pkis2: pd.DataFrame,
    output_path: str,
    cfg: DynamicRangePlotConfig | None = None,
    rc: MatplotlibRCConfig | None = None,
) -> None:
    """Create a histogram comparing dynamic assay ranges between Davis and PKIS2.

    Parameters:
    -----------
    df_davis : pd.DataFrame
        DataFrame with 'y' column containing Kd values.
    df_pkis2 : pd.DataFrame
        DataFrame with 'y' column containing percent inhibition values.
    output_path : str
        Path to save the plot files (will save both SVG and PNG).
    cfg : DynamicRangePlotConfig | None
        Plot aesthetics config; uses defaults when None.
    rc : MatplotlibRCConfig | None
        Matplotlib rcParams config; uses defaults when None.
    """
    import matplotlib
    import matplotlib.ticker

    if cfg is None:
        cfg = DynamicRangePlotConfig()
    if rc is None:
        rc = MatplotlibRCConfig()

    apply_matplotlib_rc(rc)

    # process Davis data
    col_davis_y = "y"
    col_davis_y_transformed = "y_trans"
    df_davis = df_davis.copy()
    df_davis[col_davis_y] = 10 ** (-df_davis[col_davis_y])
    df_davis[col_davis_y_transformed] = convert_to_percentile(df_davis[col_davis_y])

    # process PKIS2 data
    df_pkis2 = df_pkis2.copy()
    df_pkis2["1-Percent Inhibition"] = 100 - df_pkis2["y"]

    # calculate no binding percentages
    na_davis = (
        sum(df_davis[col_davis_y] == df_davis[col_davis_y].max()) / df_davis.shape[0]
    )
    na_pkis2 = sum(df_pkis2["y"] == 0) / df_pkis2.shape[0]

    fig, ax1 = plt.subplots()
    fig.set_size_inches(*cfg.figsize)
    matplotlib.rcParams.update({"font.size": cfg.font_size})
    matplotlib.rcParams.update(
        {"axes.titlesize": cfg.axes_titlesize, "axes.labelsize": cfg.axes_labelsize}
    )
    matplotlib.rcParams.update({"figure.titlesize": cfg.figure_titlesize})

    sns.histplot(
        data=df_pkis2,
        x="1-Percent Inhibition",
        ax=ax1,
        bins=cfg.bins,
        log=True,
        color=cfg.color_pkis2,
        alpha=cfg.alpha,
        label=f"PKIS2 (n={df_pkis2.shape[0]:,})",
    )

    sns.histplot(
        data=df_davis,
        x=col_davis_y_transformed,
        ax=ax1,
        bins=cfg.bins,
        log=True,
        color=cfg.color_davis,
        alpha=cfg.alpha,
        label=f"Davis (n={df_davis.shape[0]:,})",
    )

    ax1.xaxis.label.set_size(cfg.axis_label_fontsize)
    ax1.yaxis.label.set_size(cfg.axis_label_fontsize)
    ax1.tick_params(axis="x", labelsize=cfg.tick_labelsize)
    ax1.tick_params(axis="y", labelsize=cfg.tick_labelsize)
    ax1.set_xlabel(
        "1-% inhibition (PKIS2)",
        color=cfg.color_pkis2,
        fontsize=cfg.axis_label_fontsize,
    )
    ax1.axvline(x=cfg.axvline_x, color=cfg.axvline_color, linestyle="--")

    ax1.text(
        x=0.5,
        y=cfg.title_y,
        s="Comparing dynamic assay ranges",
        fontsize=cfg.title_fontsize,
        weight=cfg.title_fontweight,
        ha="center",
        va="bottom",
        transform=ax1.transAxes,
    )

    ax1.text(
        x=0.5,
        y=cfg.subtitle_y,
        s=f"No binding detected: {na_davis:.1%} Davis, {na_pkis2:.1%} PKIS2",
        fontsize=cfg.subtitle_fontsize,
        alpha=cfg.subtitle_alpha,
        ha="center",
        va="bottom",
        transform=ax1.transAxes,
    )

    ax1.get_xaxis().set_major_formatter(lambda x, p: format(x / 100, ".0%"))
    ax2 = ax1.secondary_xaxis(
        "top", functions=(convert_from_percentile, convert_to_percentile)
    )
    ax2.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax2.xaxis.label.set_size(cfg.axis_label_fontsize)
    ax2.set_xlabel(
        r"$K_{\mathrm{d}}$ ($\mu$M) (Davis)",
        color=cfg.color_davis,
        fontsize=cfg.axis_label_fontsize,
    )
    ax1.set_ylabel(r"$\mathrm{log_{10}}$(count)", fontsize=cfg.axis_label_fontsize)
    plt.legend(loc="upper left")
    plt.xlim(0, 100)
    plt.tight_layout()

    save_plot(
        fig,
        os.path.basename(output_path),
        "Dynamic range plot",
        bool_force_local=False,
        bool_image_subdir=False,
        output_path=os.path.dirname(output_path),
    )


def plot_ridgeline(
    df: pd.DataFrame,
    output_path: str,
    cfg: RidgelinePlotConfig | None = None,
    rc: MatplotlibRCConfig | None = None,
    family_cfg: FamilyColorConfig | None = None,
) -> None:
    """Create a ridgeline plot showing distribution of fraction_construct by family.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: kinase_name, family, fraction_construct, source.
    output_path : str
        Path to save the SVG file.
    cfg : RidgelinePlotConfig | None
        Plot aesthetics config; uses defaults when None.
    rc : MatplotlibRCConfig | None
        Matplotlib rcParams config; uses defaults when None.
    family_cfg : FamilyColorConfig | None
        Family color config; uses defaults when None.
    """
    from scipy import stats

    if cfg is None:
        cfg = RidgelinePlotConfig()
    if rc is None:
        rc = MatplotlibRCConfig()

    apply_matplotlib_rc(rc)

    # remove rows with None values
    df_clean = df.dropna(subset=["family", "fraction_construct"])

    # get color palette
    family_colors = family_cfg.get_colors()

    # get unique sources
    sources = sorted(df_clean["source"].unique())

    # sort families by Davis median
    df_davis = df_clean[df_clean["source"] == "Davis"]
    family_medians = (
        df_davis.groupby("family")["fraction_construct"]
        .median()
        .sort_values(ascending=False)
    )
    families = family_medians.index.tolist()

    # create figure with subplots for each source (horizontal faceting)
    fig, axes = plt.subplots(1, len(sources), figsize=cfg.figsize, sharey=True)
    if len(sources) == 1:
        axes = [axes]

    for idx, source in enumerate(sources):
        ax = axes[idx]
        df_source = df_clean[df_clean["source"] == source]

        # count number of unique constructs
        n_constructs = df_source["kinase_name"].nunique()

        for i, family in enumerate(families):
            df_family = df_source[df_source["family"] == family]
            if len(df_family) > 0:
                data = df_family["fraction_construct"].values

                # calculate kernel density estimate
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(0, 1, 200)
                density = kde(x_range)

                # normalize and scale density
                density = density / density.max() * cfg.scale

                # plot the density as a filled curve with overlap
                y_base = i * (1 - cfg.overlap)
                x_range_pct = x_range * 100
                ax.fill_between(
                    x_range_pct,
                    y_base,
                    y_base + density,
                    color=family_colors.get(family, "grey"),
                    alpha=cfg.fill_alpha,
                    edgecolor=cfg.edgecolor,
                    linewidth=cfg.edge_linewidth,
                    zorder=len(families) - i,
                )

                # add a baseline for reference
                ax.plot(
                    x_range_pct,
                    [y_base] * len(x_range_pct),
                    color="black",
                    linewidth=cfg.baseline_linewidth,
                    alpha=cfg.baseline_alpha,
                )

        # set labels and title with construct count
        y_positions = [i * (1 - cfg.overlap) for i in range(len(families))]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(families, fontsize=cfg.ytick_fontsize)
        if source.lower() == "davis":
            title_color = cfg.title_color_davis
        elif source.lower() == "pkis2":
            title_color = cfg.title_color_pkis2
        else:
            title_color = "black"
        ax.set_title(
            f"{source} (n={n_constructs})",
            fontsize=cfg.title_fontsize,
            fontweight=cfg.title_fontweight,
            color=title_color,
        )
        ax.tick_params(axis="x", labelsize=cfg.xtick_fontsize)

        # set y limits with padding
        ax.set_ylim(-0.5, max(y_positions) + cfg.scale + 0.5)
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=cfg.grid_alpha)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # only show y-axis labels on the leftmost plot
    for idx in range(1, len(axes)):
        axes[idx].set_ylabel("")
        axes[idx].tick_params(left=False)

    # add shared x-axis label
    fig.text(
        0.5,
        0.02,
        cfg.xlabel_text,
        ha="center",
        fontsize=cfg.xlabel_fontsize,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    save_plot(
        fig,
        os.path.basename(output_path),
        "Ridgeline plot",
        bool_force_local=False,
        bool_image_subdir=False,
        output_path=os.path.dirname(output_path),
    )


def plot_stacked_barchart(
    df: pd.DataFrame,
    output_path: str,
    cfg: StackedBarchartConfig | None = None,
    rc: MatplotlibRCConfig | None = None,
    family_cfg: FamilyColorConfig | None = None,
) -> None:
    """Create a stacked bar chart showing counts by family.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: family, bool_uniprot2refseq, count, source.
    output_path : str
        Path to save the SVG file.
    cfg : StackedBarchartConfig | None
        Plot aesthetics config; uses defaults when None.
    rc : MatplotlibRCConfig | None
        Matplotlib rcParams config; uses defaults when None.
    family_cfg : FamilyColorConfig | None
        Family color config; uses defaults when None.
    """
    if cfg is None:
        cfg = StackedBarchartConfig()
    if rc is None:
        rc = MatplotlibRCConfig()

    apply_matplotlib_rc(rc)

    # remove rows with None values
    df_clean = df.dropna(subset=["family"])

    # get color palette
    family_colors = family_cfg.get_colors()

    # pivot data for stacking
    sources = sorted(df_clean["source"].unique())

    # create figure with subplots for each source
    nrows = cfg.layout_nrows
    ncols = int(np.ceil(len(sources) / nrows))
    if nrows == 1:
        fig_w = cfg.figsize_width_per_source * ncols
        fig_h = cfg.figsize_height
    else:
        fig_w = cfg.figsize_width_per_source
        fig_h = cfg.figsize_height * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    if len(sources) == 1:
        axes = [axes]
    else:
        axes = list(np.array(axes).flat)

    stack_colors = {"True": cfg.stack_color_true, "False": cfg.stack_color_false}

    for idx, source in enumerate(sources):
        ax = axes[idx]
        df_source = df_clean[df_clean["source"] == source]

        # pivot to get families vs bool_uniprot2refseq
        df_pivot = df_source.pivot_table(
            index="family", columns="bool_uniprot2refseq", values="count", fill_value=0
        )

        # calculate percentages
        df_pivot_pct = df_pivot.div(df_pivot.sum(axis=1), axis=0) * 100

        # calculate total counts for each family
        df_counts = df_pivot.sum(axis=1)

        # total number of constructs
        n_constructs = int(df_counts.sum())

        # sort families by highest % False to lowest % False
        if False in df_pivot_pct.columns:
            family_order = (
                df_pivot_pct[False].sort_values(ascending=False).index.tolist()
            )
        else:
            family_order = sorted(df_pivot_pct.index.tolist())

        # prepare data for stacked bar chart
        x_pos = np.arange(len(family_order))

        # plot False on top, True on bottom (reverse order)
        bool_values = [True, False]

        bottom = np.zeros(len(family_order))
        for bool_val in bool_values:
            if bool_val in df_pivot_pct.columns:
                values = [
                    df_pivot_pct.loc[fam, bool_val] if fam in df_pivot_pct.index else 0
                    for fam in family_order
                ]

                bars = ax.bar(
                    x_pos,
                    values,
                    bottom=bottom,
                    color=stack_colors[str(bool_val)],
                    edgecolor=cfg.bar_edgecolor,
                    linewidth=cfg.bar_linewidth,
                    alpha=cfg.bar_alpha,
                )

                # add percentage labels
                text_color = "white" if bool_val is False else "black"
                for i, (bar, val) in enumerate(zip(bars, values)):
                    if val > cfg.pct_label_min_threshold:
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            bottom[i] + height / 2.0,
                            f"{val:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=cfg.pct_label_fontsize,
                            fontweight=cfg.pct_label_fontweight,
                            color=text_color,
                        )

                bottom += values

        # create x-axis labels with family colors and counts
        family_labels = []
        for fam in family_order:
            count = int(df_counts[fam]) if fam in df_counts else 0
            family_labels.append(f"{fam}\n(n={count})")

        # set labels and title
        ax.set_xticks(x_pos)
        ax.set_xticklabels(family_labels, ha="center", fontsize=cfg.xtick_fontsize)

        # color the x-axis labels
        for i, (tick_label, fam) in enumerate(zip(ax.get_xticklabels(), family_order)):
            tick_label.set_color(family_colors.get(fam, "grey"))
            tick_label.set_fontweight("bold")

        # only show x-axis label on bottom row
        is_bottom_row = (idx // ncols) == (nrows - 1)
        if is_bottom_row or nrows == 1:
            ax.set_xlabel("Kinase Family", fontsize=cfg.xlabel_fontsize)
        else:
            ax.set_xlabel("")
        ax.set_ylabel("Percentage (%)", fontsize=cfg.ylabel_fontsize)
        if source.lower() == "davis":
            title_color = cfg.title_color_davis
        elif source.lower() == "pkis2":
            title_color = cfg.title_color_pkis2
        else:
            title_color = "black"
        ax.set_title(
            f"{source} (n={n_constructs})",
            fontsize=cfg.title_fontsize,
            fontweight=cfg.title_fontweight,
            color=title_color,
        )
        ax.tick_params(axis="y", labelsize=cfg.ytick_fontsize)
        ax.set_ylim(0, cfg.ylim_max)

    # create shared legend patches
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=stack_colors["True"], edgecolor="black", label="True"),
        Patch(facecolor=stack_colors["False"], edgecolor="black", label="False"),
    ]

    fig.legend(
        handles=legend_elements,
        title="RefSeq sequence identical to UniProt sequence",
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=cfg.legend_fontsize,
        title_fontsize=cfg.legend_title_fontsize,
        bbox_to_anchor=(0.5, cfg.legend_bbox_y),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=cfg.bottom_adjust)

    save_plot(
        fig,
        os.path.basename(output_path),
        "Stacked bar chart",
        bool_force_local=False,
        bool_image_subdir=False,
        output_path=os.path.dirname(output_path),
    )


def plot_venn_diagram(
    df: pd.DataFrame,
    output_path: str,
    source_name: str,
    cfg: VennDiagramConfig | None = None,
    rc: MatplotlibRCConfig | None = None,
    color_cfg: ColKinaseColorConfig | None = None,
) -> None:
    """Create a Venn diagram showing overlap of kinases across different sequence types.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: kinase_name, seq_construct_unaligned, seq_klifs_region_aligned, seq_klifs_residues_only.
    output_path : str
        Path to save the SVG file.
    source_name : str
        Name of the source (e.g., 'Davis', 'PKIS2').
    cfg : VennDiagramConfig | None
        Plot aesthetics config; uses defaults when None.
    rc : MatplotlibRCConfig | None
        Matplotlib rcParams config; uses defaults when None.
    color_cfg : ColKinaseColorConfig | None
        Col kinase color config; uses defaults when None.
    """
    from matplotlib_venn import venn3

    if cfg is None:
        cfg = VennDiagramConfig()
    if rc is None:
        rc = MatplotlibRCConfig()
    if color_cfg is None:
        color_cfg = ColKinaseColorConfig()

    apply_matplotlib_rc(rc)

    colors = color_cfg.as_rgb_dict()

    # generate the Venn diagram dictionary
    venn_dict = generate_venn_diagram_dict(df)

    # convert lists to sets for Venn diagram
    set_construct_unaligned = set(venn_dict["Construct Unaligned"])
    set_klifs_region_aligned = set(venn_dict["KLIFS Region Aligned"])
    set_klifs_residues_only = set(venn_dict["Klifs Residues Only"])

    # create figure
    fig, ax = plt.subplots(figsize=cfg.figsize)

    # create Venn diagram
    venn = venn3(
        [set_construct_unaligned, set_klifs_region_aligned, set_klifs_residues_only],
        set_labels=(
            "Construct Unaligned",
            "KLIFS Region Aligned",
            "KLIFS Residues Only",
        ),
        ax=ax,
    )

    # color the circles
    if venn.get_patch_by_id("100"):
        venn.get_patch_by_id("100").set_color(colors["construct_unaligned"])
        venn.get_patch_by_id("100").set_alpha(cfg.circle_alpha)
    if venn.get_patch_by_id("010"):
        venn.get_patch_by_id("010").set_color(colors["klifs_region_aligned"])
        venn.get_patch_by_id("010").set_alpha(cfg.circle_alpha)
    if venn.get_patch_by_id("001"):
        venn.get_patch_by_id("001").set_color(colors["klifs_residues_only"])
        venn.get_patch_by_id("001").set_alpha(cfg.circle_alpha)

    # color intersections
    for patch_id in ["110", "101", "011", "111"]:
        patch = venn.get_patch_by_id(patch_id)
        if patch:
            patch.set_color(cfg.intersection_color)
            patch.set_alpha(cfg.intersection_alpha)

    # increase label font sizes
    for label in venn.set_labels:
        if label:
            label.set_fontsize(cfg.set_label_fontsize)
            label.set_fontweight(cfg.set_label_fontweight)

    # increase count font sizes
    for label in venn.subset_labels:
        if label:
            label.set_fontsize(cfg.subset_label_fontsize)

    ax.set_title(
        f"{source_name} Kinase Coverage",
        fontsize=cfg.title_fontsize,
        fontweight=cfg.title_fontweight,
    )

    plt.tight_layout()

    save_plot(
        fig,
        os.path.basename(output_path),
        "Venn diagram",
        bool_force_local=False,
        bool_image_subdir=False,
        output_path=os.path.dirname(output_path),
    )


def plot_metrics_boxplot(
    df: pd.DataFrame,
    output_path: str,
    cfg: MetricsBoxplotConfig | None = None,
    rc: MatplotlibRCConfig | None = None,
    color_cfg: ColKinaseColorConfig | None = None,
) -> None:
    """Create a boxplot with jitter showing MSE values by col_kinase and source.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: col_kinase, source, fold, avg_stable_epoch, mse.
    output_path : str
        Path to save the SVG file.
    cfg : MetricsBoxplotConfig | None
        Plot aesthetics config; uses defaults when None.
    rc : MatplotlibRCConfig | None
        Matplotlib rcParams config; uses defaults when None.
    color_cfg : ColKinaseColorConfig | None
        Col kinase color config; uses defaults when None.
    """
    if cfg is None:
        cfg = MetricsBoxplotConfig()
    if rc is None:
        rc = MatplotlibRCConfig()
    if color_cfg is None:
        color_cfg = ColKinaseColorConfig()

    apply_matplotlib_rc(rc)

    col_kinase_colors = color_cfg.as_rgb_dict()

    # format col_kinase labels
    def format_col_kinase(col_name, avg_epoch):
        formatted = col_name.replace("_", " ").title()
        formatted = formatted.replace("Klifs", "KLIFS")
        return f"{formatted}\n(Epoch={int(avg_epoch)})"

    # get unique col_kinase values and their average stable epochs per source
    col_kinase_source_info = (
        df.groupby(["col_kinase", "source"])["avg_stable_epoch"].first().to_dict()
    )

    # get unique sources and sort
    sources = sorted(df["source"].unique())

    # calculate width ratios based on number of unique col_kinase values per source
    width_ratios = []
    for source in sources:
        df_source = df[df["source"] == source]
        n_col_kinases = df_source["col_kinase"].nunique()
        width_ratios.append(n_col_kinases)

    # create figure with subplots for each source
    fig, axes = plt.subplots(
        1,
        len(sources),
        figsize=cfg.figsize,
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
    )
    if len(sources) == 1:
        axes = [axes]

    for idx, source in enumerate(sources):
        ax = axes[idx]
        df_source = df[df["source"] == source]

        # get unique col_kinase values for this source
        col_kinases = sorted(df_source["col_kinase"].unique())

        # prepare data for boxplot
        data_for_boxplot = []
        labels = []
        colors = []
        for col_kinase in col_kinases:
            df_col = df_source[df_source["col_kinase"] == col_kinase]
            data_for_boxplot.append(df_col["mse"].values)
            avg_epoch = col_kinase_source_info[(col_kinase, source)]
            labels.append(format_col_kinase(col_kinase, avg_epoch))
            colors.append(col_kinase_colors.get(col_kinase, (0.5, 0.5, 0.5)))

        # create boxplot
        bp = ax.boxplot(
            data_for_boxplot,
            positions=range(len(col_kinases)),
            widths=cfg.box_widths,
            patch_artist=True,
            medianprops=dict(color=cfg.median_color, linewidth=cfg.median_linewidth),
            whiskerprops=dict(color=cfg.whisker_color, linewidth=cfg.whisker_linewidth),
            capprops=dict(color=cfg.cap_color, linewidth=cfg.cap_linewidth),
        )

        # color each box
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(cfg.box_alpha)

        # add jittered points
        for i, col_kinase in enumerate(col_kinases):
            df_col = df_source[df_source["col_kinase"] == col_kinase]
            y = df_col["mse"].values
            x = np.random.normal(i, cfg.jitter_std, size=len(y))
            ax.scatter(
                x,
                y,
                alpha=cfg.jitter_alpha,
                s=cfg.jitter_size,
                color=cfg.jitter_color,
                zorder=3,
            )

        # set labels and title
        ax.set_xticks(range(len(col_kinases)))
        ax.set_xticklabels(labels, fontsize=cfg.xtick_fontsize)
        ax.set_ylabel(cfg.ylabel_text, fontsize=cfg.ylabel_fontsize)
        # color facet title by dataset
        if source.lower() == "davis":
            title_color = cfg.title_color_davis
        elif source.lower() == "pkis2":
            title_color = cfg.title_color_pkis2
        else:
            title_color = "black"
        ax.set_title(
            source.title() if source == "davis" else source.upper(),
            fontsize=cfg.title_fontsize,
            fontweight=cfg.title_fontweight,
            color=title_color,
        )
        ax.tick_params(axis="y", labelsize=cfg.ytick_fontsize)
        ax.grid(axis="y", alpha=cfg.grid_alpha)

    # store comparison info for later
    comparison_info = []
    for idx, source in enumerate(sources):
        ax = axes[idx]
        df_source = df[df["source"] == source]
        col_kinases = sorted(df_source["col_kinase"].unique())

        # add p-value comparisons for Davis dataset only
        if source == "davis" and "construct_unaligned" in col_kinases:
            from scipy import stats as scipy_stats

            df_construct_unaligned = df_source[
                df_source["col_kinase"] == "construct_unaligned"
            ]
            data_construct_unaligned = df_construct_unaligned["mse"].values
            pos_construct_unaligned = col_kinases.index("construct_unaligned")

            comparisons = []
            if "klifs_region_aligned" in col_kinases:
                df_klifs_region = df_source[
                    df_source["col_kinase"] == "klifs_region_aligned"
                ]
                data_klifs_region = df_klifs_region["mse"].values
                pos_klifs_region = col_kinases.index("klifs_region_aligned")

                t_stat, p_val = scipy_stats.ttest_ind(
                    data_construct_unaligned, data_klifs_region
                )
                comparisons.append((pos_construct_unaligned, pos_klifs_region, p_val))

            if "klifs_residues_only" in col_kinases:
                df_klifs_residues = df_source[
                    df_source["col_kinase"] == "klifs_residues_only"
                ]
                data_klifs_residues = df_klifs_residues["mse"].values
                pos_klifs_residues = col_kinases.index("klifs_residues_only")

                t_stat, p_val = scipy_stats.ttest_ind(
                    data_construct_unaligned, data_klifs_residues
                )
                comparisons.append((pos_construct_unaligned, pos_klifs_residues, p_val))

            comparison_info.append((idx, comparisons))

    # draw comparison brackets after determining the shared y-axis limits
    if comparison_info:
        y_max = max(ax.get_ylim()[1] for ax in axes)
        y_min = min(ax.get_ylim()[0] for ax in axes)
        y_range = y_max - y_min

        bracket_start = y_max + cfg.bracket_start_pct * y_range
        bracket_spacing = cfg.bracket_spacing_pct * y_range
        bracket_height = cfg.bracket_height_pct * y_range

        for ax_idx, comparisons in comparison_info:
            ax = axes[ax_idx]

            for i, (pos1, pos2, p_val) in enumerate(comparisons):
                y = bracket_start + i * bracket_spacing
                h = bracket_height

                ax.plot(
                    [pos1, pos1, pos2, pos2],
                    [y, y + h, y + h, y],
                    lw=cfg.bracket_linewidth,
                    c="black",
                )

                if p_val < 0.001:
                    p_text = "p < 0.001"
                elif p_val < 0.01:
                    p_text = f"p = {p_val:.3f}**"
                elif p_val < 0.05:
                    p_text = f"p = {p_val:.2f}*"
                else:
                    p_text = f"p = {p_val:.2f}"

                ax.text(
                    (pos1 + pos2) * 0.5,
                    y + h,
                    p_text,
                    ha="center",
                    va="bottom",
                    fontsize=cfg.pvalue_fontsize,
                    fontweight=cfg.pvalue_fontweight,
                )

        new_y_max = (
            bracket_start
            + max(len(comps) for _, comps in comparison_info) * bracket_spacing
            + 0.1 * y_range
        )
        for ax in axes:
            ax.set_ylim(y_min, new_y_max)

    # only show y-axis label on the leftmost plot
    for idx in range(1, len(axes)):
        axes[idx].set_ylabel("")

    plt.tight_layout()

    save_plot(
        fig,
        os.path.basename(output_path),
        "Metrics boxplot",
        bool_force_local=False,
        bool_image_subdir=False,
        output_path=os.path.dirname(output_path),
    )


def _get_klifs_position_colors() -> list[tuple[str, str]]:
    """Build a list of (region_name, color) for each of the 85 KLIFS pocket positions.

    Returns:
    --------
    list[tuple[str, str]]
        List of (region_name, color) tuples, one per KLIFS pocket position (85 total).
    """
    from mkt.databases.klifs import DICT_POCKET_KLIFS_REGIONS

    klifs_pos = []
    for region, info in DICT_POCKET_KLIFS_REGIONS.items():
        for _ in range(info["start"], info["end"] + 1):
            klifs_pos.append((region, info["color"]))
    return klifs_pos


def _map_aligned_to_klifs_colors(
    seq_aligned: str,
    seq_klifs_only: str,
    dict_aa_colors: dict[str, str],
    klifs_pos_colors: list[tuple[str, str]],
    gap_color: str = "white",
) -> list[str]:
    """Map each position in an aligned sequence to a color.

    Core KLIFS pocket residues get their KLIFS region color; inter/intra MSA
    residues get alphabet palette colors; gaps get the gap color.

    Parameters:
    -----------
    seq_aligned : str
        Full aligned sequence (e.g. 970 chars).
    seq_klifs_only : str
        KLIFS-residues-only sequence (85 chars).
    dict_aa_colors : dict[str, str]
        Amino acid to color mapping (alphabet palette).
    klifs_pos_colors : list[tuple[str, str]]
        Per-position (region_name, color) from ``_get_klifs_position_colors``.
    gap_color : str
        Color for gap characters.

    Returns:
    --------
    list[str]
        Color string for each position in ``seq_aligned``.
    """
    # build set of KLIFS non-gap residue indices in the aligned sequence
    klifs_nongap_chars = [(i, c) for i, c in enumerate(seq_klifs_only) if c != "-"]
    aligned_nongap_positions = [i for i, c in enumerate(seq_aligned) if c != "-"]

    # match KLIFS non-gap residues to aligned non-gap positions (subsequence match)
    klifs_aligned_pos_to_region_color = {}
    klifs_idx = 0
    aligned_nongap_idx = 0
    for pos in aligned_nongap_positions:
        if klifs_idx < len(klifs_nongap_chars):
            # the KLIFS pocket index (0-84) for this KLIFS residue
            klifs_pocket_idx = klifs_nongap_chars[klifs_idx][0]
            if seq_aligned[pos] == klifs_nongap_chars[klifs_idx][1]:
                _, color = klifs_pos_colors[klifs_pocket_idx]
                klifs_aligned_pos_to_region_color[pos] = color
                klifs_idx += 1
                continue
        aligned_nongap_idx += 1

    colors = []
    for i, c in enumerate(seq_aligned):
        if c == "-":
            colors.append(gap_color)
        elif i in klifs_aligned_pos_to_region_color:
            colors.append(klifs_aligned_pos_to_region_color[i])
        else:
            colors.append(dict_aa_colors.get(c, "#CCCCCC"))
    return colors


def _draw_sequence_strip(
    ax,
    y: float,
    colors: list[str],
    rect_height: float = 0.6,
    x_offset: float = 0,
) -> None:
    """Draw a horizontal strip of colored rectangles representing a sequence.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    y : float
        Vertical center of the strip.
    colors : list[str]
        Color for each position.
    rect_height : float
        Height of each rectangle.
    x_offset : float
        Horizontal offset for the strip start.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    patches = []
    facecolors = []
    for i, color in enumerate(colors):
        if color == "white":
            continue
        patches.append(Rectangle((x_offset + i, y - rect_height / 2), 1, rect_height))
        facecolors.append(color)

    if patches:
        pc = PatchCollection(patches, facecolors=facecolors, edgecolors="none")
        ax.add_collection(pc)


def plot_sequence_input_schematic(
    df: pd.DataFrame,
    output_path: str,
    list_kinases: list[str] | None = None,
    cfg: SequenceSchematicConfig | None = None,
    rc: MatplotlibRCConfig | None = None,
) -> None:
    """Create a 3-panel schematic comparing the three sequence input representations.

    Panel A shows unaligned construct sequences (variable length, alphabet palette).
    Panel B shows KLIFS-region-aligned sequences (fixed length, KLIFS palette for
    pocket regions, alphabet palette for MSA inter/intra regions).
    Panel C shows KLIFS-residues-only sequences (85 positions, KLIFS palette).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: kinase_name, seq_construct_unaligned,
        seq_klifs_region_aligned, seq_klifs_residues_only.
    output_path : str
        Path to save the plot files.
    list_kinases : list[str] | None
        Kinase names to display. If None, uses a default selection of 5 kinases
        with comparable unaligned sequence lengths.
    cfg : SequenceSchematicConfig | None
        Plot aesthetics config; uses defaults when None.
    rc : MatplotlibRCConfig | None
        Matplotlib rcParams config; uses defaults when None.
    """
    from matplotlib.patches import Patch
    from mkt.databases.colors import DICT_COLORS

    if cfg is None:
        cfg = SequenceSchematicConfig()
    if rc is None:
        rc = MatplotlibRCConfig()

    apply_matplotlib_rc(rc)
    plt.rcParams["font.family"] = "Arial"

    dict_aa_colors = DICT_COLORS["ALPHABET_PROJECT"]["DICT_COLORS"]
    klifs_pos_colors = _get_klifs_position_colors()

    # select kinases
    df_valid = df.dropna(
        subset=[
            "seq_construct_unaligned",
            "seq_klifs_region_aligned",
            "seq_klifs_residues_only",
        ]
    )
    kinase_seqs = df_valid.drop_duplicates("kinase_name").set_index("kinase_name")

    if list_kinases is None:
        # pick 5 kinases with comparable but different unaligned lengths
        kinase_seqs["_len"] = kinase_seqs["seq_construct_unaligned"].str.len()
        median_len = kinase_seqs["_len"].median()
        candidates = kinase_seqs.iloc[
            (kinase_seqs["_len"] - median_len).abs().argsort()
        ]
        # select 5 with distinct lengths near the median
        selected = []
        seen_lengths = set()
        for name, row in candidates.iterrows():
            seq_len = len(row["seq_construct_unaligned"])
            if seq_len not in seen_lengths:
                selected.append(name)
                seen_lengths.add(seq_len)
            if len(selected) == 5:
                break
        list_kinases = sorted(
            selected,
            key=lambda k: len(kinase_seqs.loc[k, "seq_construct_unaligned"]),
        )

    n_kinases = len(list_kinases)

    # --- build figure with 3 panels ---
    fig, axes = plt.subplots(
        1, 3, figsize=cfg.figsize, gridspec_kw={"width_ratios": [3, 5, 1]}
    )

    # =====================================================================
    # panel A: construct unaligned (variable length, truncated with ellipsis)
    # =====================================================================
    ax_a = axes[0]
    n_start = cfg.n_show_start
    n_end = cfg.n_show_end

    max_len = max(
        len(kinase_seqs.loc[k, "seq_construct_unaligned"]) for k in list_kinases
    )
    min_len = min(
        len(kinase_seqs.loc[k, "seq_construct_unaligned"]) for k in list_kinases
    )
    # gap between shown start and end blocks (scaled to emphasize length diffs)
    ellipsis_gap = max(15, int((max_len - n_start - n_end) * 0.08))

    for i, kinase in enumerate(list_kinases):
        seq = kinase_seqs.loc[kinase, "seq_construct_unaligned"]
        seq_len = len(seq)
        y = n_kinases - 1 - i

        # first n_start residues
        colors_start = [dict_aa_colors.get(c, "#CCCCCC") for c in seq[:n_start]]
        _draw_sequence_strip(ax_a, y, colors_start, cfg.rect_height)

        # last n_end residues, positioned proportionally to actual length
        # the longest kinase's end block ends at a fixed right edge;
        # shorter kinases end earlier
        length_range = max_len - min_len if max_len > min_len else 1
        extra = (seq_len - min_len) / length_range * n_end
        end_x = n_start + ellipsis_gap + extra
        colors_end = [dict_aa_colors.get(c, "#CCCCCC") for c in seq[-n_end:]]
        _draw_sequence_strip(ax_a, y, colors_end, cfg.rect_height, x_offset=end_x)

        # ellipsis dots
        mid_x = n_start + ellipsis_gap / 2
        ax_a.text(
            mid_x,
            y,
            "\u2026",
            ha="center",
            va="center",
            fontsize=cfg.ellipsis_fontsize,
            color=cfg.ellipsis_color,
        )

        # length annotation at right edge of each strip
        ax_a.text(
            end_x + n_end + 0.5,
            y,
            str(seq_len),
            ha="left",
            va="center",
            fontsize=cfg.label_fontsize - 2,
            color="#555555",
        )

    right_edge = n_start + ellipsis_gap + n_end * 2 + 8
    ax_a.set_yticks(range(n_kinases))
    ax_a.set_yticklabels(list_kinases[::-1], fontsize=cfg.label_fontsize)
    ax_a.set_xlim(-1, right_edge)
    ax_a.set_ylim(-0.5, n_kinases - 0.5)
    ax_a.set_title(
        "A. Construct Unaligned",
        fontsize=cfg.panel_title_fontsize,
        fontweight=cfg.panel_title_fontweight,
        loc="left",
    )
    ax_a.set_xlabel("Residue position", fontsize=cfg.label_fontsize)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.spines["left"].set_visible(False)
    ax_a.spines["bottom"].set_visible(False)
    ax_a.tick_params(left=False, bottom=False, labelbottom=False)

    # =====================================================================
    # panel B: KLIFS region aligned (fixed length, dual palette)
    # =====================================================================
    ax_b = axes[1]
    aligned_len = len(kinase_seqs.loc[list_kinases[0], "seq_klifs_region_aligned"])

    for i, kinase in enumerate(list_kinases):
        seq_aligned = kinase_seqs.loc[kinase, "seq_klifs_region_aligned"]
        seq_klifs = kinase_seqs.loc[kinase, "seq_klifs_residues_only"]
        y = n_kinases - 1 - i

        colors = _map_aligned_to_klifs_colors(
            seq_aligned, seq_klifs, dict_aa_colors, klifs_pos_colors, cfg.gap_color
        )
        _draw_sequence_strip(ax_b, y, colors, cfg.rect_height)

    ax_b.set_yticks(range(n_kinases))
    ax_b.set_yticklabels(list_kinases[::-1], fontsize=cfg.label_fontsize)
    ax_b.set_xlim(-1, aligned_len + 1)
    ax_b.set_ylim(-0.5, n_kinases - 0.5)
    ax_b.set_title(
        "B. KLIFS Region Aligned",
        fontsize=cfg.panel_title_fontsize,
        fontweight=cfg.panel_title_fontweight,
        loc="left",
    )
    ax_b.set_xlabel(
        f"Aligned position (length = {aligned_len})", fontsize=cfg.label_fontsize
    )
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.spines["left"].set_visible(False)
    ax_b.spines["bottom"].set_visible(False)
    ax_b.tick_params(left=False, bottom=False, labelbottom=False)

    # =====================================================================
    # panel C: KLIFS residues only (85 positions, KLIFS palette)
    # =====================================================================
    ax_c = axes[2]
    klifs_len = len(kinase_seqs.loc[list_kinases[0], "seq_klifs_residues_only"])

    for i, kinase in enumerate(list_kinases):
        seq_klifs = kinase_seqs.loc[kinase, "seq_klifs_residues_only"]
        y = n_kinases - 1 - i

        colors = []
        for j, c in enumerate(seq_klifs):
            if c == "-":
                colors.append(cfg.gap_color)
            else:
                _, color = klifs_pos_colors[j]
                colors.append(color)
        _draw_sequence_strip(ax_c, y, colors, cfg.rect_height)

    ax_c.set_yticks(range(n_kinases))
    ax_c.set_yticklabels(list_kinases[::-1], fontsize=cfg.label_fontsize)
    ax_c.set_xlim(-1, klifs_len + 1)
    ax_c.set_ylim(-0.5, n_kinases - 0.5)
    ax_c.set_title(
        "C. KLIFS Residues Only",
        fontsize=cfg.panel_title_fontsize,
        fontweight=cfg.panel_title_fontweight,
        loc="left",
    )
    ax_c.set_xlabel(
        f"KLIFS position (length = {klifs_len})", fontsize=cfg.label_fontsize
    )
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.spines["left"].set_visible(False)
    ax_c.spines["bottom"].set_visible(False)
    ax_c.tick_params(left=False, bottom=False, labelbottom=False)

    # --- shared legend ---
    from mkt.databases.klifs import DICT_POCKET_KLIFS_REGIONS

    # build unique KLIFS region color legend entries
    seen_colors = {}
    for region, info in DICT_POCKET_KLIFS_REGIONS.items():
        color = info["color"]
        if color not in seen_colors:
            seen_colors[color] = region
        else:
            seen_colors[color] += f", {region}"

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=f"KLIFS: {label}")
        for color, label in seen_colors.items()
    ]
    # use a representative alphabet palette color for the MSA legend entry
    legend_handles.append(
        Patch(facecolor="#F0A3FF", edgecolor="none", label="MSA / amino acid")
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=cfg.label_fontsize - 2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    save_plot(
        fig,
        os.path.basename(output_path),
        "Sequence input schematic",
        bool_force_local=False,
        bool_image_subdir=False,
        output_path=os.path.dirname(output_path),
    )
