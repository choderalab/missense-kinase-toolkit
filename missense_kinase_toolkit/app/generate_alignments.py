import logging
from itertools import chain
from typing import Any

import numpy as np
from bokeh.models import (
    ColumnDataSource,
    CustomJSTickFormatter,
    FixedTicker,
    Label,
)
from bokeh.models.glyphs import Rect, Text
from bokeh.plotting import figure
from mkt.databases.klifs import DICT_POCKET_KLIFS_REGIONS
from mkt.databases.utils import try_except_return_none_rgetattr
from mkt.schema.kinase_schema import KinaseInfo
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


DICT_ALIGNMENT = {
    "UniProt": {
        "seq": "uniprot.canonical_seq",
        "start": 1,
        "end": lambda x: len(x),
    },
    "Pfam": {
        "seq": None,
        "start": "pfam.start",
        "end": "pfam.end",
    },
    "KinCore, FASTA": {
        "seq": "kincore.fasta.seq",
        "start": "kincore.fasta.start",
        "end": "kincore.fasta.end",
    },
    "KinCore, CIF": {
        "seq": "kincore.cif.cif",  # need to get from dict "_entity_poly.pdbx_seq_one_letter_code"
        "start": "kincore.cif.start",
        "end": "kincore.cif.end",
    },
    "Phosphosite": {
        "seq": "uniprot.phospho_sites",
        "start": None,
        "end": None,
    },
    "KLIFS": {
        "seq": "KLIFS2UniProtIdx",
        "start": lambda x: min([x for x in x.values() if x is not None]),
        "end": lambda x: max([x for x in x.values() if x is not None]),
    },
}


@dataclass
class SequenceAlignment:

    obj_kinase: KinaseInfo
    """KinaseInfo object from which to extract sequences."""
    dict_color: dict[str, str]
    """Color dictionary for sequence viewer."""
    bool_mismatch: bool = True
    """If True, show mismatches with UniProt seq in crimson, by default True."""
    bool_klifs: bool = True
    """If True, shade KLIFS pocket residues using KLIFS pocket colors, by default True."""
    bool_reverse: bool = True
    """Whether or not to reverse order of inputs"""
    font_size: int = 9
    """Font size for alignment."""
    plot_width: int = 800
    """Width of the plot."""

    def __post_init__(self):
        # generate and save alignments
        dict_align = self.generate_alignments()
        self.list_sequences = [v["str_seq"] for v in dict_align.values()]
        self.list_ids = list(dict_align.keys())
        self.list_colors = [v["list_colors"] for v in dict_align.values()]

        # reverse alignment entries, if desired
        if self.bool_reverse:
            self.list_sequences = self.list_sequences[::-1]
            self.list_ids = self.list_ids[::-1]
            self.list_colors = self.list_colors[::-1]

        # generate plot to render
        self.plot = self.generate_plot()

    @staticmethod
    def _map_single_alignment(
        idx_start: int,
        idx_end: int,
        str_uniprot: str,
        seq_obj: str | None = None,
    ):
        """Map the indices of the alignment to the original sequence.

        Parameters
        ----------
        idx_start : int
            Start index of the alignment.
        idx_end : int
            End index of the alignment.
        str_uniprot : str
            Full canonical UniProt sequence.
        seq_obj : str | None
            Seq obj provided by user or database. If None, use UniProt sequence.
            This is the case for Pfam, which only provides start and end indices.

        Returns
        -------
        str
            Output string with the alignment mapped to the original sequence.

        """
        # if either idx is None, return string of dashes length of UniProt
        if any([i is None for i in [idx_start, idx_end]]):
            str_out = "-" * len(str_uniprot)
            # UniProt phosphosites
            if isinstance(seq_obj, list):
                str_out = "".join(
                    [
                        str_uniprot[idx] if idx + 1 in seq_obj else i
                        for idx, i in enumerate(str_out)
                    ]
                )
            return str_out

        n_before, n_after = idx_start - 1, len(str_uniprot) - idx_end
        # KLIFS
        if isinstance(seq_obj, dict):
            list_seq = [
                str_uniprot[i - 1] if i in seq_obj.values() else "-"
                for i in range(idx_start, idx_end + 1)
            ]
            str_out = "-" * n_before + "".join(list_seq) + "-" * n_after
        # Pfam just provides start and end indices - extract from UniProt
        elif seq_obj is None:
            str_out = "".join(
                [
                    str_uniprot[i - 1] if i in range(idx_start, idx_end + 1) else "-"
                    for i in range(1, len(str_uniprot) + 1)
                ]
            )
        # all others provide sequences as a contiguous string
        else:
            str_out = "-" * n_before + seq_obj + "-" * n_after

        return str_out

    def _parse_start_end_values(self, start_or_end: Any, str_seq: str) -> int | None:
        """Parse the start and end keys for the alignment.

        Parameters
        ----------
        start_or_end : Any
            The start or end key to parse.
        str_seq : str
            The sequence to parse - only used if start_or_end callable.

        Returns
        -------
        int | None
            The parsed start or end index, or None if not found.
        """
        try:
            if isinstance(start_or_end, str):
                output = try_except_return_none_rgetattr(self.obj_kinase, start_or_end)
            elif isinstance(start_or_end, int):
                output = start_or_end
            elif callable(start_or_end):
                output = start_or_end(str_seq)
            else:
                logger.error(
                    f"Start or end value {start_or_end} "
                    "is not a string, int, or callable "
                    "and cannot be parsed. Returning None..."
                )
                output = None
        except Exception as e:
            logger.error(f"Error parsing start/end values: {e}. Returning None...")
            output = None
        return output

    def generate_alignments(
        self,
    ) -> dict[str, str | list[str]]:
        """Iterate through the KinaseInfo object and generate a list of sequences and a list of colors.

        Returns
        -------
        dict[str, str | list[str]]
            A dictionary with keys DICT_ALIGNMENT containing the
            sequences and a list of colors per residue.
        """
        uniprot_seq = self.obj_kinase.uniprot.canonical_seq

        dict_out = {
            k: dict.fromkeys(["str_seq", "list_colors"]) for k in DICT_ALIGNMENT.keys()
        }

        for key, value in DICT_ALIGNMENT.items():
            seq = try_except_return_none_rgetattr(self.obj_kinase, value["seq"])

            # KinCore CIF sequence needs to be extracted from dict and have linebreaks removed
            if key == "KinCore, CIF" and seq is not None:
                seq = seq["_entity_poly.pdbx_seq_one_letter_code"][0].replace("\n", "")

            # CDKL1 KinCore FASTA and CIF have an extra M at the start - remove and add back
            if self.obj_kinase.hgnc_name == "CDKL1" and key.startswith("KinCore"):
                seq = seq[1:]

            start = self._parse_start_end_values(value["start"], self.obj_kinase, seq)
            end = self._parse_start_end_values(value["end"], self.obj_kinase, seq)

            seq_out = self._map_single_alignment(start, end, uniprot_seq, seq)
            # CDKL1 KinCore FASTA and CIF have an extra M at the start
            # add back and add "-" for all other sequences
            if self.obj_kinase.hgnc_name == "CDKL1":
                if key.startswith("KinCore"):
                    seq_out = "M" + seq_out
                else:
                    seq_out = "-" + seq_out
            dict_out[key]["str_seq"] = seq_out

            if key == "KLIFS" and seq is not None and self.bool_klifs:
                seq_rev = {v: k for k, v in seq.items()}
                list_cols = [
                    (
                        DICT_POCKET_KLIFS_REGIONS[seq_rev[i].split(":")[0]]["color"]
                        if i in seq_rev
                        else self.dict_color["-"]
                    )
                    for i in range(start, end + 1)
                ]
                n_before, n_after = start - 1, len(uniprot_seq) - end
                list_cols = (
                    [self.dict_color["-"]]
                    * (n_before + 1 * (self.obj_kinase.hgnc_name == "CDKL1"))
                    + list_cols
                    + [self.dict_color["-"]] * n_after
                )
            else:
                list_cols = [self.dict_color[i] for i in seq_out]
            dict_out[key]["list_colors"] = list_cols

        if self.obj_kinase.hgnc_name == "CDKL1":
            uniprot_seq = "-" + uniprot_seq
        if self.bool_mismatch:
            for key, value in dict_out.items():
                for idx, (ref_seq, map_seq) in enumerate(
                    zip(uniprot_seq, value["str_seq"])
                ):
                    if ref_seq != map_seq and map_seq != "-":
                        # Claude proposed crimson
                        value["list_colors"][idx] = "#DC143C"

        return dict_out

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

        if N > 100:
            viewlen = 100
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
        view_range = (0, viewlen)
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
        p1.xaxis.formatter = CustomJSTickFormatter(code="return String(tick)")

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

        # TODO: Add second axis for KLIFS pocket

        p1.yaxis.minor_tick_line_width = 0
        p1.yaxis.major_tick_line_width = 0
        p1.yaxis.axis_label_text_color = "black"
        p1.yaxis.major_label_text_color = "black"
        p1.xaxis.axis_label_text_color = "black"
        p1.xaxis.major_label_text_color = "black"

        return p1
