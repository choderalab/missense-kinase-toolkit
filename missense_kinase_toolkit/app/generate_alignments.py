import numpy as np

# from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
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
        list_text = [i for s in self.list_sequences for i in s]
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
        if N > 100:
            viewlen = 100
        else:
            viewlen = N

        # sequence text view with ability to scroll along x axis
        # view_range is for the close up view
        view_range = (0, viewlen)
        plot_height = 50
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
        p1.xaxis.major_label_text_font_style = "bold"
        p1.yaxis.minor_tick_line_width = 0
        p1.yaxis.major_tick_line_width = 0
        p1.yaxis.axis_label_text_color = "black"
        p1.yaxis.major_label_text_color = "black"
        p1.xaxis.axis_label_text_color = "black"
        p1.xaxis.major_label_text_color = "black"

        self.plot = p1


# from mkt.databases.utils import rgetattr

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
    "KLIFS": {
        "seq": "KLIFS2UniProtIdx",
        "start": lambda x: min(x.values()),
        "end": lambda x: max(x.values()),
    },
}

# recursively check "seq" split on "."

# class SequenceAlignment:
#     def __init__(
#         self,
#         obj_kinaseinfo: KinaseInfo,
#         dict_colors: dict[str, str],
#         font_size: int = 9,
#         plot_width: int = 800,
#     ):
#         self.kinase_info = obj_kinaseinfo
#         self.dict_colors = dict_colors
#         self.font_size = font_size
#         self.plot_width = plot_width
#         self.list_sequences = []
#         self.list_ids = []
#         self.generate_alignment()
#         self.plot = None
#         self.plot_alignment()

#     def _map_single_alignment(
#         idx_start: int,
#         idx_end: int,
#         str_uniprot: str,
#         str_in: str | None = None,
#     ):
#         """Map the indices of the alignment to the original sequence.

#         Parameters
#         ----------
#         idx_start : int
#             Start index of the alignment.
#         idx_end : int
#             End index of the alignment.
#         str_in : str | None
#             Sequence provided by
#         str_uniprot : str
#             Full canonical UniProt sequence.

#         Returns
#         -------
#         str
#             Output string with the alignment mapped to the original sequence.

#         """
#         n_before, n_after = idx_start - 1, len(uniprot_seq) - idx_end

#         # use UniProt canonical sequence if no sequence provided (Pfam)
#         if str_in is None:
#             str_out = "".join(
#                 [
#                     str_uniprot[i-1] if i in range(idx_start, idx_end + 1) \
#                     else "-" for i in range(1, len(str_uniprot)+1)
#                 ]
#             )

#         # use
#         else:
#             #TODO
#             pass

#         return str_out

#     def generate_alignment(self):
#         """Generate the alignment."""
#         for key, value in self.dict_in.items():
#             if value is not None:
#                 seq = rgetattr(value, DICT_ALIGNMENT[key]["seq"])
#                 start = rgetattr(value, DICT_ALIGNMENT[key]["start"])
#                 end = rgetattr(value, DICT_ALIGNMENT[key]["end"])
#                 self.list_sequences.append(seq)
#                 self.list_ids.append(key)

#         def generate_alignments(
#             obj_in: KinaseInfo,
#             dict_col: dict[str, str],
#         ) -> dict[str, str]:
#             """Generate sequence alignment plot.

#             Returns
#             -------
#             obj_in : KinaseInfo
#                 KinaseInfo object from dict_kinase.
#             dashboard_state : DashboardState
#                 The state of the dashboard containing the selected kinase and color palette.

#             """
#             list_keys = [
#                 "UniProt",
#                 "KinCore, FASTA",
#                 "KinCore, CIF",
#                 "Pfam",
#                 "KLIFS",
#             ]

#             dict_out = {
#                 "str_seq": dict.fromkeys(list_keys),
#                 "list_col": dict.fromkeys(list_keys),
#             }

#             #TODO: if obj_in.hgnc_name == "CDKL1"l; adjust KinCore sequences

#             # UniProt
#             key = "UniProt"
#             dict_out["str_seq"][key] = obj_in.uniprot.canonical_seq
#             dict_out["list_col"][key] = [dict_col[i] for i in uniprot_seq]
#             # Pfam
#             key = "Pfam"
#             if obj_in.pfam is not None:
#                 dict_out["str_seq"][key] = self._map_single_alignment(
#                     obj_temp.pfam.start,
#                     obj_temp.pfam.end,
#                     uniprot_seq
#                 )
#                 dict_out["list_col"][key] = [dict_col[i] for i in dict_out["str_seq"][key]]
#             # KinCore FASTA
#             key = "KinCore, FASTA"
#             if obj_in.kincore is not None:
#                 dict_out["str_seq"][key] = self._map_single_alignment(
#                     obj_temp.kincore.fasta.start,
#                     obj_temp.kincore.fasta.end,
#                     uniprot_seq,
#                 )
#                 list_kincore_col = [dict_col[i] for i in uniprot_seq]
#                 # colors
#                 dict_out["list_col"][key] = [dict_col[i] for i in dict_out["str_seq"][key]]
#             # KinCore CIF
#             key = "KinCore, CIF"
#             if obj_in.kincore is not None:
#                 if obj_in.kincore.cif is not None:
#                     dict_out["str_seq"][key] = self._map_single_alignment(
#                         obj_temp.kincore.fasta.start,
#                         obj_temp.kincore.fasta.end,
#                         uniprot_seq,
#                     )
#                     dict_out["list_col"][key] = [dict_col[i] for i in dict_out["str_seq"][key]]
#             # KLIFS
#             dict_klifs = obj_in.KLIFS2UniProtIdx
#             if dict_klifs is not None:
#                 idx_klifs_min, idx_klifs_max = min(dict_klifs.values()), max(dict_klifs.values())
#                 n_before, n_after = idx_klifs_min - 1, len(uniprot_seq) - idx_klifs_max
#                 # sequence
#                 list_klifs_seq = [
#                     uniprot_seq[i-1] if i in dict_klifs.values() else "-" \
#                     for i in range(idx_klifs_min, idx_klifs_max + 1)
#                 ]
#                 dict_out["str_seq"]["KLIFS"] = "".join(
#                     ["-" * n_before] + list_klifs_seq + ["-" * n_after]
#                 )
#                 # colors
#                 list_klifs_col = [
#                     DICT_POCKET_KLIFS_REGIONS[dict_klifs_rev[i].split(":")[0]]["color"] if i in \
#                     dict_klifs_rev else dict_col["-"] for i in range(idx_klifs_min, idx_klifs_max + 1)
#                 ]
#                 dict_out["list_col"]["KLIFS"] = \
#                     [dict_col["-"]] * n_before + list_color + [dict_col["-"]] * n_after
#             return dict_out
