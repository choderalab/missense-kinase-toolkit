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
