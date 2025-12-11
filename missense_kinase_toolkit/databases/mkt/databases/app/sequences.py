import logging
from typing import Any

from mkt.databases.klifs import DICT_POCKET_KLIFS_REGIONS
from mkt.schema.kinase_schema import KinaseInfo
from mkt.schema.utils import rgetattr
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
    "Phosphosites": {
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
    """Class to generate sequence alignments for kinase sequences."""

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

    def __post_init__(self):
        # generate and save alignments
        self.dict_align = self.generate_alignments()
        self.list_sequences = [v["str_seq"] for v in self.dict_align.values()]
        self.list_ids = list(self.dict_align.keys())
        self.list_colors = [v["list_colors"] for v in self.dict_align.values()]

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
                output = rgetattr(self.obj_kinase, start_or_end)
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
            seq = rgetattr(self.obj_kinase, value["seq"])

            # KinCore CIF sequence needs to be extracted from dict and have linebreaks removed
            if key == "KinCore, CIF" and seq is not None:
                seq = seq["_entity_poly.pdbx_seq_one_letter_code"][0].replace("\n", "")

            # CDKL1 KinCore FASTA and CIF have an extra M at the start - remove and add back
            if self.obj_kinase.hgnc_name == "CDKL1" and key.startswith("KinCore"):
                seq = seq[1:]

            start = self._parse_start_end_values(value["start"], seq)
            end = self._parse_start_end_values(value["end"], seq)

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
