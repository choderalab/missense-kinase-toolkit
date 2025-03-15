import os
import re

from Bio import SeqIO
from mkt.databases.aligners import Kincore2UniProtAligner
from mkt.databases.io_utils import get_repo_root


def extract_pk_fasta_info_as_dict(
    str_filename: str = "Human-PK.fasta",
) -> dict[str, dict[str, str | int]]:
    """Parse KinCore Human-PK.fasta file to extract information for KinaseInfo object.

    Parameters
    ----------
    str_filename : str, optional
        Filename of the fasta file, by default "Human-PK.fasta"

    Returns
    -------
    dict[str, dict[str, str | int]]
        Dictionary of {uniprot : {seq : str, start : int, end : int}}
    """
    fasta_sequences = SeqIO.parse(
        open(os.path.join(get_repo_root(), "data", str_filename)), "fasta"
    )
    list_description, list_seq = [], []
    for fasta in fasta_sequences:
        list_description.append(fasta.description)
        list_seq.append(str(fasta.seq))

    list_uniprot = [x.split(" ")[-1] for x in list_description]

    dict_out = {
        list_uniprot[i]: {
            "seq": list_seq[i],
        }
        for i in range(len(list_uniprot))
    }

    return dict_out


def align_kincore2uniprot(
    str_kincore: str,
    str_uniprot: str,
) -> dict[str, dict[str, str | int | list[int] | None]]:
    """Align KinCore Human-PK.fasta to canonical Uniprot sequences.

    Parameters
    ----------
    str_kicore : str
        KinCore sequence
    str_uniprot : str
        Uniprot sequence

    Returns
    -------
    dict[str, dict[str, str | None]]
        Dictionary of {start : int | None, end : int, mismatch : list[int]}
    """

    dict_out = dict.fromkeys(["seq", "start", "end", "mismatch"])
    dict_out["seq"] = str_kincore

    aligner = Kincore2UniProtAligner()
    alignments = aligner.align(str_kincore, str_uniprot)

    # if multiple alignments, return None
    if len(alignments) != 1:
        print(f"Multiple alignments found for {str_kincore} and {str_uniprot}")
        return dict_out

    alignment = alignments[0]

    # if alignment does not include full sequence, None
    if alignment.sequences[0] != alignment[0, :]:
        print(
            f"Alignment does not include full sequence \
              for {str_kincore} and {str_uniprot}"
        )
        pass

    start = int(alignment.aligned[1][0][0])
    dict_out["start"] = start + 1

    end = int(alignment.aligned[1][0][1])
    dict_out["end"] = end

    # if mismatch, provide idx of mismatch in KinCore sequence
    str_align = "".join(
        [
            i.split(" ")[-1]
            for idx, i in enumerate(str(alignment).split("\n"))
            if (idx + 1) % 2 == 0
        ]
    )
    str_align = re.sub(r"[a-zA-Z0-9]", "", str_align)
    if "." in str_align:
        dict_out["mismatch"] = [idx for idx, i in enumerate(str_align) if i == "."]

    return dict_out
