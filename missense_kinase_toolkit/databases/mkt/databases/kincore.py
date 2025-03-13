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


# # NOT IN USE - USED TO GENERATE ABOVE

# import pandas as pd

# from mkt.databases import kinase_schema

# # generate these in databases.ipynb
# df_kinhub = pd.read_csv("../data/kinhub.csv")
# df_klifs = pd.read_csv("../data/kinhub_klifs.csv")
# df_uniprot = pd.read_csv("../data/kinhub_uniprot.csv")
# df_pfam = pd.read_csv("../data/kinhub_pfam.csv")

# df_merge = kinase_schema.concatenate_source_dataframe(
#     df_kinhub,
#     df_uniprot,
#     df_klifs,
#     df_pfam
# )

# dict_kin = kinase_schema.create_kinase_models_from_df(df_merge)

# dict_kincore = {key: val for key, val in dict_kin.items() if val.KinCore is not None}

# aligner = kincore2uniprot_aligner()
# dict_kincore_alignments = {val.hgnc_name: aligner.align(val.KinCore.seq, val.UniProt.canonical_seq) \
#                            for key, val in dict_kincore.items()}

# dict_kincore_idx = {}
# for hgnc, alignments in dict_kincore_alignments.items():
#     dict_temp = dict.fromkeys(["start", "end", "mismatch"])
#     dict_kincore_idx[hgnc] = dict_temp
#     # if multiple alignments, None
#     if len(alignments) != 1:
#         pass
#     for alignment in alignments:
#         # if alignment does not include full sequence, None
#         if alignment.sequences[0] != alignment[0, :]:
#             pass
#         start = int(alignment.aligned[1][0][0])
#         end = int(alignment.aligned[1][0][1])
#         str_align = "".join([i.split(" ")[-1] for idx, i in \
#                              enumerate(str(alignment).split("\n")) if (idx+1) % 2 == 0])
#         str_align = re.sub(r"[a-zA-Z0-9]", "", str_align)
#         if "." in str_align:
#             dict_kincore_idx[hgnc]["mismatch"] = \
#             [idx for idx, i in enumerate(str_align) if i == "."]
#         dict_kincore_idx[hgnc]["start"] = start + 1
#         dict_kincore_idx[hgnc]["end"] = end

# name = "CDKL1" # only apparent mismatch
# idx = dict_kincore_idx[name]["mismatch"][0] # 148
# print(dict_kin[name].KinCore.seq[idx]) # A
# print(dict_kin[name].UniProt.canonical_seq[idx + dict_kincore_idx[name]["start"] - 1]) # T

# print(dict_kincore_alignments["CDKL1"][0]) # alignment object - see mismatch at 148
