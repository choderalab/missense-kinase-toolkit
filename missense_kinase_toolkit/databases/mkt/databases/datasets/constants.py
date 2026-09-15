from enum import Enum


class KinaseGroupSource(Enum):
    """Enum for kinase groups."""

    kincore = "kincore.fasta.group"
    kinhub = "kinhub.group"
    klifs = "klifs.group"
    consensus = None


# TODO: implement this in the future
# class KinaseKDSequenceSource(Enum):
#     """Enum for kinase KD sequence sources."""

#     kincore = "kincore.fasta.kd_sequence"
#     klifs = "klifs.kd_sequence"
#     consensus = None
