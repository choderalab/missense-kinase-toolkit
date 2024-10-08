from dataclasses import dataclass

from Bio import Align

@dataclass
class CustomAligner():
    mode: str = "local"
    """str: Alignment mode. Default is "local"."""
    substitution_matrix: str = "BLOSUM62"
    """str: Substitution matrix. Default is BLOSUM62."""
    gap_score: int = -5
    """int: Gap score. Default is -5."""
    extend_gap_score: int = -1
    """int: Gap extension score. Default is -1."""

    def __post_init__(self):
        self.aligner = Align.PairwiseAligner()
        self.aligner.mode = self.mode
        self.aligner.substitution_matrix = Align.substitution_matrices.load(self.substitution_matrix)
        self.aligner.open_gap_score = self.gap_score
        self.aligner.extend_gap_score = self.extend_gap_score


    def align(self, seq1: str, seq2: str) -> Align.MultipleSeqAlignment:
        return self.aligner.align(seq1, seq2)
    

@dataclass
class BL2UniProtAligner(CustomAligner):
    mode: str = "global"
    """str: Alignment mode. Default is "global."""

    def __post_init__(self):
        super().__post_init__()


@dataclass
class Kincore2UniProtAligner(CustomAligner):
    mode: str = "local"
    """str: Alignment mode. Default is "local."""
    
    def __post_init__(self):
        super().__post_init__()