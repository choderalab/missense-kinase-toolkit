from dataclasses import dataclass
from abc import ABC, abstractmethod


class CustomAligner(ABC):
    """Custom aligner class for aligning sequences."""

    substitution_matrix: str = "BLOSUM62"
    """str: Substitution matrix used. Default is BLOSUM62."""

    @abstractmethod
    def align(self, *args, **kwargs):
        """Abstract method for aligning sequences."""
        ...


@dataclass
class ClustalOmegaAligner(CustomAligner):
    """ClustalOmega aligner class for multiple sequence alignments (need to initialize with list of sequences)."""

    list_sequences: list[str]
    """list[str]: List of sequences to align."""
    path_bin: str = "/usr/local/bin/clustalo"
    """str: Path to clustalo binary. Default is "/usr/local/bin/clustalo"."""
    
    def __post_init__(self):
        from biotite.sequence import ProteinSequence, align

        self.alphabet = ProteinSequence.alphabet
        self.matrix_substitution = align.SubstitutionMatrix(
            self.alphabet, 
            self.alphabet, 
            self.substitution_matrix
        )
        self.list_sequences = [ProteinSequence(seq) for seq in self.list_sequences]
        self.align()


    def align(self) -> str:
        from biotite.application import clustalo

        app = clustalo.ClustalOmegaApp(
            self.list_sequences, 
            self.path_bin, 
            self.matrix_substitution
        )

        app.start()
        app.join()
        self.alignments = app.get_alignment()
        self.list_alignments = self.alignments.get_gapped_sequences()


@dataclass
class BioAligner(CustomAligner):
    """BioPython aligner class for aligning sequences. Initialized without sequences"""
    from Bio import Align

    mode: str = "local"
    """str: Alignment mode. Default is "local"."""
    gap_score: int = -5
    """int: Gap score. Default is -5."""
    extend_gap_score: int = -1
    """int: Gap extension score. Default is -1."""

    def __post_init__(self):
        from Bio import Align
    
        self.aligner = Align.PairwiseAligner()
        self.aligner.mode = self.mode
        self.aligner.substitution_matrix = Align.substitution_matrices.load(
            self.substitution_matrix
        )
        self.aligner.open_gap_score = self.gap_score
        self.aligner.extend_gap_score = self.extend_gap_score

    def align(self, seq1: str, seq2: str) -> Align.MultipleSeqAlignment:
        return self.aligner.align(seq1, seq2)

@dataclass
class BL2UniProtAligner(BioAligner):
    mode: str = "global"
    """str: Alignment mode. Default is "global."""

    def __post_init__(self):
        super().__post_init__()


@dataclass
class Kincore2UniProtAligner(BioAligner):
    mode: str = "local"
    """str: Alignment mode. Default is "local."""

    def __post_init__(self):
        super().__post_init__()
