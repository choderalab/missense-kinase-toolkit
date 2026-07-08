"""Position-specific residue statistics: background-relative information content with a
substitution-aware (Henikoff) pseudocount prior.

General-purpose MSA-column conservation scoring, independent of any particular alignment.
A column's information content is the KL divergence (in bits) of its smoothed residue
distribution against an amino-acid background, where the smoothing imputes unseen residues
toward the substitution neighbors of the residues actually observed (Henikoff & Henikoff
1996, as in PSI-BLAST PSSM construction). This credits conservative substitution (e.g.
K/R) and behaves sensibly at low effective-count columns, unlike a bare consensus-frequency
threshold or plain Shannon entropy.

:class:`SubstitutionPseudocounts` wraps a Biopython substitution matrix (default BLOSUM62)
and provides the prior only -- the conditional target probabilities, the per-column
pseudocount / smoothed frequencies, and the information content against a supplied
background. Estimating a dataset-specific background and analyzing particular columns is
the caller's responsibility.
"""

from dataclasses import dataclass, field

import numpy as np
from Bio.Align import substitution_matrices

STR_AA20 = "ARNDCQEGHILKMFPSTWYV"
"""str: The 20 standard amino acids, in the order used for all count/frequency vectors."""
DICT_AA20_INDEX = {aa: i for i, aa in enumerate(STR_AA20)}
"""dict[str, int]: Amino acid -> index into a length-20 vector (:data:`STR_AA20` order)."""

STR_GAP_CHARS = "-X"
"""str: Characters treated as gap/unknown and excluded from column conservation."""


def column_conservation(
    sequences: list[str],
    gap_chars: str = STR_GAP_CHARS,
    weights: list[float] | None = None,
) -> list[tuple[str | None, float]]:
    """Per-column consensus residue and its frequency for a set of aligned sequences.

    The column-level conservation primitive: any subset of aligned sequences (e.g. a
    whole MSA, or the members of a clade in a hierarchical tree) can be scored without
    instantiating a stateful analyzer. Gap/unknown characters in ``gap_chars`` are
    excluded; a column that is all gaps yields ``(None, 0.0)``.

    Optional per-sequence ``weights`` generalize the unweighted residue counts to a
    weighted consensus (e.g. Henikoff sequence weights to down-weight redundant
    subfamilies); ``None`` reproduces the unweighted counts exactly. Because the
    reported fraction is a ratio of summed weights it is invariant to the overall
    weight scale.

    Parameters
    ----------
    sequences : list[str]
        Aligned sequence strings of equal length.
    gap_chars : str
        Characters treated as gap/unknown and excluded from the consensus
        (default: :data:`STR_GAP_CHARS`).
    weights : list[float] | None
        Optional per-sequence weights (same length as ``sequences``). ``None``
        weights every sequence equally (default: None).

    Returns
    -------
    list[tuple[str | None, float]]
        One ``(consensus_aa, fraction)`` tuple per column, where ``fraction`` is
        the consensus residue's (weighted) frequency among the observed (non-gap)
        residues.
    """
    if weights is None:
        weights = [1.0] * len(sequences)

    out = []
    for column in zip(*sequences):
        weighted_counts: dict[str, float] = {}
        total = 0.0
        for residue, weight in zip(column, weights):
            if residue in gap_chars:
                continue
            weighted_counts[residue] = weighted_counts.get(residue, 0.0) + weight
            total += weight
        if not weighted_counts:
            out.append((None, 0.0))
            continue
        aa = max(weighted_counts, key=weighted_counts.get)
        out.append((aa, weighted_counts[aa] / total))
    return out


# Swiss-Prot average composition (HMMER's default null model), in STR_AA20 order.
ARR_SWISSPROT_BG = np.array(
    [
        0.0825,
        0.0553,
        0.0406,
        0.0546,
        0.0137,
        0.0393,
        0.0674,
        0.0708,
        0.0227,
        0.0596,
        0.0966,
        0.0585,
        0.0242,
        0.0386,
        0.0470,
        0.0657,
        0.0534,
        0.0108,
        0.0292,
        0.0687,
    ]
)
"""np.ndarray: Swiss-Prot background frequencies q_a; the default KL background."""
ARR_SWISSPROT_BG = ARR_SWISSPROT_BG / ARR_SWISSPROT_BG.sum()


@dataclass
class SubstitutionPseudocounts:
    """Substitution-aware pseudocount / information-content scorer for MSA columns.

    Parameters
    ----------
    matrix : str
        Biopython substitution matrix name (default ``"BLOSUM62"``); its implied
        background and conditional target probabilities ``P(a | b)`` are recovered from
        the matrix itself (no hardcoded frequency tables).
    beta : float
        Henikoff/PSI-BLAST pseudocount strength; the total pseudocount mass is
        ``beta * (distinct residue types observed)`` per column (default 10).
    background : np.ndarray | None
        KL background ``q_a`` (length-20, :data:`STR_AA20` order) for
        :meth:`column_information`; defaults to :data:`ARR_SWISSPROT_BG`. Supply a
        dataset-specific composition for "surprising relative to this dataset".
    """

    matrix: str = "BLOSUM62"
    beta: float = 10.0
    background: np.ndarray | None = field(default=None)

    def __post_init__(self):
        if self.background is None:
            self.background = ARR_SWISSPROT_BG
        self._target_probs = self._build_target_probs()

    def _build_target_probs(self) -> np.ndarray:
        """Conditional target probabilities ``P(a | b)`` recovered from the matrix.

        From the log-odds scores ``S_ab`` (half-bits) form the odds ``Q_ab = 2^(S_ab/2)``.
        The implied background ``f`` is the distribution whose joint ``f_a f_b Q_ab`` has
        marginal ``f`` -- i.e. ``Q f = 1`` -- and ``P(a | b) = f_a Q_ab``, column-normalized
        to absorb the log-odds rounding.
        """
        mat = substitution_matrices.load(self.matrix)
        scores = np.array([[float(mat[a, b]) for b in STR_AA20] for a in STR_AA20])
        odds = 2.0 ** (scores / 2.0)
        f = np.linalg.solve(odds, np.ones(len(STR_AA20)))
        f = np.clip(f, 1e-6, None)
        f /= f.sum()
        joint = f[:, None] * f[None, :] * odds
        return joint / joint.sum(axis=0, keepdims=True)

    @property
    def target_probs(self) -> np.ndarray:
        """``20 x 20`` conditional target probabilities ``P[a, b] = P(a | b)`` (columns sum to 1)."""
        return self._target_probs

    def pseudocounts(self, counts: np.ndarray) -> np.ndarray:
        """Substitution-aware pseudocount vector ``B * g_a`` for a column.

        ``g_a = sum_b (n_b / N) P(a | b)`` spreads each observed residue's mass onto its
        substitution neighbors; ``B = beta * (# distinct residue types observed)``.

        Parameters
        ----------
        counts : np.ndarray
            Length-20 (weighted) residue counts ``n_a`` in :data:`STR_AA20` order.

        Returns
        -------
        np.ndarray
            Length-20 pseudocount vector ``B * g_a`` (zeros for an empty column).
        """
        counts = np.asarray(counts, dtype=float)
        total = counts.sum()
        if total <= 0:
            return np.zeros_like(counts)
        mass = self.beta * int(np.count_nonzero(counts))
        return mass * (self._target_probs @ (counts / total))

    def smoothed_freqs(self, counts: np.ndarray) -> np.ndarray | None:
        """Posterior-mean residue frequencies ``p_a = (n_a + B g_a) / (N + B)``.

        Returns
        -------
        np.ndarray | None
            Length-20 smoothed distribution, or ``None`` for an empty column (``N = 0``).
        """
        counts = np.asarray(counts, dtype=float)
        total = counts.sum()
        if total <= 0:
            return None
        pseudo = self.pseudocounts(counts)
        return (counts + pseudo) / (total + pseudo.sum())

    def column_information(self, counts: np.ndarray) -> float | None:
        """Background-relative information content of a column, in bits.

        ``IC = sum_a p_a log2(p_a / q_a)`` (KL of the smoothed column against
        :attr:`background`); >= 0, higher = more conserved-and-surprising. An empty column
        returns ``None`` rather than scoring the bare pseudocount against the background.

        Parameters
        ----------
        counts : np.ndarray
            Length-20 (weighted) residue counts ``n_a``.

        Returns
        -------
        float | None
            Information content in bits, or ``None`` for an empty column.
        """
        p = self.smoothed_freqs(counts)
        if p is None:
            return None
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / self.background[mask])))
