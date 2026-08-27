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


def column_counts(
    sequences: list[str],
    gap_chars: str = STR_GAP_CHARS,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Per-column (weighted) residue counts for a set of aligned sequences.

    The counts primitive underlying background-relative scoring: builds one
    length-20 count vector per alignment column (in :data:`STR_AA20` order) from
    any subset of aligned sequences, without instantiating a stateful analyzer.
    Gap/unknown characters in ``gap_chars`` (and any residue outside
    :data:`STR_AA20`) are excluded. Optional per-sequence ``weights`` (e.g.
    :func:`henikoff_weights`) generalize the raw counts to weighted counts;
    ``None`` reproduces the unweighted counts.

    Parameters
    ----------
    sequences : list[str]
        Aligned sequence strings of equal length.
    gap_chars : str
        Characters treated as gap/unknown and excluded (default: :data:`STR_GAP_CHARS`).
    weights : list[float] | None
        Optional per-sequence weights (same length as ``sequences``); ``None``
        weights every sequence equally (default: None).

    Returns
    -------
    np.ndarray
        ``(n_columns, 20)`` array of (weighted) residue counts in :data:`STR_AA20` order.
    """
    n = len(sequences)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    excluded = set(gap_chars)
    idx = np.array(
        [
            [DICT_AA20_INDEX.get(ch, -1) if ch not in excluded else -1 for ch in seq]
            for seq in sequences
        ]
    )
    n_cols = idx.shape[1] if n else 0
    out = np.zeros((n_cols, 20))
    for a_i in range(20):
        out[:, a_i] = ((idx == a_i) * w[:, None]).sum(axis=0)
    return out


def henikoff_weights(
    sequences: list[str],
    gap_chars: str = STR_GAP_CHARS,
) -> np.ndarray:
    """Henikoff & Henikoff (1994) position-based sequence weights.

    Down-weights redundant (near-duplicate) sequences so a few over-represented
    subfamilies do not dominate a column's composition -- the standard correction
    for phylogenetic non-independence in a large MSA. Each sequence's weight is
    ``sum_columns 1 / (k_col * n_{col, residue})`` over its non-gap positions,
    where ``k_col`` is the number of distinct residue types in the column and
    ``n_{col, residue}`` the count of this sequence's residue there. Returned
    unnormalized (the scale cancels in any weighted frequency).

    Parameters
    ----------
    sequences : list[str]
        Aligned sequence strings of equal length.
    gap_chars : str
        Characters treated as gap/unknown and skipped (default: :data:`STR_GAP_CHARS`).

    Returns
    -------
    np.ndarray
        Length-``len(sequences)`` array of unnormalized sequence weights.
    """
    weights = np.zeros(len(sequences))
    for column in zip(*sequences):
        counts: dict[str, int] = {}
        for residue in column:
            if residue in gap_chars:
                continue
            counts[residue] = counts.get(residue, 0) + 1
        k = len(counts)
        if k == 0:
            continue
        for i, residue in enumerate(column):
            if residue in gap_chars:
                continue
            weights[i] += 1.0 / (k * counts[residue])
    return weights


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

        Where ``q_a == 0`` the KL term is formally infinite, but such a zero only
        arises from an empirically estimated background that never observed that
        residue, so an infinite score would be a sampling artifact. Those residues
        are dropped and ``p`` is renormalized over the background's support, which
        keeps the result a KL between two distributions on that support -- and so
        still ``>= 0``. A background with no zeros (the default) is unaffected.

        Parameters
        ----------
        counts : np.ndarray
            Length-20 (weighted) residue counts ``n_a``.

        Returns
        -------
        float | None
            Information content in bits, or ``None`` for an empty column, or when
            the column and the background share no support.
        """
        p = self.smoothed_freqs(counts)
        if p is None:
            return None
        mask_bg = self.background > 0
        p_support, q_support = p[mask_bg], self.background[mask_bg]
        float_total = p_support.sum()
        if float_total <= 0:
            return None
        p_support = p_support / float_total
        mask = p_support > 0
        return float(
            np.sum(p_support[mask] * np.log2(p_support[mask] / q_support[mask]))
        )


def column_information_content(
    sequences: list[str],
    scorer: SubstitutionPseudocounts | None = None,
    gap_chars: str = STR_GAP_CHARS,
    weights: list[float] | None = None,
) -> list[float | None]:
    """Per-column background-relative information content (bits) for an MSA.

    Convenience wrapper mirroring :func:`column_conservation`: builds the weighted
    per-column counts (:func:`column_counts`) and scores each column with
    :meth:`SubstitutionPseudocounts.column_information`, so any subset of aligned
    sequences (a whole MSA, a clade, a bootstrap resample) is scored through the
    single shared implementation. Pass :func:`henikoff_weights` as ``weights`` to
    down-weight redundant subfamilies, and a ``scorer`` with a dataset-specific
    ``background`` for "surprising relative to this dataset".

    Parameters
    ----------
    sequences : list[str]
        Aligned sequence strings of equal length.
    scorer : SubstitutionPseudocounts | None
        Pseudocount/information scorer; a default :class:`SubstitutionPseudocounts`
        (BLOSUM62, Swiss-Prot background) is built when ``None``.
    gap_chars : str
        Characters treated as gap/unknown and excluded (default: :data:`STR_GAP_CHARS`).
    weights : list[float] | None
        Optional per-sequence weights (default: None -> unweighted).

    Returns
    -------
    list[float | None]
        One information-content value (bits) per column; ``None`` for an all-gap column.
    """
    if scorer is None:
        scorer = SubstitutionPseudocounts()
    counts = column_counts(sequences, gap_chars=gap_chars, weights=weights)
    return [scorer.column_information(column) for column in counts]


def consurf_grade_boundaries(
    scores: list[float | None],
    n_bins: int = 9,
) -> np.ndarray:
    """Interior quantile boundaries splitting ``scores`` into ``n_bins`` grades.

    The ConSurf grading is equal-frequency (quantile) binning of a continuous
    conservation score. This returns the ``n_bins - 1`` interior score cut points
    (ascending), so a caller can map a specific grade cutoff back to a score
    threshold (e.g. the score at the grade 6|7 boundary). ``None``/NaN scores are
    ignored when estimating the quantiles.

    Parameters
    ----------
    scores : list[float | None]
        Per-position conservation scores (any continuous scale).
    n_bins : int
        Number of grades (default 9, the ConSurf convention).

    Returns
    -------
    numpy.ndarray
        Ascending array of the ``n_bins - 1`` interior quantile boundaries;
        empty if no finite scores are supplied.
    """
    arr = np.array([np.nan if s is None else s for s in scores], dtype=float)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.empty(0)
    return np.quantile(vals, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])


def consurf_grades(
    scores: list[float | None],
    n_bins: int = 9,
    bool_ascending: bool = True,
) -> list[int | None]:
    """Assign ConSurf-style discrete conservation grades to continuous scores.

    Bins the finite scores into ``n_bins`` equal-frequency (quantile) grades -- the
    scheme ConSurf uses to turn a continuous conservation score into a small
    ordinal scale. With ``bool_ascending`` (higher score = more conserved, e.g.
    information content or percent identity) the most-conserved bin is grade
    ``n_bins`` and the most-variable is grade 1; pass ``bool_ascending=False`` for
    rate-type scores where lower = more conserved (e.g. Rate4Site). Grade
    ``n_bins`` is always the most conserved end regardless.

    Parameters
    ----------
    scores : list[float | None]
        Per-position conservation scores; ``None``/NaN entries get grade ``None``.
    n_bins : int
        Number of grades (default 9, the ConSurf convention).
    bool_ascending : bool
        Whether a higher score means more conserved (default True).

    Returns
    -------
    list[int | None]
        One grade in ``1..n_bins`` per input score (``None`` where the score was
        missing).
    """
    arr = np.array([np.nan if s is None else s for s in scores], dtype=float)
    finite = np.isfinite(arr)
    grades: list[int | None] = [None] * len(scores)
    if not finite.any():
        return grades

    edges = consurf_grade_boundaries(scores, n_bins=n_bins)
    # np.digitize -> ascending-in-score bin index 0..n_bins-1
    idx = np.digitize(arr[finite], edges, right=False)
    ranks = idx + 1 if bool_ascending else n_bins - idx

    j = 0
    for i in range(len(scores)):
        if finite[i]:
            grades[i] = int(ranks[j])
            j += 1
    return grades
