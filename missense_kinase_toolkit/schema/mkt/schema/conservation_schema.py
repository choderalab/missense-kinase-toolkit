"""Persisted output of the KLIFS hierarchical-conservation clustering.

Defines :class:`KLIFSConservationData`, a dependency-light (pure-Python, no
scipy/numpy) Pydantic artifact holding the pairwise distances, dendrogram, leaf
order, and display-leaf composition produced by
:class:`mkt.databases.conservation.KLIFSHierarchicalConservation`, plus the
provenance metadata describing how they were assembled. Serialized via
:func:`mkt.schema.io_utils.serialize_conservation_data` and shipped as package data
so downstream consumers can reuse the tree/distances without recomputing them (or
importing scipy).
"""

from pydantic import BaseModel, Field


class KLIFSConservationData(BaseModel):
    """Persisted distances + dendrogram from the KLIFS conservation clustering.

    The pairwise distances are stored as the condensed upper-triangle vector (the
    same row-major order SciPy's ``squareform`` uses), and the dendrogram as the
    SciPy linkage matrix (``N-1`` rows of ``[child_a, child_b, height, count]``),
    both as plain Python numbers so the schema stays free of numpy/scipy. The
    symmetric matrix and the ``(kin1, kin2, distance)`` long form are rebuilt on
    demand via :meth:`rebuild_square` and :meth:`to_long_form`; the full tree is
    reconstructed from :attr:`linkage_matrix` in ``mkt.databases`` (where scipy
    lives).
    """

    metric: str
    """Pairwise distance metric used (``"blosum"`` or ``"identity"``)."""
    blosum_name: str
    """Substitution matrix name used by the ``"blosum"`` metric."""
    linkage_method: str
    """SciPy linkage method (``"average"`` UPGMA or ``"complete"``)."""
    conservation_threshold: float
    """Minimum consensus-residue frequency for a column to count as conserved."""
    weighting: str
    """Per-node consensus weighting (``"none"`` or ``"henikoff"``)."""
    exclude_pseudokinases: bool
    """Whether predicted pseudokinases were dropped before clustering."""
    gap_chars: str
    """Characters treated as gap/unknown and excluded from scoring."""
    n_kinases: int
    """Number of kinases in the panel (``N``; matrix dimension)."""
    pocket_length: int
    """KLIFS pocket length (columns per kinase; 85 for the standard panel)."""

    names: list[str]
    """HGNC kinase names in panel order (distance-matrix row/column order)."""
    position_labels: list[str]
    """KLIFS region labels (e.g. ``"a.l:84"``) for the pocket columns."""

    distances_condensed: list[float]
    """Condensed upper-triangle distance vector (``N * (N - 1) / 2`` entries), in
    the row-major order used by ``scipy.spatial.distance.squareform``."""
    diagonal_is_similarity: bool = True
    """Documents :meth:`rebuild_square`'s default: rebuild as a similarity matrix
    (``1 - distance``, diagonal 1.0) rather than raw distance (diagonal 0.0)."""

    linkage_matrix: list[list[float]]
    """SciPy linkage matrix (``N-1`` rows of ``[child_a, child_b, height, count]``);
    fully reconstructs the dendrogram via ``scipy.cluster.hierarchy``."""
    leaves_order: list[int]
    """Leaf order (``scipy.cluster.hierarchy.leaves_list``), as indices into
    :attr:`names`, cached so consumers need not import scipy."""

    display_leaves: list[dict] = Field(default_factory=list)
    """Display-tree leaf rows from ``build_display_tree``: one entry per leaf as
    ``{"members": list[int], "kind": str}`` (indices into :attr:`names`)."""
    display_order: list[int] = Field(default_factory=list)
    """In-order (top-to-bottom) display order as indices into :attr:`display_leaves`."""

    def to_long_form(self) -> list[tuple[str, str, float]]:
        """Expand the condensed distances into ``(kin1, kin2, distance)`` triples.

        Returns
        -------
        list[tuple[str, str, float]]
            One triple per upper-triangle pair, using :attr:`names` for the labels.
        """
        triples: list[tuple[str, str, float]] = []
        idx = 0
        n = self.n_kinases
        for i in range(n):
            for j in range(i + 1, n):
                triples.append(
                    (self.names[i], self.names[j], self.distances_condensed[idx])
                )
                idx += 1
        return triples

    def rebuild_square(
        self,
        as_similarity: bool = True,
        reorder_by_leaves: bool = False,
    ) -> list[list[float]]:
        """Reconstruct the symmetric matrix from the condensed distance vector.

        Parameters
        ----------
        as_similarity : bool
            If True (default), return a similarity matrix (``1 - distance``) with a
            unit diagonal; if False, return raw distances with a zero diagonal.
        reorder_by_leaves : bool
            If True, reorder rows and columns by :attr:`leaves_order` (dendrogram
            leaf order) rather than the panel order.

        Returns
        -------
        list[list[float]]
            The symmetric ``N x N`` matrix as nested lists.
        """
        n = self.n_kinases
        diag = 1.0 if as_similarity else 0.0
        mat = [[diag] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.distances_condensed[idx]
                val = 1.0 - dist if as_similarity else dist
                mat[i][j] = val
                mat[j][i] = val
                idx += 1

        if reorder_by_leaves:
            order = self.leaves_order
            mat = [[mat[i][j] for j in order] for i in order]
        return mat
