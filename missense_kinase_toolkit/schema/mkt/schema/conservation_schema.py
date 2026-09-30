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

DICT_CLADE_ANCHORS: dict[str, tuple[str, ...]] = {
    "PDGFR/VEGFR": ("KDR", "KIT"),
    "SRC/ABL": ("SRC", "ABL1"),
    "INSR/ALK": ("INSR", "ALK"),
    "ERBB": ("EGFR", "ERBB2"),
    "EPH": ("EPHA3", "EPHB1"),
    "TRK/DDR": ("NTRK1", "DDR2"),
    "FGFR": ("FGFR2", "FGFR3"),
    "MET/AXL": ("MET", "AXL"),
    "RET/TIE": ("RET", "TEK"),
    "TEC": ("BTK", "TEC"),
}
"""Curated clade names keyed to a minimal set of *anchor* kinases that uniquely
identify each clade by membership (not by display-order number, which is fragile to
re-clustering). A display leaf is given a name only if it contains all of that
name's anchors, so a name silently disappears rather than mislabels if a clade
splits. The SYK/ZAP70 pair is deliberately omitted -- it clusters inside an EphA
subclade rather than forming a clean Syk clade, and is excluded from the clade
analysis."""


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

    def name_for_members(
        self,
        member_idx: list[int],
        dict_anchors: dict[str, tuple[str, ...]] = DICT_CLADE_ANCHORS,
    ) -> str | None:
        """Return the curated clade name for a set of member indices.

        A clade name applies only when the members contain *all* of that name's
        anchor kinases (see :data:`DICT_CLADE_ANCHORS`), so the mapping is robust to
        display-order renumbering and never mislabels a split clade.

        Parameters
        ----------
        member_idx : list[int]
            Member indices into :attr:`names` (e.g. a ``display_leaves`` entry's
            ``"members"``).
        dict_anchors : dict[str, tuple[str, ...]]
            Clade-name -> anchor-kinase mapping. Defaults to
            :data:`DICT_CLADE_ANCHORS`.

        Returns
        -------
        str | None
            The matching clade name, or None if no anchor set is fully contained.
        """
        members = {self.names[i] for i in member_idx}
        for name, anchors in dict_anchors.items():
            if all(anchor in members for anchor in anchors):
                return name
        return None

    def display_leaf_names(
        self, dict_anchors: dict[str, tuple[str, ...]] = DICT_CLADE_ANCHORS
    ) -> list[str | None]:
        """Return the curated clade name per display leaf (aligned to display_leaves).

        Parameters
        ----------
        dict_anchors : dict[str, tuple[str, ...]]
            Clade-name -> anchor-kinase mapping. Defaults to
            :data:`DICT_CLADE_ANCHORS`.

        Returns
        -------
        list[str | None]
            One name (or None) per entry of :attr:`display_leaves`, in that order.
        """
        return [
            self.name_for_members(leaf["members"], dict_anchors)
            for leaf in self.display_leaves
        ]

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
