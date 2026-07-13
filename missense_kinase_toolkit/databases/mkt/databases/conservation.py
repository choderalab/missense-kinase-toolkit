"""Study-independent KLIFS hierarchical conservation of the human kinome.

Hierarchically clusters the human kinome on KLIFS pocket similarity (from the
``DICT_KINASE`` KLIFS pocket sequences) and traces per-residue conservation up
the tree via the shared engine :class:`KLIFSHierarchicalConservation`. Two
renderers sit on top of that engine: a static matplotlib supplemental figure
(:class:`KLIFSConservationTreeFigure`) and an interactive standalone-HTML Bokeh
explorer (:class:`KLIFSTreeConservationApp`), which therefore cannot diverge on
tree topology.

Moved out of ``mkt_impact`` into ``mkt.databases`` because it needs no
cohort/study data -- it operates purely on the human-kinome KLIFS panel.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from bokeh.embed import file_html
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    Select,
    TapTool,
)
from bokeh.plotting import figure
from bokeh.resources import INLINE
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from mkt.databases.colors import (
    AA_MAPPING,
    CMAP_CRITICAL_DEPTH,
    COLOR_CONSERVATION_PRIMARY,
    COLOR_DARK_TEXT,
    COLOR_DEPTH_NA,
    COLOR_LOGO_BAR,
    COLOR_LOGO_SUBTHRESHOLD,
    COLOR_TREE_BREAK,
    COLOR_TREE_FALLBACK,
    COLOR_TREE_GUIDE,
    COLOR_TREE_MIXED_FAMILY,
    COLOR_TREE_NO_BREAK,
    COLOR_TREE_PSEUDO,
    COLOR_TREE_SPLIT_MARKER,
    COLOR_TREE_TABLE_GAP,
    COLOR_TREE_TABLE_NO_CONSENSUS,
    DICT_COLORS,
    AminoAcidPalette,
    map_single_letter_aa_to_color,
    readable_text_color,
)
from mkt.databases.klifs import DICT_POCKET_KLIFS_REGIONS, LIST_KLIFS_REGION
from mkt.databases.plot import remove_spines
from mkt.databases.pssm import (
    DICT_AA20_INDEX,
    STR_GAP_CHARS,
    SubstitutionPseudocounts,
    column_conservation,
)
from mkt.schema.conservation_schema import KLIFSConservationData
from mkt.schema.constants import DICT_KINASE_GROUP_COLORS
from mkt.schema.io_utils import (
    deserialize_kinase_dict,
    load_conservation_data,
    save_plot,
)
from mkt.schema.utils import group_name_homologs, rgetattr
from pydantic import BaseModel, Field
from scipy.cluster.hierarchy import (
    ClusterNode,
    cophenet,
    fcluster,
    leaves_list,
    linkage,
    to_tree,
)
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

logger = logging.getLogger(__name__)


def _repair_camkk1_pocket(dict_kinase: dict) -> None:
    """Temporary shim: fix CAMKK1's off-by-one KLIFS mapping in place.

    The cached ``DICT_KINASE`` carries a mis-curated CAMKK1 (UniProt Q8N5S9) KLIFS
    mapping. These indices are read 1-based (residue ``== canonical_seq[idx - 1]``), and
    every CAMKK1 entry is one too low -- so each position lands on the residue *before*
    the intended one (e.g. I:1 -> Q instead of S, and the catalytic columns onto
    non-catalytic residues). The off-by-one is uniform, so we:

    - ``KLIFS2UniProtIdx``: add 1 to every non-None entry. This realigns *all* positions,
      including the residues that flank the large alpha-C/beta-4 insert (II:13, IV:41,
      ...) -- a "shift each value to the preceding key" shortcut would mis-map exactly
      those, since the indices jump across the insert.
    - ``pocket_seq``: rebuild from the corrected indices (``canonical_seq[idx - 1]``),
      restoring the catalytic III:17=K / c.l:70=D / xDFG:81=D.
    - ``KLIFS2UniProtSeq`` (keyed by full region, incl. inter-region inserts absent from
      the 85-position index): shift the gathered residue stream left by one and append
      the newly revealed trailing residue, skipping None regions.

    Stopgap until the kinase dict is regenerated with the upstream root-cause fix; the
    ``mkt.databases`` ``change_wrong_klifs_pocket_seq`` override currently patches only
    ``pocket_seq`` upstream, so the regenerated indices will still need the root fix.

    Parameters
    ----------
    dict_kinase : dict
        The deserialized ``DICT_KINASE`` mapping, modified in place.

    Returns
    -------
    None
    """
    info = dict_kinase.get("CAMKK1")
    if (
        info is None
        or info.klifs is None
        or info.uniprot is None
        or not info.KLIFS2UniProtIdx
    ):
        return

    from mkt.schema.constants import LIST_KLIFS_REGION

    useq = info.uniprot.canonical_seq
    old_idx = info.KLIFS2UniProtIdx

    # corrected indices: every non-None entry is one too low under the 1-based convention
    new_idx = {k: (v + 1 if v is not None else None) for k, v in old_idx.items()}

    try:
        # the residue newly revealed at the (now corrected) C-terminal a.l:85 position
        trailing = useq[new_idx["a.l:85"] - 1]
        pocket = "".join(
            useq[new_idx[label] - 1] if new_idx.get(label) is not None else "-"
            for label in LIST_KLIFS_REGION
        )
    except (KeyError, IndexError):
        logger.warning("could not repair CAMKK1 mapping from KLIFS2UniProtIdx")
        return

    if len(pocket) != len(LIST_KLIFS_REGION):
        return

    # fix KLIFS2UniProtSeq by shifting its gathered residue stream left by one
    seq = info.KLIFS2UniProtSeq
    if seq is not None:
        stream = "".join(v for v in seq.values() if v is not None)
        new_stream = stream[1:] + trailing
        pos = 0
        for region, residues in seq.items():
            if residues is None:
                continue
            seq[region] = new_stream[pos : pos + len(residues)]
            pos += len(residues)

    info.KLIFS2UniProtIdx = new_idx
    info.klifs.pocket_seq = pocket


# mirrors mkt_impact's temporary CAMKK1 pocket shim; centralize/remove once the kinase dict is regenerated.
DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE")
_repair_camkk1_pocket(DICT_KINASE)

INT_KLIFS_POCKET_LENGTH = 85
"""int: Expected length of a valid KLIFS pocket alignment string."""
FLOAT_CONSERVATION_THRESHOLD = 0.80
"""float: Minimum consensus-residue frequency for a column to count as conserved at a node."""
INT_MIN_CHILD_MEMBERS = 2
"""int: Minimum child-clade size considered in survives-up pruning; smaller (e.g. singleton
outlier) children are ignored so they cannot veto an otherwise-robust residue."""
INT_TREE_MIN_CLUSTER_DISPLAY = 12
"""int: Default display threshold for :meth:`KLIFSHierarchicalConservation.build_display_tree`
-- gathered subtrees with fewer members collapse to a single leaf bar rather than expanding
into their own split node. Shared by the static figure and the interactive Bokeh app."""
# --- static conservation-tree figure (KLIFSConservationTreeFigure) ---
STR_FILE_CONSERVATION_TREE = "klifs_conservation_tree"
"""str: Output basename for the static KLIFS conservation-tree figure."""
TUP_TREE_PAGE = (8.5, 11.0)
"""tuple[float, float]: Figure size in inches (US letter, portrait)."""
FLOAT_TREE_FONT_SIZE = 4.0
"""float: Base font size for the leaf name boxes, table cells, and KLIFS labels."""
FLOAT_TREE_ROW_HEIGHT = 0.72
"""float: Height of each leaf row / table cell in y (row) units."""
FLOAT_TREE_BOX_PAD_IN = 0.028
"""float: Inner horizontal padding per side of each name box (inches)."""
FLOAT_TREE_BOX_GAP_IN = 0.03
"""float: Horizontal gap between adjacent name boxes (inches)."""
FLOAT_TREE_TABLE_GAP_IN = 0.07
"""float: Gap between the name boxes and the conservation table (inches)."""
FLOAT_TREE_RIGHT_PAD_IN = 0.05
"""float: Right margin reserved past the table (inches)."""
FLOAT_TREE_DEPTH_STEP_IN = 0.014
"""float: Dendrogram depth step (inches); keeps the tree a thin structural stub."""
FLOAT_TREE_DOMINANT_FRACTION = 0.80
"""float: Fraction of a clade that must share a family for the branch to take its color."""
FLOAT_TREE_BRANCH_LW = 1.3
"""float: Line width of the dendrogram branches."""
FLOAT_TREE_PSEUDO_BORDER_LW = 0.5
"""float: Border width for pseudokinase leaf boxes (active kinases use a thin white border)."""
FLOAT_TREE_GUIDE_LW = 0.3
"""float: Line width of the leaf->table guide lines."""

# --- split supplemental figures (summary dendrogram + top / bottom detail panels) ---
STR_FILE_CONSERVATION_TREE_SUMMARY = "klifs_conservation_tree_summary"
"""str: Output basename for the horizontal summary dendrogram."""
STR_FILE_CONSERVATION_TREE_TOP = "klifs_conservation_tree_top"
"""str: Output basename for the top (first-half) detail panel."""
STR_FILE_CONSERVATION_TREE_BOTTOM = "klifs_conservation_tree_bottom"
"""str: Output basename for the bottom (second-half) detail panel."""
FLOAT_TREE_SPLIT_FONT_SIZE = 4.5
"""float: Font size for the split detail panels (fewer rows -> a touch larger than the
single full-page figure)."""
FLOAT_TREE_DETAIL_PAGE_W = 9.5
"""float: Width (inches) of each split detail panel; height is flexible (row count)."""
FLOAT_TREE_DETAIL_ROW_IN = 0.18
"""float: Inches of page height per leaf row in a split detail panel."""
FLOAT_TREE_DETAIL_LEGEND_IN = 0.4
"""float: Inches of page height reserved beneath a split detail panel for its
kinase-group legend."""
FLOAT_TREE_DETAIL_LEGEND_FS = 13.0
"""float: Font size for the top/bottom detail-panel kinase-group legend (much larger
than the table font so it is legible in the supplement)."""
TUP_TREE_SUMMARY_PAGE = (11.0, 5.0)
"""tuple[float, float]: Figure size (inches) of the horizontal summary dendrogram."""
FLOAT_TREE_SUMMARY_LEGEND_FS = 7.0
"""float: Font size for the summary dendrogram's kinase-group legend."""
INT_TREE_SPLIT_INDEX = 47
"""int: Default leaf-order index splitting the top/bottom detail panels of the
human-kinome KLIFS conservation tree -- the boundary between the mostly-CMGC clade
and the mostly-CAMK clade. A property of the fixed KLIFS panel (study-independent),
so it is hardcoded; :meth:`KLIFSConservationTreeFigure._split_index` is retained as a
fallback (used when this index is None or out of range for a non-standard panel)."""
STR_BLOSUM_MATRIX = "BLOSUM62"
"""str: Substitution matrix name (Bio.Align.substitution_matrices) for the default metric."""
STR_METRIC_BLOSUM = "blosum"
"""str: Name of the substitution-score similarity metric (Metric A, default)."""
STR_METRIC_IDENTITY = "identity"
"""str: Name of the percent-identity distance metric (Metric B, baseline)."""
STR_LINKAGE_AVERAGE = "average"
"""str: UPGMA linkage — standard for sequence trees (primary)."""
STR_LINKAGE_COMPLETE = "complete"
"""str: Complete linkage — tighter, more compact clusters (contrast)."""
STR_WEIGHTING_NONE = "none"
"""str: Unweighted per-node consensus (every member counts equally, default)."""
STR_WEIGHTING_HENIKOFF = "henikoff"
"""str: Henikoff (1994) position-based sequence weighting for the per-node consensus."""

# --- conservation-tree explorer (KLIFSTreeConservationApp) display params ---
STR_FILE_TREE_APP = "conservation_tree_explorer.html"
"""str: Output filename for the interactive KLIFS conservation-tree explorer."""
# INT_TREE_MIN_CLUSTER_DISPLAY is defined above (shared default).
FLOAT_TREE_LOGO_CUTOFF = 0.10
"""float: Minimum consensus frequency for a residue to appear in the per-node logo."""
FLOAT_TREE_LOGO_IC_FLOOR = 0.5
"""float: Minimum information content (bits) for a residue to appear in the IC logo modes."""
FLOAT_TREE_LOGO_IC_MAX = 4.6
"""float: y-axis top (bits) for the information-content logo modes."""
INT_TREE_NAME_TRUNC = 14
"""int: Max items listed in a conservation-tree tooltip field before truncation."""

# --- per-amino-acid KLIFS dot plot (KLIFSResidueDotApp / residue-dot figure) ---
STR_FILE_RESIDUE_DOT_APP = "conservation_residue_dotplot.html"
"""str: Output filename for the interactive per-amino-acid KLIFS dot-plot explorer."""
STR_FILE_RESIDUE_DOT = "conservation_residue_dotplot"
"""str: Output basename for the static per-amino-acid KLIFS dot-plot figure."""
STR_RESIDUE_DOT_DEFAULT_AA = "C"
"""str: Default amino acid for the static dot-plot figure (cysteine)."""
INT_RESIDUE_DOT_MIN_ENCLOSE = 2
"""int: Minimum same-clade dots stacked in a column for a clade enclosure to be drawn."""
LIST_DOT_ALPHABET = [aa for _, aa, _ in AA_MAPPING] + ["-"]
"""list[str]: The 20 amino-acid single-letter codes plus the gap symbol, for the
dot-plot amino-acid selector."""
DICT_DOT_AA_LABEL = {aa: f"{name.capitalize()} ({aa})" for name, aa, _ in AA_MAPPING}
DICT_DOT_AA_LABEL["-"] = "Gap (-)"
"""dict[str, str]: Full amino-acid label (e.g. ``"Cysteine (C)"``) per dot-plot symbol,
used for the interactive selector options and heading."""
LIST_RESIDUE_DOT_KINASES = ["EGFR", "BTK", "FGFR1", "FGFR2", "FGFR3", "FGFR4"]
"""list[str]: Clinically relevant kinases of interest whose covalent-targetable residues
are called out on the static dot plot. Columns are not hardcoded: every KLIFS column where
at least :data:`INT_RESIDUE_DOT_MIN_SHARED` of these kinases carry the plotted residue is
boxed (:meth:`_annotation_callouts`), with kinases sharing a homologous column grouped in
one box. For cysteine this captures all of these kinases' pocket cysteines -- FGFR1-4 at
g.l:7 and VI:66, EGFR C797 / BTK C481 at linker:52, plus the singletons EGFR C775 (b.l:36),
FGFR4 C552 (hinge:47), and BTK C527 (VII:76) -- and tracks the current KLIFS mapping."""
INT_RESIDUE_DOT_MIN_SHARED = 1
"""int: Minimum kinases-of-interest carrying a residue at a column to draw a callout box
(1 = annotate every cysteine of the kinases of interest, grouping any that share a column)."""


def _build_klifs_panel(
    dict_kinase: dict = DICT_KINASE,
    int_length: int = INT_KLIFS_POCKET_LENGTH,
) -> tuple[list[str], list[str], list[str]]:
    """Collect the gapless KLIFS pocket panel for hierarchical clustering.

    Iterates :data:`DICT_KINASE`, keeping kinases with a valid fixed-width pocket
    string and recording their Manning group for the concordance check. Kinases
    without a pocket, or whose pocket is the wrong length, are skipped with a log
    warning.

    Parameters:
    -----------
    dict_kinase : dict
        Mapping of HGNC name to ``KinaseInfo`` (default: :data:`DICT_KINASE`).
    int_length : int
        Required pocket length (default: :data:`INT_KLIFS_POCKET_LENGTH`).

    Returns:
    --------
    tuple[list[str], list[str], list[str]]
        Parallel lists of ``(names, pockets, manning_groups)``.
    """
    names, pockets, groups = [], [], []
    for name, info in dict_kinase.items():
        if info.klifs is None or info.klifs.pocket_seq is None:
            continue
        pocket = info.klifs.pocket_seq
        if len(pocket) != int_length:
            logger.warning(
                "dropping %s: pocket length %d != %d", name, len(pocket), int_length
            )
            continue
        names.append(name)
        pockets.append(pocket)
        groups.append(info.klifs.group)
    return names, pockets, groups


_KLIFS_PANEL = _build_klifs_panel()
"""tuple[list[str], list[str], list[str]]: cached (names, pockets, groups) KLIFS panel."""


class _HandlerRoundedBox(HandlerPatch):
    """Legend handler that renders a :class:`~matplotlib.patches.Patch` handle as a
    rounded box (matching the dot-plot clade enclosures) rather than a sharp rectangle.
    """

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        box = FancyBboxPatch(
            (-xdescent + 1.0, -ydescent + 1.0),
            width - 2.0,
            height - 2.0,
            boxstyle="round,pad=0,rounding_size=2.5",
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            linewidth=orig_handle.get_linewidth(),
            transform=trans,
        )
        return [box]


@dataclass
class KLIFSDisplayLeaf:
    """A leaf row of the display tree: a collapsed clade or aggregated singletons.

    Attributes
    ----------
    members : list[int]
        Leaf indices (into :attr:`KLIFSHierarchicalConservation.names`) in this row.
    kind : str
        ``"clade"`` (a collapsed subtree) or ``"singleton"`` (a single peeled leaf,
        present only when singleton aggregation is disabled).
    """

    members: list[int]
    kind: str


@dataclass
class KLIFSDisplaySplit:
    """An internal (multifurcating) node of the display tree.

    Attributes
    ----------
    id : int
        Index of this split within :attr:`KLIFSDisplayTree.splits` (root is 0).
    node_id : int
        Underlying SciPy linkage node id.
    depth : int
        Depth from the root (root is 0).
    members : list[int]
        All leaf indices under this split.
    children : list[tuple[str, int]]
        Ordered children as ``("node", split_id)`` or ``("leaf", leaf_index)``.
    """

    id: int
    node_id: int
    depth: int
    members: list[int]
    children: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class KLIFSDisplayTree:
    """Renderer-neutral display tree shared by the static figure and Bokeh app.

    Attributes
    ----------
    splits : list[KLIFSDisplaySplit]
        Internal multifurcating nodes; ``splits[0]`` is the root.
    leaves : list[KLIFSDisplayLeaf]
        Leaf rows (collapsed clades / aggregated singletons).
    order : list[int]
        Leaf indices (into :attr:`leaves`) in in-order (top-to-bottom) display order.
    """

    splits: list[KLIFSDisplaySplit]
    leaves: list[KLIFSDisplayLeaf]
    order: list[int]


class KLIFSHierarchicalConservation(BaseModel):
    """Hierarchically cluster human kinases on KLIFS pocket similarity.

    Builds a pairwise distance matrix over the gapless 85-column KLIFS pocket
    alignment (:func:`_build_klifs_panel`), clusters it with SciPy linkage, then
    traces per-residue conservation up the tree. Per-node conservation reuses
    :meth:`MSAConservationAnalyzer.column_conservation`, keeping the tree and the
    conservation read-out in the same coordinate system.

    Two metrics are provided behind a common interface: substitution-score
    similarity (Metric A, ``"blosum"``, default) and percent identity
    (Metric B, ``"identity"``, baseline). Two views of conservation are emitted at
    each internal node: per-subtree (recompute on that node's members) and
    survives-up (the residue must be conserved at the node *and* hold — same
    consensus residue — in both child clades).
    """

    model_config = {"arbitrary_types_allowed": True}

    names: list[str] = Field(default_factory=lambda: list(_KLIFS_PANEL[0]))
    """List of HGNC kinase names (one per pocket)."""
    pockets: list[str] = Field(default_factory=lambda: list(_KLIFS_PANEL[1]))
    """List of 85-character KLIFS pocket strings, column-aligned by construction."""
    groups: list[str] = Field(default_factory=lambda: list(_KLIFS_PANEL[2]))
    """Manning group per kinase, used for the external concordance check."""
    position_labels: list[str] = Field(default_factory=lambda: list(LIST_KLIFS_REGION))
    """KLIFS region labels (e.g. ``"g.l:4"``) for the 85 pocket columns."""
    metric: str = STR_METRIC_BLOSUM
    """Pairwise distance metric: ``"blosum"`` (default) or ``"identity"``."""
    blosum_name: str = STR_BLOSUM_MATRIX
    """Substitution matrix name used by the ``"blosum"`` metric."""
    linkage_method: str = STR_LINKAGE_AVERAGE
    """SciPy linkage method: ``"average"`` (UPGMA, default) or ``"complete"``."""
    conservation_threshold: float = FLOAT_CONSERVATION_THRESHOLD
    """Minimum consensus-residue frequency for a column to count as conserved."""
    min_child_members: int = INT_MIN_CHILD_MEMBERS
    """Minimum child-clade size considered in survives-up pruning; children smaller than
    this (e.g. singleton outliers peeled off at the root) are ignored. Set to 1 to require
    every immediate child to agree (the strict, topology-sensitive definition)."""
    weighting: str = STR_WEIGHTING_NONE
    """Per-node consensus weighting: ``"none"`` (default) or ``"henikoff"``."""
    exclude_pseudokinases: bool = False
    """If True, drop predicted pseudokinases (``KinaseInfo.is_pseudokinase()``) from the
    panel before clustering, so the tree and conservation reflect catalytically active
    kinases only (default: False)."""
    gap_chars: str = STR_GAP_CHARS
    """Characters treated as gap/unknown and excluded from scoring and conservation."""

    distance_matrix: np.ndarray | None = None
    """Square ``N x N`` pairwise distance matrix; computed post-init unless supplied
    (e.g. injected by :meth:`from_conservation_data` from a persisted artifact)."""
    linkage_matrix: np.ndarray | None = None
    """SciPy linkage matrix; computed post-init unless supplied."""
    tree: ClusterNode | None = Field(default=None, init=False)
    """Root ``ClusterNode`` of the hierarchical tree, computed post-init."""
    _node_records: list[dict] | None = None
    """Cached per-node conservation records (see :meth:`analyze_nodes`)."""
    _nodelist_cache: list | None = None
    """Cached SciPy ``to_tree(rd=True)`` node list, indexed by node id."""
    _kinome_bg_cache: np.ndarray | None = None
    """Cached kinome amino-acid background (from the KLIFS panel) for the IC measure."""
    _pssm_cache: dict | None = None
    """Cached substitution-aware pseudocount models, keyed by background choice."""

    def model_post_init(self, __context) -> None:
        # precomputed distances + linkage supplied (e.g. loaded from a persisted
        # KLIFSConservationData artifact): trust them and skip the expensive recompute,
        # only reconstructing the ClusterNode tree from the linkage
        if self.distance_matrix is not None and self.linkage_matrix is not None:
            if self.tree is None:
                self.tree = to_tree(self.linkage_matrix, rd=False)
            return
        if self.exclude_pseudokinases:
            self._drop_pseudokinases()
        self.distance_matrix = self.compute_distance_matrix()
        self.linkage_matrix = linkage(
            squareform(self.distance_matrix, checks=False),
            method=self.linkage_method,
        )
        self.tree = to_tree(self.linkage_matrix, rd=False)

    @classmethod
    def from_conservation_data(
        cls, data: KLIFSConservationData, **kwargs
    ) -> "KLIFSHierarchicalConservation":
        """Build a renderer from a persisted :class:`KLIFSConservationData` artifact.

        Reuses the stored distances + linkage (so the rendered tree matches the
        persisted one and no clustering is recomputed), recovering the per-kinase
        pocket/group panel from :data:`DICT_KINASE` by name. Any renderer-specific
        keyword (e.g. ``min_cluster_size``, ``font_size``) is passed through.

        Parameters
        ----------
        data : KLIFSConservationData
            The persisted conservation-data artifact.
        **kwargs
            Extra fields forwarded to the (sub)class constructor.

        Returns
        -------
        KLIFSHierarchicalConservation
            An instance of ``cls`` with injected distances, linkage, and tree.
        """
        panel_names, panel_pockets, panel_groups = _KLIFS_PANEL
        lookup = {
            name: (pocket, group)
            for name, pocket, group in zip(panel_names, panel_pockets, panel_groups)
        }
        pockets, groups = [], []
        for name in data.names:
            pocket, group = lookup[name]
            pockets.append(pocket)
            groups.append(group)

        distance_matrix = squareform(np.array(data.distances_condensed, dtype=float))
        return cls(
            names=list(data.names),
            pockets=pockets,
            groups=groups,
            position_labels=list(data.position_labels),
            metric=data.metric,
            blosum_name=data.blosum_name,
            linkage_method=data.linkage_method,
            conservation_threshold=data.conservation_threshold,
            weighting=data.weighting,
            gap_chars=data.gap_chars,
            distance_matrix=distance_matrix,
            linkage_matrix=np.array(data.linkage_matrix, dtype=float),
            **kwargs,
        )

    def _drop_pseudokinases(self) -> None:
        """Filter the panel to catalytically active kinases (in place).

        Drops any panel member for which ``KinaseInfo.is_pseudokinase()`` is True,
        keeping :attr:`names`, :attr:`pockets` and :attr:`groups` in sync. Names absent
        from :data:`DICT_KINASE` (e.g. a custom panel) cannot be assessed and are kept.

        Returns:
        --------
        None
        """
        keep = [
            i
            for i, name in enumerate(self.names)
            if name not in DICT_KINASE or not DICT_KINASE[name].is_pseudokinase()
        ]
        n_dropped = len(self.names) - len(keep)
        self.names = [self.names[i] for i in keep]
        self.pockets = [self.pockets[i] for i in keep]
        self.groups = [self.groups[i] for i in keep]
        logger.info(
            "excluded %d predicted pseudokinases; %d kinases remain",
            n_dropped,
            len(keep),
        )

    def _encode_pockets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integer-encode the pocket panel against the substitution-matrix alphabet.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(enc, valid, mat_arr)`` where ``enc`` is the ``N x 85`` integer code
            (gaps mapped to 0 but masked out), ``valid`` is the ``N x 85`` boolean
            non-gap mask, and ``mat_arr`` is the dense substitution-score matrix.
        """
        mat = substitution_matrices.load(self.blosum_name)
        mat_arr = np.array(mat, dtype=float)
        index = {c: i for i, c in enumerate(mat.alphabet)}

        n_seqs = len(self.pockets)
        enc = np.zeros((n_seqs, INT_KLIFS_POCKET_LENGTH), dtype=int)
        valid = np.zeros((n_seqs, INT_KLIFS_POCKET_LENGTH), dtype=bool)
        for i, pocket in enumerate(self.pockets):
            for j, char in enumerate(pocket):
                if char in self.gap_chars or char not in index:
                    continue
                enc[i, j] = index[char]
                valid[i, j] = True
        return enc, valid, mat_arr

    def _blosum_distance_matrix(self) -> np.ndarray:
        """Substitution-score similarity distance (Metric A).

        Sums the substitution score over jointly non-gap columns, normalizes by each
        sequence's self-score (cosine-like in score space), and converts to a
        distance ``1 - sim``.

        Returns:
        --------
        np.ndarray
            Square ``N x N`` distance matrix with a zero diagonal.
        """
        enc, valid, mat_arr = self._encode_pockets()
        n_seqs = enc.shape[0]

        score_cross = np.zeros((n_seqs, n_seqs))
        score_self = np.zeros(n_seqs)
        for pos in range(INT_KLIFS_POCKET_LENGTH):
            col = enc[:, pos]
            obs = valid[:, pos]
            joint = obs[:, None] & obs[None, :]
            score_cross += mat_arr[np.ix_(col, col)] * joint
            score_self += np.where(obs, mat_arr[col, col], 0.0)

        norm = np.sqrt(np.outer(score_self, score_self))
        sim = np.divide(
            score_cross, norm, out=np.zeros_like(score_cross), where=norm > 0
        )
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        return dist

    def _identity_distance_matrix(self) -> np.ndarray:
        """Percent-identity distance (Metric B): ``1 - matches / non_gap_columns``.

        Returns:
        --------
        np.ndarray
            Square ``N x N`` distance matrix with a zero diagonal.
        """
        enc, valid, _ = self._encode_pockets()
        n_seqs = enc.shape[0]

        matches = np.zeros((n_seqs, n_seqs))
        n_joint = np.zeros((n_seqs, n_seqs))
        for pos in range(INT_KLIFS_POCKET_LENGTH):
            col = enc[:, pos]
            obs = valid[:, pos]
            joint = obs[:, None] & obs[None, :]
            matches += (col[:, None] == col[None, :]) & joint
            n_joint += joint

        identity = np.divide(
            matches, n_joint, out=np.zeros_like(matches), where=n_joint > 0
        )
        dist = 1.0 - identity
        np.fill_diagonal(dist, 0.0)
        return dist

    def compute_distance_matrix(self) -> np.ndarray:
        """Dispatch to the configured pairwise distance metric.

        Returns:
        --------
        np.ndarray
            Square ``N x N`` distance matrix.
        """
        if self.metric == STR_METRIC_BLOSUM:
            return self._blosum_distance_matrix()
        if self.metric == STR_METRIC_IDENTITY:
            return self._identity_distance_matrix()
        raise ValueError(
            f"unknown metric {self.metric!r}; expected "
            f"{STR_METRIC_BLOSUM!r} or {STR_METRIC_IDENTITY!r}"
        )

    def cophenetic_correlation(self) -> float:
        """Cophenetic correlation between the tree and the input distances.

        Returns:
        --------
        float
            Pearson correlation of the cophenetic distances against the condensed
            input distance matrix (higher is a more faithful tree).
        """
        coph_corr, _ = cophenet(
            self.linkage_matrix, squareform(self.distance_matrix, checks=False)
        )
        return float(coph_corr)

    def group_concordance(self, n_clusters: int | None = None) -> float:
        """Adjusted Rand index between flat clusters and Manning groups.

        Parameters:
        -----------
        n_clusters : int | None
            Number of flat clusters to cut the tree into. Defaults to the number of
            distinct Manning groups in the panel.

        Returns:
        --------
        float
            Adjusted Rand index (1.0 = perfect concordance, ~0 = chance).
        """
        if n_clusters is None:
            n_clusters = len(set(self.groups))
        cluster_labels = fcluster(
            self.linkage_matrix, t=n_clusters, criterion="maxclust"
        )
        return float(adjusted_rand_score(self.groups, cluster_labels))

    def _henikoff_weights(self, sequences: list[str]) -> list[float]:
        """Henikoff (1994) position-based sequence weights for a set of members.

        Each non-gap residue contributes ``1 / (r_c * s_c)`` to its sequence's weight,
        where ``r_c`` is the number of distinct residues in column ``c`` and ``s_c`` is
        the number of sequences sharing that residue. This down-weights redundant
        subfamilies and up-weights divergent sequences. Weights are scaled to mean 1;
        the per-column conservation fraction is scale-invariant regardless.

        Parameters:
        -----------
        sequences : list[str]
            Aligned member sequences.

        Returns:
        --------
        list[float]
            One weight per sequence.
        """
        n_seqs = len(sequences)
        if n_seqs == 1:
            return [1.0]

        weights = np.zeros(n_seqs)
        for col in zip(*sequences):
            counts: dict[str, int] = {}
            for residue in col:
                if residue in self.gap_chars:
                    continue
                counts[residue] = counts.get(residue, 0) + 1
            n_distinct = len(counts)
            if n_distinct == 0:
                continue
            for i, residue in enumerate(col):
                if residue in self.gap_chars:
                    continue
                weights[i] += 1.0 / (n_distinct * counts[residue])

        total = weights.sum()
        if total > 0:
            weights *= n_seqs / total
        return weights.tolist()

    def node_conservation(
        self, member_idx: list[int]
    ) -> list[tuple[str | None, float]]:
        """Per-column conservation for a set of member leaves.

        Reuses :meth:`MSAConservationAnalyzer.column_conservation` so the per-node
        read-out matches the panel-wide conservation analysis. When :attr:`weighting`
        is ``"henikoff"`` the consensus is computed with per-node Henikoff sequence
        weights (:meth:`_henikoff_weights`).

        Parameters:
        -----------
        member_idx : list[int]
            Indices into :attr:`pockets` of the node's member kinases.

        Returns:
        --------
        list[tuple[str | None, float]]
            One ``(consensus_aa, fraction)`` tuple per KLIFS column.
        """
        members = [self.pockets[i] for i in member_idx]
        weights = None
        if self.weighting == STR_WEIGHTING_HENIKOFF:
            weights = self._henikoff_weights(members)
        elif self.weighting != STR_WEIGHTING_NONE:
            raise ValueError(
                f"unknown weighting {self.weighting!r}; expected "
                f"{STR_WEIGHTING_NONE!r} or {STR_WEIGHTING_HENIKOFF!r}"
            )
        return column_conservation(members, gap_chars=self.gap_chars, weights=weights)

    def _conserved_columns(
        self, conservation: list[tuple[str | None, float]]
    ) -> dict[int, str]:
        """Filter per-column conservation to the conserved consensus residues.

        Parameters:
        -----------
        conservation : list[tuple[str | None, float]]
            Per-column ``(consensus_aa, fraction)`` tuples.

        Returns:
        --------
        dict[int, str]
            Mapping of column index to consensus residue for columns whose consensus
            frequency meets :attr:`conservation_threshold`.
        """
        return {
            pos: aa
            for pos, (aa, frac) in enumerate(conservation)
            if aa is not None and frac >= self.conservation_threshold
        }

    def _walk_node(
        self, node: ClusterNode, depth: int, records: list[dict]
    ) -> dict[int, str]:
        """Recursively score a node and its children, appending one record per node.

        For each node the per-subtree conserved columns are recomputed on its members.
        At an internal node the survives-up set keeps only columns conserved at the
        node whose consensus residue is *also* conserved (identical) in every child
        clade with at least :attr:`min_child_members` members. Children below that size
        (e.g. a singleton outlier peeled off at the root by UPGMA) are ignored so they
        cannot veto an otherwise-robust residue.

        Parameters:
        -----------
        node : ClusterNode
            Current node in the tree.
        depth : int
            Depth of the node (root is 0).
        records : list[dict]
            Accumulator the per-node record dicts are appended to.

        Returns:
        --------
        dict[int, str]
            The node's per-subtree conserved columns (column index -> consensus aa),
            so the parent can intersect against it for the survives-up set.
        """
        member_idx = node.pre_order()
        conserved = self._conserved_columns(self.node_conservation(member_idx))

        if node.is_leaf():
            survives_up = dict(conserved)
        else:
            left, right = node.get_left(), node.get_right()
            conserved_left = self._walk_node(left, depth + 1, records)
            conserved_right = self._walk_node(right, depth + 1, records)
            # only children above the size floor can veto a residue
            qualifying = []
            if left.get_count() >= self.min_child_members:
                qualifying.append(conserved_left)
            if right.get_count() >= self.min_child_members:
                qualifying.append(conserved_right)
            survives_up = {
                pos: aa
                for pos, aa in conserved.items()
                if all(child.get(pos) == aa for child in qualifying)
            }

        records.append(
            {
                "node_id": node.id,
                "depth": depth,
                "n_members": len(member_idx),
                "is_leaf": node.is_leaf(),
                "conserved": conserved,
                "survives_up": survives_up,
            }
        )
        return conserved

    def analyze_nodes(self) -> list[dict]:
        """Walk the tree and compute per-node conservation, cached after first call.

        Returns:
        --------
        list[dict]
            One record per node with keys ``node_id``, ``depth``, ``n_members``,
            ``is_leaf``, ``conserved`` (per-subtree column->aa) and ``survives_up``
            (the pruned, robust column->aa).
        """
        if self._node_records is None:
            records: list[dict] = []
            self._walk_node(self.tree, 0, records)
            self._node_records = records
        return self._node_records

    def nodes_summary(self) -> pd.DataFrame:
        """Tabular summary of per-node conservation counts (internal nodes only).

        Returns:
        --------
        pd.DataFrame
            One row per internal node with ``node_id``, ``depth``, ``n_members``,
            ``n_conserved`` (per-subtree), ``n_survives_up`` (pruned) and ``n_lost``
            (conserved but not robust), sorted by depth.
        """
        rows = [
            {
                "node_id": rec["node_id"],
                "depth": rec["depth"],
                "n_members": rec["n_members"],
                "n_conserved": len(rec["conserved"]),
                "n_survives_up": len(rec["survives_up"]),
                "n_lost": len(rec["conserved"]) - len(rec["survives_up"]),
            }
            for rec in self.analyze_nodes()
            if not rec["is_leaf"]
        ]
        return pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)

    def critical_depth(self) -> pd.DataFrame:
        """Most ancestral node at which each KLIFS column still survives-up.

        A column's critical depth is the shallowest (closest to root) depth at which
        it appears in a node's survives-up set. Columns surviving at depth 0 are the
        pan-kinase invariants (expect the glycine-rich loop, the HRD arginine and the
        DFG aspartate).

        Returns:
        --------
        pd.DataFrame
            One row per KLIFS column with ``column`` (0-based), ``position_label``,
            ``critical_depth`` (NaN if never robust) and ``consensus_aa`` at the
            shallowest surviving node, sorted by critical depth.
        """
        best: dict[int, tuple[int, str]] = {}
        for rec in self.analyze_nodes():
            for pos, aa in rec["survives_up"].items():
                if pos not in best or rec["depth"] < best[pos][0]:
                    best[pos] = (rec["depth"], aa)

        rows = []
        for pos in range(INT_KLIFS_POCKET_LENGTH):
            depth, aa = best.get(pos, (np.nan, None))
            rows.append(
                {
                    "column": pos,
                    "position_label": (
                        self.position_labels[pos]
                        if pos < len(self.position_labels)
                        else str(pos)
                    ),
                    "critical_depth": depth,
                    "consensus_aa": aa,
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("critical_depth", na_position="last")
            .reset_index(drop=True)
        )

    def plot_critical_depth(
        self,
        figsize: tuple[float, float] = (15, 3),
        cmap: str = CMAP_CRITICAL_DEPTH,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot per-KLIFS-column critical depth as an opaque track with a region bar.

        Each of the 85 KLIFS columns is colored by its critical depth — the shallowest
        (most ancestral) node at which it still survives-up (:meth:`critical_depth`).
        Lower depth means the residue holds across more of the kinome (depth 0 = a
        pan-kinase invariant); columns that never survive-up are drawn in grey. A KLIFS
        region color bar and position labels run beneath the track. Colors are flattened
        to opaque RGB (no alpha) for PowerPoint/PDF safety.

        Parameters:
        -----------
        figsize : tuple[float, float]
            Figure size (default: (15, 3)).
        cmap : str
            Matplotlib colormap name for the depth scale (default:
            :data:`CMAP_CRITICAL_DEPTH`).

        Returns:
        --------
        tuple[plt.Figure, plt.Axes]
            Figure and axis objects.
        """
        df = self.critical_depth().sort_values("column").reset_index(drop=True)
        depths = df["critical_depth"].to_numpy(dtype=float)
        labels = df["position_label"].tolist()
        n = len(labels)

        cmap_obj = plt.get_cmap(cmap)
        finite = depths[np.isfinite(depths)]
        vmax = float(finite.max()) if finite.size else 1.0
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("white")

        # depth track: one opaque rectangle per column (mcolors.to_hex drops any alpha)
        for i, depth in enumerate(depths):
            color = (
                COLOR_DEPTH_NA
                if not np.isfinite(depth)
                else mcolors.to_hex(cmap_obj(norm(depth)))
            )
            ax.add_patch(
                Rectangle(
                    (i - 0.5, 0),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5,
                )
            )

        # KLIFS region color bar beneath the track
        for i, label in enumerate(labels):
            region_color = DICT_POCKET_KLIFS_REGIONS[label.split(":")[0]]["color"]
            ax.add_patch(
                Rectangle(
                    (i - 0.5, -0.28),
                    1,
                    0.22,
                    facecolor=region_color,
                    edgecolor="white",
                    linewidth=0.5,
                )
            )
            ax.text(i, -0.34, label, rotation=90, ha="center", va="top", fontsize=6)

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.95, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        remove_spines(ax)

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.015, pad=0.01)
        cbar.set_label("Critical depth (0 = pan-kinase)")
        cbar.outline.set_visible(False)

        fig.tight_layout()
        return fig, ax

    # =========================================================================
    # shared display tree (consumed by the static figure and the Bokeh app)
    # =========================================================================

    def _nodelist(self) -> list:
        """Return the cached SciPy node list, indexed by node id."""
        if self._nodelist_cache is None:
            _, self._nodelist_cache = to_tree(self.linkage_matrix, rd=True)
        return self._nodelist_cache

    def tree_children(self, node_id: int) -> tuple[int, int]:
        """The two linkage children of an internal node id."""
        row = self.linkage_matrix[node_id - len(self.names)]
        return int(row[0]), int(row[1])

    def tree_count(self, node_id: int) -> int:
        """Number of leaves under ``node_id`` (1 for a leaf)."""
        n = len(self.names)
        return 1 if node_id < n else self._nodelist()[node_id].get_count()

    def tree_members(self, node_id: int) -> list[int]:
        """Leaf indices under ``node_id`` (the node itself if it is a leaf)."""
        n = len(self.names)
        return [node_id] if node_id < n else self._nodelist()[node_id].pre_order()

    def _is_kept(self, node_id: int) -> bool:
        """True if both children of ``node_id`` are clades (>=2 members each).

        A node fails this when one child is a singleton leaf peeling off, which is
        exactly the pattern :meth:`gather_children` contracts away.
        """
        n = len(self.names)
        if node_id < n:
            return False
        a, b = self.tree_children(node_id)
        return self.tree_count(a) >= 2 and self.tree_count(b) >= 2

    def gather_children(self, node_id: int) -> list[int]:
        """Singleton-contracted multifurcating children of ``node_id``.

        Descends through singleton-peel chains (nodes that are not
        :meth:`_is_kept`) so a kinase that peels off attaches directly to its clade,
        yielding a multifurcating child list of leaves and kept clades.

        Parameters
        ----------
        node_id : int
            Internal linkage node id.

        Returns
        -------
        list[int]
            Child node ids (leaves and kept internal nodes).
        """
        n = len(self.names)
        res = []
        for c in self.tree_children(node_id):
            if c < n or self._is_kept(c):
                res.append(c)
            else:
                res.extend(self.gather_children(c))
        return res

    def members_consensus(self, member_idx: list[int]) -> list[str | None]:
        """Per-column >=threshold consensus residue for a member set (else None).

        Works for any set of leaves -- a single leaf, a clade, or an aggregated row
        of unrelated singletons -- so it backs both the per-node logo and the static
        per-leaf conservation table.

        Parameters
        ----------
        member_idx : list[int]
            Leaf indices into :attr:`pockets`.

        Returns
        -------
        list[str | None]
            One entry per KLIFS column: the consensus residue if its frequency meets
            :attr:`conservation_threshold`, else ``None``.
        """
        thr = self.conservation_threshold
        return [
            aa if (aa is not None and frac >= thr) else None
            for aa, frac in self.node_conservation(member_idx)
        ]

    def _kinome_background(self) -> np.ndarray:
        """Kinome amino-acid background (20-vector) over all KLIFS panel pockets.

        The preferred KL background for the information-content measure ("surprising
        *relative to kinases*"): a residue conserved because it is the kinome norm scores
        low, while a subfamily-specific residue scores high (plan Addendum A.1)."""
        if self._kinome_bg_cache is None:
            counts = np.zeros(len(DICT_AA20_INDEX))
            for pocket in self.pockets:
                for char in pocket:
                    idx = DICT_AA20_INDEX.get(char)
                    if idx is not None:
                        counts[idx] += 1.0
            self._kinome_bg_cache = counts / counts.sum()
        return self._kinome_bg_cache

    def _pseudocount_model(
        self, background: str = "swissprot"
    ) -> SubstitutionPseudocounts:
        """Cached pseudocount model for the requested KL background.

        ``"swissprot"`` (default) scores conservation relative to a random protein -- the
        continuous companion to the >=80% rule (pan-invariants score high). ``"kinome"``
        scores conservation relative to the KLIFS panel itself, so subfamily-defining
        residues pop while pan-invariants score near zero (plan-preferred for the
        subfamily-delta view)."""
        if self._pssm_cache is None:
            self._pssm_cache = {}
        if background not in self._pssm_cache:
            bg = self._kinome_background() if background == "kinome" else None
            self._pssm_cache[background] = SubstitutionPseudocounts(background=bg)
        return self._pssm_cache[background]

    def node_information(
        self, member_idx: list[int], background: str = "swissprot"
    ) -> tuple[list[float | None], list[float]]:
        """Per-column background-relative information content (bits) and effective count.

        Continuous companion to :meth:`members_consensus` (plan Addendum A.2): for each
        KLIFS column, the KL divergence of the members' substitution-smoothed residue
        distribution against the chosen background, plus the effective (weighted) count so
        the reader can see where the score is prior-influenced vs data-dominated. Sequence
        weighting follows :attr:`weighting` (uniform by default; Henikoff when set), the
        substitution-aware pseudocount is always applied.

        Parameters
        ----------
        member_idx : list[int]
            Leaf indices into :attr:`pockets`.
        background : str, optional
            KL background: ``"swissprot"`` (default) or ``"kinome"`` (see
            :meth:`_pseudocount_model`).

        Returns
        -------
        tuple[list[float | None], list[float]]
            ``(ic_per_column, n_eff_per_column)``; ``ic`` is ``None`` for an all-gap column.
        """
        pssm = self._pseudocount_model(background)
        members = [self.pockets[i] for i in member_idx]
        if self.weighting == STR_WEIGHTING_HENIKOFF:
            weights = self._henikoff_weights(members)
        else:
            weights = [1.0] * len(members)

        ic: list[float | None] = []
        n_eff: list[float] = []
        for col in range(len(self.position_labels)):
            counts = np.zeros(len(DICT_AA20_INDEX))
            for seq, w in zip(members, weights):
                idx = DICT_AA20_INDEX.get(seq[col])
                if idx is not None:
                    counts[idx] += w
            ic.append(pssm.column_information(counts))
            n_eff.append(float(counts.sum()))
        return ic, n_eff

    def build_display_tree(
        self,
        min_cluster_size: int = INT_TREE_MIN_CLUSTER_DISPLAY,
        aggregate_singletons: bool = True,
    ) -> KLIFSDisplayTree:
        """Build the renderer-neutral display tree from the linkage.

        Starting at the root, :meth:`gather_children` contracts singleton-peel chains;
        each gathered child then either expands into its own split (>= ``min_cluster_size``
        members), collapses to a clade leaf, or is a peeled singleton leaf. When
        ``aggregate_singletons`` is True (default) singleton leaves are folded by tree
        adjacency so no leaf row holds a single sequence (which would read as
        artificially 100% conserved): >=2 sibling singletons fold into one row, a lone
        sibling singleton folds into its nearest sibling clade, and otherwise it bubbles
        up to the parent split.

        Parameters
        ----------
        min_cluster_size : int, optional
            Members below which a gathered subtree collapses to a single leaf bar,
            by default :data:`INT_TREE_MIN_CLUSTER_DISPLAY`.
        aggregate_singletons : bool, optional
            Fold singleton leaves so every leaf row has >=2 members, by default True.

        Returns
        -------
        KLIFSDisplayTree
            Splits, leaves, and the in-order leaf display order.
        """
        n = len(self.names)
        splits: list[KLIFSDisplaySplit] = []
        leaves: list[KLIFSDisplayLeaf] = []

        def build(node_id: int, depth: int) -> int:
            sid = len(splits)
            splits.append(
                KLIFSDisplaySplit(sid, node_id, depth, self.tree_members(node_id), [])
            )
            for c in self.gather_children(node_id):
                if c < n:  # peeled singleton leaf
                    leaves.append(KLIFSDisplayLeaf([c], "singleton"))
                    splits[sid].children.append(("leaf", len(leaves) - 1))
                elif self.tree_count(c) >= min_cluster_size:  # expand
                    splits[sid].children.append(("node", build(c, depth + 1)))
                else:  # collapse to a clade leaf
                    leaves.append(KLIFSDisplayLeaf(self.tree_members(c), "clade"))
                    splits[sid].children.append(("leaf", len(leaves) - 1))
            return sid

        build(2 * n - 2, 0)
        if aggregate_singletons:
            self._aggregate_singletons(splits, leaves)

        order: list[int] = []

        def inorder(sid: int) -> None:
            for kind, ref in splits[sid].children:
                if kind == "node":
                    inorder(ref)
                else:
                    order.append(ref)

        inorder(0)
        return KLIFSDisplayTree(splits, leaves, order)

    def _aggregate_singletons(
        self,
        splits: list[KLIFSDisplaySplit],
        leaves: list[KLIFSDisplayLeaf],
    ) -> None:
        """Fold singleton leaves by tree adjacency so no leaf row holds one sequence.

        Mutates ``splits``/``leaves`` in place: >=2 sibling singletons fold into one
        aggregated clade row, a lone sibling singleton folds into its nearest sibling
        clade, otherwise it bubbles up to the parent split (with a final safety net into
        the largest clade). Dead leaves are dropped and child references are remapped.
        """
        alive = [True] * len(leaves)
        parent_split = {r: s.id for s in splits for k, r in s.children if k == "node"}
        bubbled: dict[int, list[int]] = defaultdict(list)

        def merge(dst: int, src: int) -> None:
            leaves[dst].members.extend(leaves[src].members)
            alive[src] = False

        for s in reversed(
            splits
        ):  # bottom-up so a bubbled singleton reaches its parent
            slots = [
                (i, r)
                for i, (k, r) in enumerate(s.children)
                if k == "leaf" and alive[r]
            ]
            sing = [(i, r) for i, r in slots if leaves[r].kind == "singleton"]
            sing += [(len(s.children), r) for r in bubbled[s.id] if alive[r]]
            clades = [(i, r) for i, r in slots if leaves[r].kind == "clade"]
            if len(sing) >= 2:  # fold all sibling singletons into one aggregated row
                members: list[int] = []
                for _, r in sing:
                    members += leaves[r].members
                    alive[r] = False
                leaves.append(KLIFSDisplayLeaf(members, "clade"))
                alive.append(True)
                s.children.append(("leaf", len(leaves) - 1))
            elif (
                len(sing) == 1
            ):  # lone singleton -> nearest sibling clade, else bubble up
                slot, r = sing[0]
                cand = [(abs(slot - j), t) for j, t in clades if alive[t]]
                if cand:
                    merge(min(cand)[1], r)
                elif s.id in parent_split:
                    bubbled[parent_split[s.id]].append(r)

        for idx in range(
            len(leaves)
        ):  # safety net: fold any survivor into largest clade
            if alive[idx] and leaves[idx].kind == "singleton":
                tgt = max(
                    (
                        j
                        for j in range(len(leaves))
                        if alive[j] and j != idx and leaves[j].kind == "clade"
                    ),
                    key=lambda j: len(leaves[j].members),
                    default=None,
                )
                if tgt is not None:
                    merge(tgt, idx)

        remap, kept = {}, []
        for i, leaf in enumerate(leaves):
            if alive[i]:
                remap[i] = len(kept)
                kept.append(leaf)
        for s in splits:
            s.children = [
                (k, remap[r] if k == "leaf" else r)
                for k, r in s.children
                if not (k == "leaf" and not alive[r])
            ]
        leaves[:] = kept

    def _member_style(self, i: int) -> tuple[str, str, float]:
        """``(fill, edge, linewidth)`` for leaf ``i``'s name box / dot.

        Lipid kinases get no border, pseudokinases a black border, active kinases a
        thin white border; the fill is the Manning-group (or Lipid) color. Shared by the
        static tree figure and the interactive dot-plot explorer.
        """
        info = DICT_KINASE.get(self.names[i])
        if info is not None and info.is_lipid_kinase():
            return (
                DICT_KINASE_GROUP_COLORS.get("Lipid", COLOR_TREE_FALLBACK),
                "none",
                0.0,
            )
        color = DICT_KINASE_GROUP_COLORS.get(self.groups[i], COLOR_TREE_FALLBACK)
        if info is not None and info.is_pseudokinase():
            return color, "#000000", FLOAT_TREE_PSEUDO_BORDER_LW
        return color, "#FFFFFF", 0.3

    def _family_label(self, i: int) -> str:
        """Family used for the color: Lipid or Manning group (pseudo folds in)."""
        info = DICT_KINASE.get(self.names[i])
        if info is not None and info.is_lipid_kinase():
            return "Lipid"
        return self.groups[i]

    def _uniprot_index_at(self, i: int, pos: int) -> int | None:
        """1-based UniProt canonical index of leaf ``i`` at KLIFS column ``pos``.

        Read from ``KinaseInfo.KLIFS2UniProtIdx`` keyed by the column's region label;
        None when the kinase, mapping, or that position is unavailable.
        """
        info = DICT_KINASE.get(self.names[i])
        if info is None or info.KLIFS2UniProtIdx is None:
            return None
        return info.KLIFS2UniProtIdx.get(self.position_labels[pos])

    @staticmethod
    def _family_name(value) -> str:
        """Label for a family field, which may be a ``Family`` enum or a bare string."""
        if value is None:
            return ""
        # Family enums expose ``.name``; deserialized fields can arrive as plain strings
        return getattr(value, "name", None) or str(value)

    def _hover_families(self, i: int) -> tuple[str, str]:
        """``(kinhub_family, klifs_family)`` labels for leaf ``i``'s dot hover."""
        info = DICT_KINASE.get(self.names[i])
        return (
            self._family_name(rgetattr(info, "kinhub.family")),
            self._family_name(rgetattr(info, "klifs.family")),
        )

    def _annotation_callouts(self, aa: str) -> list[tuple[list[str], int]]:
        """Find the shared-residue columns for the curated kinases of interest.

        Searches every KLIFS column and, at each, collects the kinases from
        :data:`LIST_RESIDUE_DOT_KINASES` that carry ``aa`` there; a column is called out
        when at least :data:`INT_RESIDUE_DOT_MIN_SHARED` of them share it. There is no
        pre-grouping, so any kinases sharing a homologous residue at one column are boxed
        together, and the callouts are discovered from the current alignment (surviving a
        shift in the KLIFS mapping) rather than hardcoded.

        Parameters
        ----------
        aa : str
            The plotted residue (e.g. ``"C"``).

        Returns
        -------
        list[tuple[list[str], int]]
            ``(shared_kinases, column_index)`` per discovered callout.
        """
        ncol = len(self.position_labels)
        pockets = {}
        for kinase in LIST_RESIDUE_DOT_KINASES:
            pocket = rgetattr(DICT_KINASE.get(kinase), "klifs.pocket_seq")
            if pocket and len(pocket) == ncol:
                pockets[kinase] = pocket

        callouts = []
        for pos in range(ncol):
            shared = [k for k, p in pockets.items() if p[pos] == aa]
            if len(shared) >= INT_RESIDUE_DOT_MIN_SHARED:
                callouts.append((shared, pos))
        return callouts

    def _annotation_lines(self, kinases: list[str], pos: int, label: str) -> list[str]:
        """``"{kinase} {residue}{uniprot_idx}"`` per kinase at KLIFS column ``pos``.

        Used for the curated conserved-residue callout boxes; the residue letter and
        UniProt index are read from each kinase's pocket and ``KLIFS2UniProtIdx`` in
        :data:`DICT_KINASE`, so the numbers stay correct without hardcoding.
        """
        lines = []
        for kinase in kinases:
            info = DICT_KINASE.get(kinase)
            pocket = rgetattr(info, "klifs.pocket_seq")
            residue = pocket[pos] if pocket else "?"
            idx = (info.KLIFS2UniProtIdx or {}).get(label) if info is not None else None
            lines.append(
                f"{kinase} {residue}{idx}" if idx is not None else f"{kinase} {residue}"
            )
        return lines

    def residue_dot_layout(
        self, min_cluster_size: int = INT_TREE_MIN_CLUSTER_DISPLAY
    ) -> tuple[
        KLIFSDisplayTree, list[int], dict[int, int], dict[int, int], list[list[int]]
    ]:
        """Row/leaf assignment for the per-amino-acid dot plot.

        Orders kinases by the display-tree leaf order (dendrogram order), assigning each
        kinase a y row and the 1-based number of the display leaf (clade) it belongs to.
        Shared by the interactive Bokeh explorer and the static companion figure so their
        ordering and clade grouping cannot diverge.

        Parameters
        ----------
        min_cluster_size : int, optional
            Display-tree collapse threshold, by default :data:`INT_TREE_MIN_CLUSTER_DISPLAY`.

        Returns
        -------
        tuple
            ``(tree, ordered_members, row_of_member, leaf_of_member, leaf_members)``:
            ``ordered_members`` are leaf indices (into :attr:`names`) top-to-bottom,
            ``row_of_member`` maps each to its y row, ``leaf_of_member`` to its 1-based
            display-leaf number, and ``leaf_members`` lists members per display leaf in
            display order.
        """
        tree = self.build_display_tree(
            min_cluster_size=min_cluster_size, aggregate_singletons=True
        )
        ordered_members: list[int] = []
        leaf_of_member: dict[int, int] = {}
        leaf_members: list[list[int]] = []
        for number, li in enumerate(tree.order, start=1):
            members = list(tree.leaves[li].members)
            leaf_members.append(members)
            for member in members:
                leaf_of_member[member] = number
                ordered_members.append(member)
        row_of_member = {member: y for y, member in enumerate(ordered_members)}
        return tree, ordered_members, row_of_member, leaf_of_member, leaf_members

    def to_conservation_data(
        self,
        min_cluster_size: int = INT_TREE_MIN_CLUSTER_DISPLAY,
        aggregate_singletons: bool = True,
    ) -> KLIFSConservationData:
        """Package the clustering output as a serializable :class:`KLIFSConservationData`.

        Stores the condensed upper-triangle distances, the SciPy linkage matrix, the
        leaf order, and the display-tree leaf composition (all as plain Python numbers),
        alongside the provenance metadata describing how they were assembled.

        Parameters
        ----------
        min_cluster_size : int, optional
            Display-tree collapse threshold passed to :meth:`build_display_tree`,
            by default :data:`INT_TREE_MIN_CLUSTER_DISPLAY`.
        aggregate_singletons : bool, optional
            Whether to fold singleton leaves in the display tree, by default True.

        Returns
        -------
        KLIFSConservationData
            The serializable conservation-data artifact.
        """
        tree = self.build_display_tree(
            min_cluster_size=min_cluster_size,
            aggregate_singletons=aggregate_singletons,
        )
        return KLIFSConservationData(
            metric=self.metric,
            blosum_name=self.blosum_name,
            linkage_method=self.linkage_method,
            conservation_threshold=self.conservation_threshold,
            weighting=self.weighting,
            exclude_pseudokinases=self.exclude_pseudokinases,
            gap_chars=self.gap_chars,
            n_kinases=len(self.names),
            pocket_length=INT_KLIFS_POCKET_LENGTH,
            names=list(self.names),
            position_labels=list(self.position_labels),
            distances_condensed=squareform(self.distance_matrix, checks=False).tolist(),
            linkage_matrix=self.linkage_matrix.tolist(),
            leaves_order=leaves_list(self.linkage_matrix).tolist(),
            display_leaves=[
                {"members": list(leaf.members), "kind": leaf.kind}
                for leaf in tree.leaves
            ],
            display_order=list(tree.order),
        )


def load_conservation_renderer(cls, **kwargs):
    """Build a conservation renderer from the shipped :class:`KLIFSConservationData`.

    Deserializes the packaged artifact (via
    :func:`mkt.schema.io_utils.load_conservation_data`) and injects its distances +
    linkage into ``cls`` through :meth:`KLIFSHierarchicalConservation.from_conservation_data`,
    so no clustering is recomputed. If the artifact is not found, falls back to building
    ``cls`` from the live KLIFS panel with a warning (run ``generate_conservation_data``
    to persist it).

    Parameters
    ----------
    cls : type[KLIFSHierarchicalConservation]
        The renderer (sub)class to instantiate.
    **kwargs
        Renderer-specific keyword arguments forwarded to the constructor.

    Returns
    -------
    KLIFSHierarchicalConservation
        An instance of ``cls``.
    """
    try:
        data = load_conservation_data(str_name="KLIFSConservationData")
        return cls.from_conservation_data(data, **kwargs)
    except FileNotFoundError:
        logger.warning(
            "no persisted KLIFSConservationData found; building from the live panel "
            "(run generate_conservation_data to persist it)"
        )
        return cls(**kwargs)


def rebuild_tree_from_data(data: KLIFSConservationData) -> ClusterNode:
    """Reconstruct the SciPy hierarchical tree from a persisted artifact.

    The scipy bridge for :class:`KLIFSConservationData`: consumers holding only the
    schema (which is scipy-free) call this in ``mkt.databases`` to recover the full
    ``ClusterNode`` tree from the stored linkage matrix.

    Parameters
    ----------
    data : KLIFSConservationData
        The persisted conservation-data artifact.

    Returns
    -------
    scipy.cluster.hierarchy.ClusterNode
        Root node of the reconstructed hierarchical tree.
    """
    return to_tree(np.array(data.linkage_matrix), rd=False)


class KLIFSConservationTreeFigure(KLIFSHierarchicalConservation):
    """Static letter-page figure of the KLIFS conservation tree + per-leaf table.

    The matplotlib sibling of the interactive Bokeh explorer
    (:class:`mkt_impact.analysis.interactive.KLIFSTreeConservationApp`): a thin
    multifurcating dendrogram of the human kinome (singleton-peel chains contracted,
    singletons aggregated by tree adjacency so no leaf row is a single sequence),
    homolog-collapsed leaf name boxes, and a per-leaf table rendering each row's >=80%
    consensus residue at all 85 KLIFS positions colored by the amino-acid alphabet
    palette. Both views consume the one shared :meth:`build_display_tree` /
    :meth:`members_consensus`, so the tree topology cannot diverge between them.

    Panel-only :class:`KLIFSTreeView`: construct standalone (defaults to the full
    human-kinome KLIFS panel) or via ``from_dataloader`` in the pipeline, then call
    :meth:`build_figure` (returns the figure) or :meth:`plot` (saves svg/png/pdf to the
    structured output tree).
    """

    model_config = {"arbitrary_types_allowed": True}

    min_cluster_size: int = INT_TREE_MIN_CLUSTER_DISPLAY
    """Gathered subtrees with fewer members collapse to a single leaf bar."""
    font_size: float = FLOAT_TREE_FONT_SIZE
    """Base font size; raising it widens the (measured) name boxes and narrows the table."""
    split_index: int | None = INT_TREE_SPLIT_INDEX
    """Leaf-order index splitting the top/bottom detail panels. Defaults to the curated
    CMGC/CAMK boundary of the human-kinome KLIFS tree (:data:`INT_TREE_SPLIT_INDEX`).
    Set to None (or an out-of-range value) to auto-pick the group boundary nearest the
    midpoint (:meth:`_split_index`)."""

    # -- per-leaf styling (family color, pseudokinase border, branch color) --

    def _dominant_color(self, members: list[int]) -> str:
        """Branch color: the family color if >=80% of members share it, else mixed-grey."""
        fam, k = Counter(self._family_label(m) for m in members).most_common(1)[0]
        if k / len(members) >= FLOAT_TREE_DOMINANT_FRACTION:
            return DICT_KINASE_GROUP_COLORS.get(fam, COLOR_TREE_FALLBACK)
        return COLOR_TREE_MIXED_FAMILY

    def _partition_key(self, i: int) -> tuple:
        """Key separating homolog lumping by lipid / pseudokinase / Manning group."""
        info = DICT_KINASE.get(self.names[i])
        lipid = info is not None and info.is_lipid_kinase()
        pseudo = info is not None and info.is_pseudokinase()
        return (True, None) if lipid else (False, pseudo, self.groups[i])

    def _leaf_groups(self, members: list[int]) -> list[tuple[str, int]]:
        """Collapse a leaf row's members into ``(label, representative_index)`` boxes.

        Members are partitioned by family/pseudo/lipid key first (so e.g. a pseudo
        ERBB3 stays apart), then name-homologs within each partition are collapsed via
        the shared :func:`mkt.schema.utils.group_name_homologs`.
        """
        parts: dict[tuple, list[int]] = defaultdict(list)
        for m in members:
            parts[self._partition_key(m)].append(m)
        groups: list[tuple[str, int]] = []
        for key in sorted(parts, key=str):
            name2member = {self.names[m]: m for m in parts[key]}
            for label, gnames in group_name_homologs(
                [self.names[m] for m in parts[key]], show_count=False
            ):
                groups.append((label, name2member[gnames[0]]))
        return groups

    def _group_legend_handles(self, fams: set[str]) -> list[Patch]:
        """Kinase-group legend handles, ordered by the standard Manning-group order.

        Shared by the summary dendrogram and the top/bottom detail panels so their
        kinase-group legends match. Families outside the curated order are appended
        alphabetically.

        Parameters
        ----------
        fams : set[str]
            Family labels present in the drawn subset (Manning group or Lipid).

        Returns
        -------
        list[matplotlib.patches.Patch]
            One color swatch per present family, in display order.
        """
        fam_order = [
            "TK",
            "TKL",
            "STE",
            "CK1",
            "AGC",
            "CAMK",
            "CMGC",
            "NEK",
            "RGC",
            "Other",
            "Atypical",
            "Lipid",
        ]
        present = [g for g in fam_order if g in fams] + sorted(fams - set(fam_order))
        return [
            Patch(
                facecolor=DICT_KINASE_GROUP_COLORS.get(g, COLOR_TREE_FALLBACK), label=g
            )
            for g in present
        ]

    def build_figure(self) -> tuple[plt.Figure, plt.Axes]:
        """Render the full conservation tree + per-leaf table on a US-letter portrait page.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
            The figure and its single (axis-off) axes, sized to :data:`TUP_TREE_PAGE`.
        """
        tree = self.build_display_tree(
            min_cluster_size=self.min_cluster_size, aggregate_singletons=True
        )
        return self._render_panel(tree, tree.order, TUP_TREE_PAGE, self.font_size)

    def _render_panel(self, tree, subset, page, fs, show_legend=False):
        """Render a dendrogram + name boxes + conservation table for a leaf subset.

        Shared by the full figure and each split detail panel. ``subset`` is a
        contiguous slice of ``tree.order``; the dendrogram is pruned to (and rooted,
        via relative depth, at) the subset's spanning subtree, the x-axis is laid out in
        inches so name boxes never overhang, the KLIFS table fills the remaining width,
        and the top margin is sized to the tallest rotated KLIFS label so there is no
        slack.

        Parameters
        ----------
        tree : KLIFSDisplayTree
            The display tree (from :meth:`build_display_tree`).
        subset : list[int]
            Leaf indices (into ``tree.leaves``) to draw, in display order.
        page : tuple[float, float]
            Figure size in inches.
        fs : float
            Base font size for the boxes / table / KLIFS labels.
        show_legend : bool, optional
            If True, reserve a bottom band and draw the kinase-group legend beneath the
            panel (used by the top/bottom detail panels; the caller must have added
            :data:`FLOAT_TREE_DETAIL_LEGEND_IN` to ``page`` height). Default False.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        """
        splits, leaves = tree.splits, tree.leaves
        sub = list(subset)
        yof = {li: i for i, li in enumerate(sub)}
        n_rows = len(sub)
        labels = self.position_labels
        n_cols = len(labels)
        ch = FLOAT_TREE_ROW_HEIGHT

        for li in sub:
            leaves[li].y = float(yof[li])

        # split y-centers over in-subset descendants only (None if a split has none)
        memo: dict[int, float | None] = {}

        def split_yc(sid):
            if sid in memo:
                return memo[sid]
            ys = []
            for k, r in splits[sid].children:
                if k == "leaf":
                    if r in yof:
                        ys.append(leaves[r].y)
                else:
                    cy = split_yc(r)
                    if cy is not None:
                        ys.append(cy)
            memo[sid] = sum(ys) / len(ys) if ys else None
            return memo[sid]

        for s in splits:
            split_yc(s.id)
        in_splits = [s.id for s in splits if memo[s.id] is not None]
        base_depth = min(splits[i].depth for i in in_splits)
        max_depth = max(splits[i].depth for i in in_splits) - base_depth

        for li in sub:
            leaves[li].groups = self._leaf_groups(leaves[li].members)

        fig = plt.figure(figsize=page, facecolor="white")
        # reserve a bottom band for the group legend when requested, so the inch-based
        # layout below (which reads ax.get_position()) shrinks to fit rather than overlap
        legend_frac = FLOAT_TREE_DETAIL_LEGEND_IN / page[1] if show_legend else 0.0
        ax = fig.add_axes([0.008, 0.008 + legend_frac, 0.984, 0.984 - legend_frac])
        ax.set_facecolor("white")

        # x-axis laid out in INCHES: size name boxes from the rendered text so they
        # never overhang, then let the KLIFS table fill the remaining width
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        wcache: dict[str, float] = {}

        def text_width_in(label):
            if label not in wcache:
                t = ax.text(0, 0, label, fontsize=fs)
                wcache[label] = t.get_window_extent(renderer=renderer).width / fig.dpi
                t.remove()
            return wcache[label]

        axes_w = ax.get_position().width * page[0]
        for li in sub:
            leaf = leaves[li]
            leaf.box_w = [
                text_width_in(lbl) + 2 * FLOAT_TREE_BOX_PAD_IN for lbl, _ in leaf.groups
            ]
            leaf.width = sum(leaf.box_w) + FLOAT_TREE_BOX_GAP_IN * len(leaf.box_w)
        max_bar = max(leaves[i].width for i in sub)

        x_leaf = (max_depth + 1) * FLOAT_TREE_DEPTH_STEP_IN
        x_table = x_leaf + max_bar + FLOAT_TREE_TABLE_GAP_IN
        colw = (axes_w - x_table - FLOAT_TREE_RIGHT_PAD_IN) / n_cols
        logger.info(
            "conservation-tree panel: %d rows, dendro %.2fin, widest bar %.2fin, "
            "col %.3fin over %d KLIFS positions",
            n_rows,
            x_leaf,
            max_bar,
            colw,
            n_cols,
        )

        self._draw_dendrogram(ax, splits, leaves, memo, base_depth, x_leaf, yof)
        self._draw_name_boxes(ax, leaves, sub, x_leaf, ch, fs)
        self._draw_table(ax, leaves, sub, labels, x_table, colw, ch, fs)

        ax.set_xlim(-0.05, axes_w)
        # size the top margin to the tallest rotated KLIFS label (its vertical extent is
        # the unrotated text width) so there is no slack above the header
        axes_h = ax.get_position().height * page[1]
        max_lbl_in = max(text_width_in(lb) for lb in labels)
        yrange = n_rows + 4.0
        for _ in range(3):  # fixed-point: top margin <-> y-scale
            y_top = -1.4 - max_lbl_in / (axes_h / yrange) - 0.2
            yrange = (n_rows + 0.5) - y_top
        ax.set_ylim(y_top, n_rows + 0.5)
        ax.invert_yaxis()
        ax.axis("off")

        # kinase-group legend in the reserved bottom band (matches the summary panel)
        if show_legend:
            fams = {self._family_label(m) for li in sub for m in leaves[li].members}
            handles = self._group_legend_handles(fams)
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.005),
                ncol=len(handles),
                frameon=False,
                fontsize=FLOAT_TREE_DETAIL_LEGEND_FS,
                handlelength=1.4,
                handletextpad=0.5,
                columnspacing=1.3,
            )
        return fig, ax

    def _draw_dendrogram(self, ax, splits, leaves, split_yc, base_depth, x_leaf, yof):
        """Draw the thin multifurcating dendrogram, pruned to in-subset branches and
        rooted (relative depth) at the subset's spanning subtree."""
        dx = FLOAT_TREE_DEPTH_STEP_IN
        for s in splits:
            if split_yc[s.id] is None:
                continue
            px = (s.depth - base_depth) * dx
            kids_y = []
            for kind, ref in s.children:
                ky = (
                    leaves[ref].y
                    if (kind == "leaf" and ref in yof)
                    else (split_yc[ref] if kind == "node" else None)
                )
                if ky is None:
                    continue
                kids_y.append(ky)
                cx = x_leaf if kind == "leaf" else (splits[ref].depth - base_depth) * dx
                members = leaves[ref].members if kind == "leaf" else splits[ref].members
                ax.plot(
                    [px, cx],
                    [ky, ky],
                    color=self._dominant_color(members),
                    lw=FLOAT_TREE_BRANCH_LW,
                    zorder=1,
                )
            if kids_y:
                ax.plot(
                    [px, px],
                    [min(kids_y), max(kids_y)],
                    color=self._dominant_color(s.members),
                    lw=FLOAT_TREE_BRANCH_LW,
                    zorder=1,
                )

    def _draw_name_boxes(self, ax, leaves, order, x_leaf, ch, fs):
        """Draw the homolog-collapsed leaf name boxes (measured widths)."""
        for leaf_idx in order:
            leaf = leaves[leaf_idx]
            y = leaf.y
            x = x_leaf
            for (label, i0), w in zip(leaf.groups, leaf.box_w):
                fill, edge, lw = self._member_style(i0)
                ax.add_patch(
                    Rectangle(
                        (x, y - ch / 2),
                        w,
                        ch,
                        facecolor=fill,
                        edgecolor=edge,
                        linewidth=lw,
                        zorder=3,
                    )
                )
                ax.text(
                    x + w / 2,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color=readable_text_color(fill),
                    zorder=4,
                )
                x += w + FLOAT_TREE_BOX_GAP_IN

    def _table_consensus(self, members: list[int]) -> list[str | None]:
        """Per-column >=threshold consensus INCLUDING gaps.

        Returns, per KLIFS column, an amino acid, a gap character (when a gap is the
        >=threshold consensus), or None when nothing reaches the threshold. Counting
        gaps means an all-but-absent pocket position reads as a gap rather than
        borrowing the consensus of the handful of members that do carry a residue.
        """
        thr = self.conservation_threshold
        out: list[str | None] = []
        for col in range(len(self.position_labels)):
            chars = [self.pockets[m][col] for m in members]
            char, cnt = Counter(chars).most_common(1)[0]
            out.append(char if cnt / len(chars) >= thr else None)
        return out

    def _draw_table(self, ax, leaves, order, labels, x_table, colw, ch, fs):
        """Draw the per-leaf >=80% consensus table with the KLIFS region header."""
        aa_pal = DICT_COLORS[AminoAcidPalette.ALPHABET_PROJECT.value]["DICT_COLORS"]
        n_cols = len(labels)
        # pan-kinome invariants (>=80% across the whole panel) get a black header border
        pan = {
            c
            for c, aa in enumerate(
                self.members_consensus(self.tree_members(2 * len(self.names) - 2))
            )
            if aa is not None
        }

        def region_color(col):
            return DICT_POCKET_KLIFS_REGIONS[labels[col].split(":")[0]]["color"]

        def aa_color(aa):
            return map_single_letter_aa_to_color(aa, aa_pal) or "#DDDDDD"

        # leaf -> table guide lines, behind everything
        for leaf_idx in order:
            yc = leaves[leaf_idx].y
            ax.plot(
                [0, x_table + n_cols * colw],
                [yc, yc],
                color=COLOR_TREE_GUIDE,
                lw=FLOAT_TREE_GUIDE_LW,
                zorder=0,
            )

        # header: KLIFS position label on top, then the region color bar (black-bordered
        # for pan-kinome invariants)
        for col in range(n_cols):
            cx = x_table + col * colw
            invariant = col in pan
            ax.add_patch(
                Rectangle(
                    (cx, -1.1),
                    colw * 0.92,
                    0.55,
                    facecolor=region_color(col),
                    edgecolor="black" if invariant else "white",
                    linewidth=0.6 if invariant else 0.25,
                    zorder=2 if invariant else 1,
                )
            )
            ax.text(
                cx + colw * 0.46,
                -1.4,
                labels[col],
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=fs,
            )

        # cells: each leaf's >=80% consensus -- amino acid, a gap ('-' on mid grey), or
        # light grey when nothing reaches the threshold
        for leaf_idx in order:
            leaf = leaves[leaf_idx]
            y = leaf.y
            for col, aa in enumerate(self._table_consensus(leaf.members)):
                cx = x_table + col * colw
                if aa is None:
                    ax.add_patch(
                        Rectangle(
                            (cx, y - ch / 2),
                            colw * 0.92,
                            ch,
                            facecolor=COLOR_TREE_TABLE_NO_CONSENSUS,
                            edgecolor="white",
                            linewidth=0.2,
                        )
                    )
                    continue
                if aa in self.gap_chars:
                    ax.add_patch(
                        Rectangle(
                            (cx, y - ch / 2),
                            colw * 0.92,
                            ch,
                            facecolor=COLOR_TREE_TABLE_GAP,
                            edgecolor="white",
                            linewidth=0.2,
                        )
                    )
                    ax.text(
                        cx + colw * 0.46,
                        y,
                        "-",
                        ha="center",
                        va="center",
                        fontsize=fs,
                        color=readable_text_color(COLOR_TREE_TABLE_GAP),
                    )
                    continue
                fill = aa_color(aa)
                ax.add_patch(
                    Rectangle(
                        (cx, y - ch / 2),
                        colw * 0.92,
                        ch,
                        facecolor=fill,
                        edgecolor="white",
                        linewidth=0.2,
                    )
                )
                ax.text(
                    cx + colw * 0.46,
                    y,
                    aa,
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color=readable_text_color(fill),
                )

    def plot(
        self, output_path: str, formats: tuple[str, ...] = ("svg", "png", "pdf")
    ) -> None:
        """Render and save the single full-page figure to ``output_path``.

        Writes ``{STR_FILE_CONSERVATION_TREE}.{ext}`` into ``output_path`` for each
        requested format. No tight bbox, so the page stays exactly :data:`TUP_TREE_PAGE`.

        Parameters
        ----------
        output_path : str
            Directory to write the figure into.
        formats : tuple[str, ...], optional
            File extensions to write, by default ``("svg", "png", "pdf")``.
        """
        fig, _ = self.build_figure()
        save_plot(
            fig,
            STR_FILE_CONSERVATION_TREE,
            plot_type="KLIFS conservation tree",
            bool_force_local=False,
            bool_image_subdir=False,
            output_path=output_path,
            bool_svg="svg" in formats,
            bool_png="png" in formats,
            bool_pdf="pdf" in formats,
            dpi=300,
            bbox_inches=None,
            facecolor="white",
        )

    # -- split supplemental figures (summary + top / bottom detail panels) --

    def _split_index(self, tree: KLIFSDisplayTree) -> int:
        """Leaf-order index to split the detail panels: the kinase-group boundary
        nearest the midpoint (so each panel is roughly half and starts a new group)."""
        order, leaves = tree.order, tree.leaves

        def dom(li):
            return Counter(
                self._family_label(m) for m in leaves[li].members
            ).most_common(1)[0][0]

        fams = [dom(li) for li in order]
        boundaries = [i for i in range(1, len(order)) if fams[i] != fams[i - 1]]
        mid = len(order) / 2
        return (
            min(boundaries, key=lambda i: abs(i - mid))
            if boundaries
            else len(order) // 2
        )

    def _build_summary_panel(
        self, tree: KLIFSDisplayTree, split_index: int
    ) -> plt.Figure:
        """Horizontal (top-down) overview dendrogram: leaves along x as family-colored
        blocks, depth down the y-axis (root at top), split marked, group legend below.
        """
        splits, leaves, order = tree.splits, tree.leaves, tree.order
        n = len(order)
        xof = {li: i for i, li in enumerate(order)}
        memo: dict[int, float] = {}

        def split_x(sid):
            if sid in memo:
                return memo[sid]
            xs = [
                float(xof[r]) if k == "leaf" else split_x(r)
                for k, r in splits[sid].children
            ]
            memo[sid] = sum(xs) / len(xs)
            return memo[sid]

        for s in splits:
            split_x(s.id)
        max_depth = max(s.depth for s in splits)
        leaf_y = max_depth + 1

        fig = plt.figure(figsize=TUP_TREE_SUMMARY_PAGE, facecolor="white")
        ax = fig.add_axes([0.01, 0.11, 0.98, 0.87])
        for s in splits:
            py = s.depth
            cxs = [float(xof[r]) if k == "leaf" else memo[r] for k, r in s.children]
            ax.plot(
                [min(cxs), max(cxs)],
                [py, py],
                color=self._dominant_color(s.members),
                lw=1.1,
                zorder=1,
            )
            for k, r in s.children:
                cx = float(xof[r]) if k == "leaf" else memo[r]
                cyd = leaf_y if k == "leaf" else splits[r].depth
                mem = leaves[r].members if k == "leaf" else splits[r].members
                ax.plot(
                    [cx, cx],
                    [py, cyd],
                    color=self._dominant_color(mem),
                    lw=1.1,
                    zorder=1,
                )

        for li in order:
            x = float(xof[li])
            fill, _, _ = self._member_style(leaves[li].members[0])
            ax.add_patch(
                Rectangle(
                    (x - 0.46, leaf_y),
                    0.92,
                    1.0,
                    facecolor=fill,
                    edgecolor="white",
                    linewidth=0.3,
                    zorder=3,
                )
            )

        ax.axvline(
            split_index - 0.5,
            color=COLOR_TREE_SPLIT_MARKER,
            lw=1.0,
            ls=(0, (5, 3)),
            zorder=5,
        )

        fams = set()
        for li in order:
            fams.update(self._family_label(m) for m in leaves[li].members)
        handles = self._group_legend_handles(fams)
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=len(handles),
            frameon=False,
            fontsize=FLOAT_TREE_SUMMARY_LEGEND_FS,
            handlelength=1.0,
            handletextpad=0.4,
            columnspacing=1.3,
        )

        ax.set_xlim(-1, n)
        ax.set_ylim(-0.5, leaf_y + 1.5)
        ax.invert_yaxis()
        ax.axis("off")
        return fig

    def build_split_figures(
        self, split_index: int | None = None
    ) -> list[tuple[str, plt.Figure]]:
        """Build the three supplemental figures: horizontal summary + top / bottom
        detail panels, split at ``split_index`` (default: :data:`INT_TREE_SPLIT_INDEX`,
        the CMGC/CAMK boundary; falls back to :meth:`_split_index` when the resolved
        index is None or out of range for the current panel).

        Returns
        -------
        list[tuple[str, matplotlib.figure.Figure]]
            ``(output_basename, figure)`` for the summary, top, and bottom panels.
        """
        tree = self.build_display_tree(
            min_cluster_size=self.min_cluster_size, aggregate_singletons=True
        )
        if split_index is None:
            split_index = self.split_index
        # fall back to the auto midpoint boundary for a None or out-of-range hardcode
        # (e.g. a non-standard / reduced panel where the curated index no longer applies)
        if split_index is None or not 0 < split_index < len(tree.order):
            split_index = self._split_index(tree)
        top, bottom = tree.order[:split_index], tree.order[split_index:]

        def detail(sub):
            page = (
                FLOAT_TREE_DETAIL_PAGE_W,
                max(7.0, len(sub) * FLOAT_TREE_DETAIL_ROW_IN)
                + FLOAT_TREE_DETAIL_LEGEND_IN,
            )
            return self._render_panel(
                tree, sub, page, FLOAT_TREE_SPLIT_FONT_SIZE, show_legend=True
            )[0]

        return [
            (
                STR_FILE_CONSERVATION_TREE_SUMMARY,
                self._build_summary_panel(tree, split_index),
            ),
            (STR_FILE_CONSERVATION_TREE_TOP, detail(top)),
            (STR_FILE_CONSERVATION_TREE_BOTTOM, detail(bottom)),
        ]

    def plot_split(
        self,
        output_path: str,
        formats: tuple[str, ...] = ("pdf",),
        split_index: int | None = None,
    ) -> None:
        """Render and save the three split supplemental figures to ``output_path``.

        Writes ``{basename}.{ext}`` into ``output_path`` for each format (PDF by
        default, as these are supplemental materials).

        Parameters
        ----------
        output_path : str
            Directory to write the figures into.
        formats : tuple[str, ...], optional
            File extensions to write, by default ``("pdf",)``.
        split_index : int | None, optional
            Leaf-order split index; default auto (:meth:`_split_index`).
        """
        for basename, fig in self.build_split_figures(split_index=split_index):
            save_plot(
                fig,
                basename,
                plot_type=basename,
                bool_force_local=False,
                bool_image_subdir=False,
                output_path=output_path,
                bool_svg="svg" in formats,
                bool_png="png" in formats,
                bool_pdf="pdf" in formats,
                dpi=300,
                bbox_inches=None,
                facecolor="white",
            )

    # -- static per-amino-acid dot-plot companion (see KLIFSResidueDotApp) --

    def build_residue_dot_figure(
        self,
        aa: str = STR_RESIDUE_DOT_DEFAULT_AA,
        figsize: tuple[float, float] | None = None,
        highlight_targets: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Static companion to :class:`KLIFSResidueDotApp` for a single amino acid.

        A frequency dot plot: within each of the 85 KLIFS columns (x) the kinases
        carrying ``aa`` are stacked one dot per kinase (y = count) in dendrogram leaf
        order, so empty columns simply have no dots. Dots are colored by Manning group /
        Lipid (black outline for pseudokinases); runs of
        :data:`INT_RESIDUE_DOT_MIN_ENCLOSE`+ consecutive same-clade dots within a column
        are enclosed in a light-grey rounded box. A KLIFS region color bar and rotated
        position labels run beneath, and the kinase-group legend sits below.

        Parameters
        ----------
        aa : str
            Amino-acid single-letter code (or ``"-"``) to plot, by default cysteine.
        figsize : tuple[float, float] | None, optional
            Figure size; by default sized to the tallest column stack.
        highlight_targets : bool, optional
            If True, draw the curated conserved-cysteine callout boxes for the kinases in
            :data:`LIST_RESIDUE_DOT_KINASES` -- a labeled box of kinases + residue numbers
            with an arrow to each shared-residue column found by searching the alignment
            (:meth:`_annotation_callouts`) -- by default False.

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        """
        (
            _tree,
            ordered_members,
            _row_of_member,
            leaf_of_member,
            _leaf_members,
        ) = self.residue_dot_layout(min_cluster_size=self.min_cluster_size)
        labels = self.position_labels
        ncol = len(labels)

        # per-column stacks in dendrogram leaf order (y = count); collect dots +
        # same-clade enclosure runs
        xs, ys, fills, lines = [], [], [], []
        enclosures = []
        col_count = {}
        max_count = 0
        for pos in range(ncol):
            stack = [i for i in ordered_members if self.pockets[i][pos] == aa]
            col_count[pos] = len(stack)
            max_count = max(max_count, len(stack))
            for height, i in enumerate(stack):
                fill, edge, _ = self._member_style(i)
                xs.append(pos)
                ys.append(height)
                fills.append(fill)
                lines.append(COLOR_TREE_PSEUDO if edge == "#000000" else fill)
            start = 0
            for end in range(1, len(stack) + 1):
                same = (
                    end < len(stack)
                    and leaf_of_member[stack[end]] == leaf_of_member[stack[start]]
                )
                if not same:
                    if end - start >= INT_RESIDUE_DOT_MIN_ENCLOSE:
                        lo, hi = start, end - 1
                        enclosures.append((pos, (lo + hi) / 2.0, (hi - lo) + 1.0))
                    start = end

        # wide, low-aspect figure sized to the placement slot; a fixed strip below
        # carries the KLIFS region bar + rotated labels
        fig_w = 15.5
        count_in = 0.04  # inches of dot-panel height per stacked kinase
        region_in = 1.0  # inches reserved for the region bar + rotated labels
        dot_in = max(2.5, max_count * count_in)
        if figsize is None:
            figsize = (fig_w, dot_in + region_in + 0.3)
        fig, (ax, axr) = plt.subplots(
            2,
            1,
            figsize=figsize,
            facecolor="white",
            sharex=True,
            height_ratios=[dot_in, region_in],
            layout="constrained",
        )
        fig.get_layout_engine().set(hspace=0.0, h_pad=0.0)
        ax.set_facecolor("white")

        # marker size: with the panel height fixed, let dots grow to slightly overlap
        # vertically (the wide layout leaves ample horizontal room) so the stacks read
        # bolder / more prominent without making the figure taller
        pts_per_count = dot_in * 72.0 / (max_count + 2)
        marker_s = float(min(60.0, max(9.0, (1.15 * pts_per_count) ** 2)))

        # clade enclosures (light-grey rounded rectangles, drawn under the dots)
        for x, y_center, height in enclosures:
            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.32, y_center - height / 2.0),
                    0.64,
                    height,
                    boxstyle="round,pad=0,rounding_size=0.18",
                    facecolor=COLOR_TREE_TABLE_NO_CONSENSUS,
                    edgecolor=COLOR_TREE_PSEUDO,
                    lw=0.6,
                    zorder=1,
                )
            )
        ax.scatter(
            xs, ys, s=marker_s, c=fills, edgecolors=lines, linewidths=0.3, zorder=4
        )

        # curated conserved-cysteine callouts: a labeled box (kinase + residue number)
        # per column found by searching the alignment. Boxes sit on their own column and
        # are bottom-aligned at a common height (~3/4 up); a box whose span would cover a
        # taller neighbouring stack (e.g. hinge:47 next to the tall hinge:48) is lifted to
        # just above that stack instead, so it never clashes with the dots.
        callouts = sorted(
            self._annotation_callouts(aa) if highlight_targets else [],
            key=lambda kp: kp[1],
        )
        anno_fontsize = 13
        base_top = max(max_count + 1, 90)
        tail_y = 0.72 * base_top
        pt_per_unit = fig_w * 0.92 * 72.0 / ncol  # x-axis points per data column

        boxes, y_top = [], base_top
        for kinases, pos in callouts:
            lines = self._annotation_lines(kinases, pos, labels[pos])
            max_chars = max((len(s) for s in lines), default=1)
            half = (max_chars * 0.6 + 0.8) * anno_fontsize / pt_per_unit / 2.0
            lo, hi = max(0, round(pos - half)), min(ncol - 1, round(pos + half))
            tallest = max(col_count[c] for c in range(lo, hi + 1))
            box_bottom = tail_y if tallest <= tail_y else tallest + 4.0
            boxes.append((pos, "\n".join(lines), box_bottom))
            # ~7.5 counts per text line + padding, to keep the box below the axis top
            y_top = max(y_top, box_bottom + len(lines) * 7.5 + 6.0)

        for pos, text, box_bottom in boxes:
            # outline the box in its KLIFS region color for clarity
            region_edge = DICT_POCKET_KLIFS_REGIONS[labels[pos].split(":")[0]]["color"]
            ax.annotate(
                text,
                xy=(pos, col_count[pos] + 1.0),  # arrowhead at the stack top
                xytext=(
                    pos,
                    box_bottom,
                ),  # box bottom-aligned at the common tail height
                ha="center",
                va="bottom",
                fontsize=anno_fontsize,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    edgecolor=region_edge,
                    linewidth=1.5,
                ),
                arrowprops=dict(
                    arrowstyle="-|>", color="black", lw=2.0, mutation_scale=20
                ),
                annotation_clip=False,
                zorder=6,
            )

        ax.set_xlim(-0.6, ncol - 0.4)
        # extra bottom margin so the y=0 dots are not clipped; y ticks 0-90 by 15
        ax.set_ylim(-1.5, y_top)
        ax.set_yticks(list(range(0, 91, 15)))
        ax.tick_params(axis="y", labelsize=14)
        ax.set_xticks([])
        remove_spines(ax)
        ax.set_ylabel("Count", fontsize=17)

        # region bar (top of strip) + rotated labels below it, in a blended transform
        # (x = data column, y = axes fraction) so label height is independent of scale
        xtrans = axr.get_xaxis_transform()
        for pos, label in enumerate(labels):
            region_color = DICT_POCKET_KLIFS_REGIONS[label.split(":")[0]]["color"]
            axr.add_patch(
                Rectangle(
                    (pos - 0.5, 0.88),
                    1,
                    0.12,
                    facecolor=region_color,
                    edgecolor="white",
                    linewidth=0.4,
                    transform=xtrans,
                    clip_on=False,
                )
            )
            axr.text(
                pos,
                0.83,
                label,
                rotation=90,
                ha="center",
                va="top",
                fontsize=11,
                transform=xtrans,
            )
        axr.set_ylim(0.0, 1.0)
        axr.axis("off")

        fams = {self._family_label(m) for m in ordered_members}
        # kinase-group swatches plus the pseudokinase-outline and clade-enclosure keys
        pseudo_handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLOR_TREE_PSEUDO,
            markeredgewidth=1.3,
            markersize=9,
            label="Pseudokinase",
        )
        shared_handle = Patch(
            facecolor=COLOR_TREE_TABLE_NO_CONSENSUS,
            edgecolor=COLOR_TREE_PSEUDO,
            linewidth=0.8,
            label="Shared kinase family",
        )
        handles = self._group_legend_handles(fams) + [pseudo_handle, shared_handle]
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=len(handles),
            frameon=False,
            fontsize=12,
            handlelength=1.0,
            handletextpad=0.4,
            columnspacing=1.3,
            handler_map={shared_handle: _HandlerRoundedBox()},
        )
        return fig, ax

    def plot_residue_dot(
        self,
        output_path: str,
        aa: str = STR_RESIDUE_DOT_DEFAULT_AA,
        formats: tuple[str, ...] = ("pdf",),
        highlight_targets: bool = False,
    ) -> None:
        """Render and save the static per-amino-acid dot-plot figure to ``output_path``.

        Parameters
        ----------
        output_path : str
            Directory to write the figure into.
        aa : str
            Amino-acid single-letter code (or ``"-"``) to plot, by default cysteine.
        formats : tuple[str, ...], optional
            File extensions to write, by default ``("pdf",)``.
        highlight_targets : bool, optional
            If True, draw target arrows beneath the conserved inhibitor-targetable
            columns, by default False.
        """
        fig, _ = self.build_residue_dot_figure(
            aa=aa, highlight_targets=highlight_targets
        )
        save_plot(
            fig,
            STR_FILE_RESIDUE_DOT,
            plot_type="KLIFS residue dot plot",
            bool_force_local=False,
            bool_image_subdir=False,
            output_path=output_path,
            bool_svg="svg" in formats,
            bool_png="png" in formats,
            bool_pdf="pdf" in formats,
            dpi=300,
            facecolor="white",
        )


class KLIFSTreeConservationApp(KLIFSHierarchicalConservation):
    """Interactive standalone-HTML Bokeh explorer for the KLIFS conservation tree.

    Renders the agglomerative similarity dendrogram with singleton-peel chains
    contracted into multifurcating nodes (so a kinase that peels off attaches
    directly to its clade), above a per-node conservation logo that updates on tap.
    In the logo, letter height is the consensus frequency and color encodes novelty
    relative to the parent clade (region color = newly conserved/changed, black =
    inherited, grey = below threshold). Tree branches are colored by family while a
    clade is monophyletic (lipid > pseudokinase > Manning group) and grey once mixed;
    split nodes are filled by whether they carry a fixed-difference breakpoint. All
    per-node data is precomputed and embedded, so the HTML needs no server.

    Panel-only view built on :class:`KLIFSHierarchicalConservation` (no cohort data); ``save_app`` writes a
    self-contained HTML file to the structured output tree.
    """

    model_config = {"arbitrary_types_allowed": True}

    min_cluster_size: int = INT_TREE_MIN_CLUSTER_DISPLAY
    """Subtrees with fewer members collapse to a single tappable bar."""
    logo_cutoff: float = FLOAT_TREE_LOGO_CUTOFF
    """Minimum consensus frequency for a residue to appear in the per-node logo."""
    name_trunc: int = INT_TREE_NAME_TRUNC
    """Max items shown in a tooltip field before truncation."""

    def _family_types(self) -> tuple[list[str], dict[int, str | None], dict[str, str]]:
        """Per-leaf family label and per-node subtree purity, with a color map.

        Returns:
        --------
        tuple[list[str], dict[int, str | None], dict[str, str]]
            ``(leaf_types, node_type, type_colors)`` where ``leaf_types[i]`` is the
            leaf's family (lipid > pseudokinase > Manning group), ``node_type[nid]``
            is the subtree's pure family or ``None`` if mixed, and ``type_colors``
            maps a family to its color.
        """
        type_colors = dict(DICT_KINASE_GROUP_COLORS)
        type_colors["Pseudo"] = COLOR_TREE_PSEUDO
        n_leaves = len(self.names)

        def leaf_type(i):
            info = DICT_KINASE.get(self.names[i])
            if info is not None:
                if info.is_lipid_kinase():
                    return "Lipid"
                if info.is_pseudokinase():
                    return "Pseudo"
            return self.groups[i]

        leaf_types = [leaf_type(i) for i in range(n_leaves)]
        node_type = {leaf: leaf_types[leaf] for leaf in range(n_leaves)}
        for i in range(self.linkage_matrix.shape[0]):
            t1 = node_type[int(self.linkage_matrix[i, 0])]
            t2 = node_type[int(self.linkage_matrix[i, 1])]
            node_type[n_leaves + i] = t1 if t1 == t2 else None
        return leaf_types, node_type, type_colors

    def build_layout(self):
        """Assemble the Bokeh layout (logo, legend, dendrogram) with tap interactivity.

        Returns:
        --------
        bokeh.models.LayoutDOM
            The column layout ready for :func:`bokeh.embed.file_html`.
        """
        Z = self.linkage_matrix
        labels = self.position_labels
        ncol = len(labels)
        n_leaves = len(self.names)
        thresh = self.conservation_threshold
        region_colors = [
            DICT_POCKET_KLIFS_REGIONS[lab.split(":")[0]]["color"] for lab in labels
        ]

        # node coordinates: x by leaf order, y by merge height
        _, nodelist = to_tree(Z, rd=True)
        leaf_x = {leaf: float(p) for p, leaf in enumerate(leaves_list(Z))}
        node_x, node_y = {}, {}
        for leaf in range(n_leaves):
            node_x[leaf], node_y[leaf] = leaf_x[leaf], 0.0
        for i in range(Z.shape[0]):
            c1, c2 = int(Z[i, 0]), int(Z[i, 1])
            node_x[n_leaves + i] = 0.5 * (node_x[c1] + node_x[c2])
            node_y[n_leaves + i] = float(Z[i, 2])
        bar_h = 0.02 * max(node_y.values())

        records = {r["node_id"]: r for r in self.analyze_nodes()}
        parent = {}
        for i in range(Z.shape[0]):
            parent[int(Z[i, 0])] = n_leaves + i
            parent[int(Z[i, 1])] = n_leaves + i

        # tree-topology primitives come from the shared engine so the static figure
        # (KLIFSConservationTreeFigure) and this app cannot diverge on tree structure
        count = self.tree_count
        kept = self._is_kept
        gather = self.gather_children

        def kept_parent(nid):
            p = parent.get(nid)
            while p is not None and not kept(p):
                p = parent.get(p)
            return p

        leaf_types, node_type, type_colors = self._family_types()

        def tc(c):
            if node_type[c] is None:
                return COLOR_TREE_MIXED_FAMILY
            return type_colors.get(node_type[c], COLOR_TREE_MIXED_FAMILY)

        # per-node consensus and parent-relative logo payload
        consensus = {
            nid: [
                aa if (aa is not None and f >= thresh) else None
                for aa, f in self.node_conservation(nodelist[nid].pre_order())
            ]
            for nid in records
        }

        def consensus_of(c):
            if c >= n_leaves:
                return consensus[c]
            return [ch if ch not in self.gap_chars else None for ch in self.pockets[c]]

        # per-node consensus argmax residue (regardless of threshold) for IC-mode novelty
        argmax = {
            nid: [aa for aa, _ in self.node_conservation(nodelist[nid].pre_order())]
            for nid in records
        }

        def _freq_mode(nid, cons, par_cons):
            """Discrete >=80% consensus-frequency logo (letter height = frequency)."""
            aa_list, top_list, color_list, n_hi = [], [], [], 0
            for c, (aa, frac) in enumerate(cons):
                conserved = aa is not None and frac >= thresh
                show = aa is not None and frac >= self.logo_cutoff
                aa_list.append(aa if show else "")
                top_list.append(round(frac, 3) if show else 0.0)
                if conserved and par_cons[c] != aa:
                    color_list.append(region_colors[c])
                    n_hi += 1
                elif conserved:
                    color_list.append(COLOR_TREE_PSEUDO)
                else:
                    color_list.append(COLOR_LOGO_SUBTHRESHOLD)
            return dict(aa=aa_list, top=top_list, color=color_list, n_hi=n_hi)

        def _ic_mode(nid, background, par_aa):
            """Continuous information-content logo (letter height = IC in bits)."""
            ic, _ = self.node_information(
                nodelist[nid].pre_order(), background=background
            )
            aa_list, top_list, color_list, n_hi = [], [], [], 0
            for c, val in enumerate(ic):
                aa = argmax[nid][c]
                show = (
                    aa is not None
                    and val is not None
                    and val >= FLOAT_TREE_LOGO_IC_FLOOR
                )
                aa_list.append(aa if show else "")
                top_list.append(round(val, 3) if (show and val is not None) else 0.0)
                if show and par_aa[c] != aa:
                    color_list.append(region_colors[c])
                    n_hi += 1
                elif show:
                    color_list.append(COLOR_TREE_PSEUDO)
                else:
                    color_list.append(COLOR_LOGO_SUBTHRESHOLD)
            return dict(aa=aa_list, top=top_list, color=color_list, n_hi=n_hi)

        node_data = {}
        for nid in records:
            cons = self.node_conservation(nodelist[nid].pre_order())
            kp = kept_parent(nid)
            par_cons = consensus[kp] if kp is not None else [None] * ncol
            par_aa = argmax[kp] if kp is not None else [None] * ncol
            node_data[str(nid)] = dict(
                n_members=count(nid),
                n_conserved=len(records[nid]["conserved"]),
                modes=dict(
                    freq=_freq_mode(nid, cons, par_cons),
                    ic_sp=_ic_mode(nid, "swissprot", par_aa),
                    ic_kin=_ic_mode(nid, "kinome", par_aa),
                ),
            )

        def names_of(nid):
            if nid < n_leaves:
                return [self.names[nid]]
            return [self.names[m] for m in nodelist[nid].pre_order()]

        def trunc(items):
            if len(items) <= self.name_trunc:
                return ", ".join(items)
            return (
                ", ".join(items[: self.name_trunc])
                + f" … (+{len(items) - self.name_trunc})"
            )

        def breakpoint_cols(children):
            clades = [c for c in children if c >= n_leaves and count(c) >= 2]
            diffs = []
            for col in range(ncol):
                seen = {}
                for c in clades:
                    aa = consensus_of(c)[col]
                    if aa is not None:
                        seen[aa] = seen.get(aa, 0) + 1
                if len(seen) >= 2:
                    diffs.append(f"{labels[col]} {'/'.join(sorted(seen))}")
            return diffs

        # contracted multifurcating layout
        seg = dict(x0=[], y0=[], x1=[], y1=[], color=[])
        intn = dict(x=[], y=[], node_id=[], split=[], n=[], fill=[], nbreak=[])
        bars = dict(
            left=[], right=[], top=[], bottom=[], node_id=[], n=[], members=[], color=[]
        )
        leafd = dict(x=[], y=[], color=[], name=[], ftype=[])

        def add_seg(x0, y0, x1, y1, color):
            seg["x0"].append(x0)
            seg["y0"].append(y0)
            seg["x1"].append(x1)
            seg["y1"].append(y1)
            seg["color"].append(color)

        def render(nid):
            y = node_y[nid]
            children = gather(nid)
            for c in children:
                if c < n_leaves:
                    add_seg(node_x[c], 0.0, node_x[c], y, tc(c))
                    leafd["x"].append(node_x[c])
                    leafd["y"].append(0.0)
                    leafd["color"].append(tc(c))
                    leafd["name"].append(self.names[c])
                    leafd["ftype"].append(leaf_types[c])
                elif count(c) >= self.min_cluster_size:
                    add_seg(node_x[c], node_y[c], node_x[c], y, tc(c))
                    render(c)
                else:
                    add_seg(node_x[c], bar_h, node_x[c], y, tc(c))
                    xs = [leaf_x[m] for m in nodelist[c].pre_order()]
                    bars["left"].append(min(xs) - 0.45)
                    bars["right"].append(max(xs) + 0.45)
                    bars["top"].append(bar_h)
                    bars["bottom"].append(0.0)
                    bars["node_id"].append(str(c))
                    bars["n"].append(count(c))
                    bars["members"].append(trunc(names_of(c)))
                    bars["color"].append(tc(c))
            xs = [node_x[c] for c in children]
            add_seg(min(xs), y, max(xs), y, tc(nid))
            diffs = breakpoint_cols(children)
            intn["x"].append(node_x[nid])
            intn["y"].append(y)
            intn["node_id"].append(str(nid))
            intn["n"].append(count(nid))
            intn["split"].append(
                trunc(diffs) if diffs else "(no fixed differences ≥80%)"
            )
            intn["nbreak"].append(len(diffs))
            intn["fill"].append(COLOR_TREE_BREAK if diffs else COLOR_TREE_NO_BREAK)

        render(2 * n_leaves - 2)

        seg_src = ColumnDataSource(seg)
        int_src = ColumnDataSource(intn)
        bar_src = ColumnDataSource(bars)
        leaf_src = ColumnDataSource(leafd)

        # --- tree figure ---
        tree = figure(
            width=1500,
            height=520,
            sizing_mode="stretch_width",
            y_axis_label="distance",
            title=f"KLIFS conservation tree (clades ≥{self.min_cluster_size}; tap a node or bar)",
            tools="tap,pan,wheel_zoom,box_zoom,reset",
            toolbar_location="above",
        )
        tree.segment(
            "x0", "y0", "x1", "y1", source=seg_src, line_color="color", line_width=1.5
        )
        leaf_glyph = tree.scatter(
            "x",
            "y",
            source=leaf_src,
            size=5,
            color="color",
            marker="circle",
            line_color=None,
        )
        bar_glyph = tree.quad(
            left="left",
            right="right",
            top="top",
            bottom="bottom",
            source=bar_src,
            fill_color="color",
            line_color="white",
            line_width=0.5,
        )
        int_glyph = tree.scatter(
            "x",
            "y",
            source=int_src,
            size=8,
            fill_color="fill",
            marker="circle",
            line_color="white",
            line_width=0.5,
        )
        tree.xaxis.visible = False
        tree.xgrid.visible = False
        tree.ygrid.visible = False
        tree.add_tools(
            HoverTool(
                renderers=[int_glyph],
                tooltips=[
                    ("node", "@node_id"),
                    ("n", "@n"),
                    ("#breakpoints", "@nbreak"),
                    ("breakpoint", "@split"),
                ],
            )
        )
        tree.add_tools(
            HoverTool(
                renderers=[bar_glyph],
                tooltips=[("clade", "@node_id"), ("n", "@n"), ("members", "@members")],
            )
        )
        tree.add_tools(
            HoverTool(
                renderers=[leaf_glyph],
                tooltips=[("kinase", "@name"), ("family", "@ftype")],
            )
        )

        # --- logo panel ---
        logo = figure(
            width=1500,
            height=340,
            sizing_mode="stretch_width",
            tools="",
            toolbar_location=None,
            title="Tap a node to see its conserved residues",
            x_range=(-0.6, ncol - 0.4),
            y_range=(-0.34, 1.08),
        )
        logo_src = ColumnDataSource(dict(x=[], aa=[], color=[], top=[], pos=[]))
        thr_line = logo.line(
            [-0.6, ncol - 0.4],
            [thresh, thresh],
            line_color=COLOR_CONSERVATION_PRIMARY,
            line_dash="dashed",
            line_width=1.5,
        )
        logo_bars = logo.vbar(
            x="x",
            width=0.82,
            top="top",
            bottom=0,
            source=logo_src,
            fill_color=COLOR_LOGO_BAR,
            line_color=None,
        )
        logo.text(
            x="x",
            y="top",
            text="aa",
            source=logo_src,
            text_color="color",
            text_font_size="11px",
            text_align="center",
            text_baseline="bottom",
        )
        logo.add_tools(
            HoverTool(
                renderers=[logo_bars],
                tooltips=[("KLIFS", "@pos"), ("aa", "@aa"), ("value", "@top{0.00}")],
            )
        )
        region_src = ColumnDataSource(
            dict(
                x=list(range(ncol)),
                color=region_colors,
                top=[-0.04] * ncol,
                bottom=[-0.20] * ncol,
            )
        )
        logo.vbar(
            x="x",
            width=1.0,
            top="top",
            bottom="bottom",
            source=region_src,
            color="color",
            line_color="white",
        )
        regs = [lab.split(":")[0] for lab in labels]
        blocks, start = [], 0
        for i in range(1, ncol + 1):
            if i == ncol or regs[i] != regs[start]:
                blocks.append((regs[start], (start + i - 1) / 2.0))
                start = i
        block_src = ColumnDataSource(
            dict(x=[c for _, c in blocks], name=[r for r, _ in blocks])
        )
        logo.text(
            x="x",
            y=-0.30,
            text="name",
            source=block_src,
            text_align="center",
            text_baseline="middle",
            text_font_size="8pt",
            text_color=COLOR_DARK_TEXT,
        )
        logo.yaxis.axis_label = "consensus frequency"
        logo.xaxis.visible = False
        logo.xgrid.visible = False
        logo.ygrid.visible = False

        header = Div(
            width=1500,
            sizing_mode="stretch_width",
            text="<b>No node selected.</b> Tap a circle (split) or bar (clade) below.",
        )
        present = [t for t in type_colors if t in set(leaf_types)]
        swatches = (
            "".join(
                f"<span style='color:{type_colors[t]};font-weight:bold'>&#9608;</span>&nbsp;{t} &nbsp;&nbsp;"
                for t in present
            )
            + f"<span style='color:{COLOR_TREE_MIXED_FAMILY};font-weight:bold'>&#9608;</span>&nbsp;mixed"
        )
        legend = Div(
            width=1500,
            sizing_mode="stretch_width",
            text="<b>Branch color</b> = family of a monophyletic clade (lipid &gt; pseudo &gt; group), "
            "grey once mixed. &nbsp; <b>Node fill</b>: "
            f"<span style='color:{COLOR_TREE_BREAK}'>&#9679;</span> has a fixed-difference breakpoint, "
            f"<span style='color:{COLOR_TREE_NO_BREAK}'>&#9679;</span> none (structural).<br>"
            + swatches,
        )

        mode_select = Select(
            title="Logo statistic",
            value="ic_sp",
            options=[
                ("freq", "Consensus frequency (≥80%)"),
                ("ic_sp", "Information content — Swiss-Prot (bits)"),
                ("ic_kin", "Information content — kinome (bits)"),
            ],
            width=320,
        )

        callback = CustomJS(
            args=dict(
                int_src=int_src,
                bar_src=bar_src,
                logo_src=logo_src,
                header=header,
                node_data=node_data,
                labels=labels,
                logo_fig=logo,
                mode_select=mode_select,
                thr_line=thr_line,
                ic_max=FLOAT_TREE_LOGO_IC_MAX,
            ),
            code="""
            function show(nid) {
              const nd = node_data[nid]; if (!nd) { return; }
              const mode = mode_select.value;
              const m = nd.modes[mode]; if (!m) { return; }
              const xs = [], pos = [];
              for (let i = 0; i < labels.length; i++) { xs.push(i); pos.push(labels[i]); }
              logo_src.data = { x: xs, aa: m.aa, color: m.color, top: m.top, pos: pos };
              logo_src.change.emit();
              const is_freq = (mode === "freq");
              logo_fig.y_range.end = is_freq ? 1.08 : ic_max;
              thr_line.visible = is_freq;
              logo_fig.yaxis[0].axis_label = is_freq
                ? "consensus frequency" : "information content (bits)";
              logo_fig.title.text = "Node " + nid + " — " + nd.n_members + " kinases";
              const ht = is_freq ? "letter height = frequency"
                                 : "letter height = information content (bits)";
              const hi = is_freq ? "newly conserved here" : "high-IC & changed vs parent";
              header.text = "<b>Node " + nid + "</b> n=" + nd.n_members +
                " | conserved (≥80%): " + nd.n_conserved +
                " | <b>" + hi + ": " + m.n_hi + "</b>" +
                "<br><span style='color:#666'>" + ht + "; " +
                "<span style='color:#c0392b'><b>colored</b></span> = changed vs parent, " +
                "<b>black</b> = inherited, grey = below</span>";
            }
            const ii = int_src.selected.indices, ib = bar_src.selected.indices;
            if (ii.length) { show(int_src.data['node_id'][ii[0]]); }
            else if (ib.length) { show(bar_src.data['node_id'][ib[0]]); }
            """,
        )
        tree.select(TapTool).callback = callback
        mode_select.js_on_change("value", callback)

        return column(
            header, mode_select, logo, legend, tree, sizing_mode="stretch_width"
        )

    def save_app(self, output_path: str, filename: str | None = None) -> None:
        """Build the explorer and write a self-contained HTML file to ``output_path``.

        Parameters
        ----------
        output_path : str
            Directory to write the HTML file into.
        filename : str | None, optional
            Output filename; defaults to :data:`STR_FILE_TREE_APP`.
        """
        import os

        layout = self.build_layout()
        html = file_html(
            layout, resources=INLINE, title="KLIFS conservation tree explorer"
        )
        if filename is None:
            filename = STR_FILE_TREE_APP
        filepath = os.path.join(output_path, filename)
        with open(filepath, "w") as f:
            f.write(html)
        logger.info("saved KLIFS conservation-tree explorer -> %s", filepath)


class KLIFSResidueDotApp(KLIFSHierarchicalConservation):
    """Interactive standalone-HTML per-amino-acid KLIFS dot-plot explorer.

    A ``Select`` dropdown chooses one of the 20 amino acids (or the ``-`` gap); the
    plot then shows, for the 85 KLIFS pocket columns (x) against the human kinome in
    dendrogram leaf order (y), a dot wherever a kinase carries the selected residue at
    that column. Dots are colored by the dendrogram leaf coloring (Manning group /
    Lipid, with a black outline for pseudokinases). Dots of the same display-leaf clade
    that share the residue at a column (:data:`INT_RESIDUE_DOT_MIN_ENCLOSE` or more) are
    enclosed, revealing columns "enriched" for that residue within a family (e.g.
    conserved cysteines). Hover reports the kinase, its display-leaf number, the UniProt
    residue (amino acid + canonical index, e.g. ``L858``), the KLIFS label, and the
    KinHub / KLIFS family labels. All per-amino-acid data is precomputed and embedded,
    so the HTML needs no server.

    Panel-only view built on :class:`KLIFSHierarchicalConservation`; ``save_app`` writes
    a self-contained HTML file to the structured output tree.
    """

    model_config = {"arbitrary_types_allowed": True}

    min_cluster_size: int = INT_TREE_MIN_CLUSTER_DISPLAY
    """Display-tree collapse threshold (sets the display-leaf clades that are enclosed)."""
    default_aa: str = STR_RESIDUE_DOT_DEFAULT_AA
    """Amino acid shown on first load (default: cysteine)."""

    def _dot_style(self, i: int) -> tuple[str, str]:
        """``(fill, line)`` colors for leaf ``i``'s dot (black outline if pseudokinase)."""
        fill, edge, _ = self._member_style(i)
        line = "#000000" if edge == "#000000" else fill
        return fill, line

    def _column_stack(
        self, aa: str, pos: int, ordered_members: list[int], meta: dict
    ) -> list[dict]:
        """Kinases carrying ``aa`` at KLIFS column ``pos``, in dendrogram leaf order.

        Each entry is the precomputed per-kinase ``meta`` dict; the list order is the
        bottom-to-top stacking order for the frequency dot plot (so same-clade dots are
        contiguous and can be enclosed).
        """
        return [meta[i] for i in ordered_members if self.pockets[i][pos] == aa]

    def _precompute_aa_data(
        self,
        ordered_members: list[int],
        leaf_of_member: dict[int, int],
    ) -> tuple[dict, dict, dict]:
        """Per-amino-acid stacked-dot and clade-enclosure payloads for the CustomJS switch.

        For each amino acid this builds a frequency dot plot: within each KLIFS column
        the kinases carrying the residue are stacked (y = 0, 1, 2, ...) in dendrogram
        leaf order, so a column's dot count is its frequency and empty columns simply
        have no dots. Runs of >= :data:`INT_RESIDUE_DOT_MIN_ENCLOSE` consecutive
        same-clade dots within a column are enclosed.

        Returns
        -------
        tuple[dict, dict, dict]
            ``(aa_data, enc_data, max_count)`` keyed by amino-acid symbol; the first two
            are dicts of parallel arrays ready to become a ``ColumnDataSource``, and
            ``max_count`` is the tallest column stack (for the y-range).
        """
        labels = self.position_labels
        ncol = len(labels)

        # per-member constants (color, outline, name, leaf number, families) + the
        # per-column UniProt indices, computed once and reused across every amino acid
        meta = {}
        for i in ordered_members:
            fill, line = self._dot_style(i)
            kinhub_family, klifs_family = self._hover_families(i)
            meta[i] = {
                "idx": i,
                "fill": fill,
                "line": line,
                "name": self.names[i],
                "leaf": leaf_of_member[i],
                "kinhub": kinhub_family,
                "klifs": klifs_family,
                "uniprot": [self._uniprot_index_at(i, pos) for pos in range(ncol)],
            }

        aa_data, enc_data, max_count = {}, {}, {}
        for aa in LIST_DOT_ALPHABET:
            dots = dict(
                x=[],
                y=[],
                color=[],
                line=[],
                name=[],
                leaf=[],
                residue=[],
                klifs=[],
                kinhub=[],
                klifs_fam=[],
            )
            enc = dict(x=[], y=[], width=[], height=[])
            tallest = 0
            for pos in range(ncol):
                stack = self._column_stack(aa, pos, ordered_members, meta)
                tallest = max(tallest, len(stack))
                for height, info in enumerate(stack):
                    uniprot_idx = info["uniprot"][pos]
                    residue = (
                        f"{aa}{uniprot_idx}" if uniprot_idx is not None else f"{aa}?"
                    )
                    dots["x"].append(pos)
                    dots["y"].append(height)
                    dots["color"].append(info["fill"])
                    dots["line"].append(info["line"])
                    dots["name"].append(info["name"])
                    dots["leaf"].append(info["leaf"])
                    dots["residue"].append(residue)
                    dots["klifs"].append(labels[pos])
                    dots["kinhub"].append(info["kinhub"])
                    dots["klifs_fam"].append(info["klifs"])

                # enclose maximal runs of >=MIN consecutive same-clade dots in the stack
                start = 0
                for end in range(1, len(stack) + 1):
                    if end == len(stack) or stack[end]["leaf"] != stack[start]["leaf"]:
                        if end - start >= INT_RESIDUE_DOT_MIN_ENCLOSE:
                            lo, hi = start, end - 1
                            enc["x"].append(pos)
                            enc["y"].append((lo + hi) / 2.0)
                            enc["width"].append(0.7)
                            enc["height"].append((hi - lo) + 0.9)
                        start = end
            aa_data[aa] = dots
            enc_data[aa] = enc
            max_count[aa] = tallest

        return aa_data, enc_data, max_count

    def build_layout(self):
        """Assemble the Bokeh dot-plot layout with an amino-acid selector.

        Returns
        -------
        bokeh.models.LayoutDOM
            The column layout ready for :func:`bokeh.embed.file_html`.
        """
        labels = self.position_labels
        ncol = len(labels)
        region_colors = [
            DICT_POCKET_KLIFS_REGIONS[lab.split(":")[0]]["color"] for lab in labels
        ]

        (
            _tree,
            ordered_members,
            _row_of_member,
            leaf_of_member,
            _leaf_members,
        ) = self.residue_dot_layout(min_cluster_size=self.min_cluster_size)

        aa_data, enc_data, max_count = self._precompute_aa_data(
            ordered_members, leaf_of_member
        )
        default_aa = (
            self.default_aa if self.default_aa in aa_data else LIST_DOT_ALPHABET[0]
        )

        dot_src = ColumnDataSource(dict(aa_data[default_aa]))
        enc_src = ColumnDataSource(dict(enc_data[default_aa]))

        # frequency dot plot: each column stacks one dot per kinase (y = count), so the
        # y-range follows the tallest stack of the selected residue and the region strip
        # sits below the zero baseline
        top = max(max_count[default_aa], 1) + 1
        p = figure(
            width=1400,
            height=520,
            sizing_mode="stretch_width",
            x_range=(-0.6, ncol - 0.4),
            y_range=(-3.5, top),
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above",
            title=f"KLIFS residue dot plot — {DICT_DOT_AA_LABEL[default_aa]}",
        )

        # clade enclosures (light-grey rounded rectangles, drawn under the dots)
        p.rect(
            x="x",
            y="y",
            width="width",
            height="height",
            source=enc_src,
            fill_color=COLOR_TREE_TABLE_NO_CONSENSUS,
            line_color=COLOR_TREE_PSEUDO,
            line_width=1.0,
            border_radius=5,
        )
        dot_glyph = p.scatter(
            x="x",
            y="y",
            source=dot_src,
            size=6,
            marker="circle",
            fill_color="color",
            line_color="line",
            line_width=0.4,
        )
        p.add_tools(
            HoverTool(
                renderers=[dot_glyph],
                tooltips=[
                    ("kinase", "@name"),
                    ("leaf", "@leaf"),
                    ("residue", "@residue"),
                    ("KLIFS", "@klifs"),
                    ("KinHub family", "@kinhub"),
                    ("KLIFS family", "@klifs_fam"),
                ],
            )
        )

        # KLIFS region color strip + block labels below the zero baseline
        region_src = ColumnDataSource(
            dict(x=list(range(ncol)), color=region_colors, y=[-1.6] * ncol)
        )
        p.rect(
            x="x",
            y="y",
            width=1.0,
            height=0.8,
            source=region_src,
            color="color",
            line_color="white",
        )
        regs = [lab.split(":")[0] for lab in labels]
        blocks, start = [], 0
        for i in range(1, ncol + 1):
            if i == ncol or regs[i] != regs[start]:
                blocks.append((regs[start], (start + i - 1) / 2.0))
                start = i
        block_src = ColumnDataSource(
            dict(x=[c for _, c in blocks], name=[r for r, _ in blocks])
        )
        p.text(
            x="x",
            y=-2.7,
            text="name",
            source=block_src,
            text_align="center",
            text_baseline="middle",
            text_font_size="8pt",
            text_color=COLOR_DARK_TEXT,
        )

        p.yaxis.axis_label = "Counts"
        p.xaxis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False

        select = Select(
            title="Amino acid",
            value=default_aa,
            options=[(aa, DICT_DOT_AA_LABEL[aa]) for aa in LIST_DOT_ALPHABET],
            width=200,
        )
        callback = CustomJS(
            args=dict(
                dot_src=dot_src,
                enc_src=enc_src,
                aa_data=aa_data,
                enc_data=enc_data,
                max_count=max_count,
                aa_label=DICT_DOT_AA_LABEL,
                select=select,
                fig=p,
            ),
            code="""
            const aa = select.value;
            dot_src.data = aa_data[aa];
            dot_src.change.emit();
            enc_src.data = enc_data[aa];
            enc_src.change.emit();
            fig.y_range.end = Math.max(max_count[aa], 1) + 1;
            fig.title.text = "KLIFS residue dot plot — " + aa_label[aa];
            """,
        )
        select.js_on_change("value", callback)

        # kinase-group legend (Manning group / Lipid) plus the pseudokinase-outline note
        present = []
        seen = set()
        for i in ordered_members:
            fam = self._family_label(i)
            if fam not in seen:
                seen.add(fam)
                present.append(fam)
        swatches = "".join(
            f"<span style='color:{DICT_KINASE_GROUP_COLORS.get(g, COLOR_TREE_FALLBACK)};"
            f"font-weight:bold'>&#9679;</span>&nbsp;{g} &nbsp;&nbsp;"
            for g in present
        )
        legend = Div(
            width=1400,
            sizing_mode="stretch_width",
            text="<b>Dot color</b> = Manning group / Lipid; "
            "<b>black outline</b> = pseudokinase. "
            f"<b>Circles</b> enclose &ge;{INT_RESIDUE_DOT_MIN_ENCLOSE} same-clade dots at "
            "a KLIFS column (residue enriched within that family).<br>" + swatches,
        )
        header = Div(
            width=1400,
            sizing_mode="stretch_width",
            text="<b>Per-amino-acid KLIFS dot plot.</b> Choose a residue; each dot is a "
            "kinase carrying it at that KLIFS column, ordered by the conservation "
            "dendrogram.",
        )

        return column(header, select, legend, p, sizing_mode="stretch_width")

    def save_app(self, output_path: str, filename: str | None = None) -> None:
        """Build the dot-plot explorer and write a self-contained HTML file.

        Parameters
        ----------
        output_path : str
            Directory to write the HTML file into.
        filename : str | None, optional
            Output filename; defaults to :data:`STR_FILE_RESIDUE_DOT_APP`.
        """
        import os

        layout = self.build_layout()
        html = file_html(
            layout, resources=INLINE, title="KLIFS residue dot plot explorer"
        )
        if filename is None:
            filename = STR_FILE_RESIDUE_DOT_APP
        filepath = os.path.join(output_path, filename)
        with open(filepath, "w") as f:
            f.write(html)
        logger.info("saved KLIFS residue dot-plot explorer -> %s", filepath)
