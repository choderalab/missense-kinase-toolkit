"""Tests for hierarchical KLIFS conservation clustering.

Covers the reusable column-conservation primitive
(:func:`mkt.databases.pssm.column_conservation`) and the
:class:`mkt.databases.conservation.KLIFSHierarchicalConservation` engine --
distance metrics, linkage, per-node conservation and the survives-up /
critical-depth pruning -- plus the static supplemental figure and the interactive
Bokeh explorer. A small synthetic 85-column panel is supplied explicitly so no
cohort data or network access is needed.
"""

import numpy as np
import pytest
from mkt.databases.conservation import (
    INT_KLIFS_POCKET_LENGTH,
    KLIFSConservationTreeFigure,
    KLIFSHierarchicalConservation,
    KLIFSTreeConservationApp,
)
from mkt.databases.pssm import column_conservation


def _panel():
    """Two-subfamily synthetic panel: column 0 invariant, the rest subfamily-split."""
    tail = INT_KLIFS_POCKET_LENGTH - 1
    pockets = [
        "G" + "A" * tail,
        "G" + "A" * tail,
        "G" + "D" * tail,
        "G" + "D" * tail,
    ]
    names = ["a1", "a2", "d1", "d2"]
    groups = ["A", "A", "D", "D"]
    return names, pockets, groups


def test_column_conservation_consensus_and_gaps():
    cons = column_conservation(["AAG", "AAC", "A-G"])
    # column 0: all A -> (A, 1.0)
    assert cons[0] == ("A", 1.0)
    # column 1: two A one gap -> consensus A over two observed residues
    assert cons[1] == ("A", 1.0)
    # column 2: G, C, G -> consensus G at 2/3
    assert cons[2][0] == "G"
    assert cons[2][1] == pytest.approx(2 / 3)
    # all-gap column -> (None, 0.0)
    assert column_conservation(["-", "-"])[0] == (None, 0.0)


def test_column_conservation_weights_match_unweighted():
    # uniform weights must reproduce the unweighted counts exactly
    seqs = ["AAG", "AAC", "A-G"]
    assert column_conservation(seqs, weights=[1.0, 1.0, 1.0]) == column_conservation(
        seqs
    )


@pytest.mark.parametrize("metric", ["identity", "blosum"])
def test_distance_matrix_is_valid(metric):
    names, pockets, groups = _panel()
    analyzer = KLIFSHierarchicalConservation(
        names=names, pockets=pockets, groups=groups, metric=metric
    )
    dist = analyzer.distance_matrix
    assert dist.shape == (4, 4)
    assert np.allclose(dist, dist.T)
    assert np.allclose(np.diag(dist), 0.0)
    # identical sequences are at distance 0; the two subfamilies are not
    assert dist[0, 1] == pytest.approx(0.0)
    assert dist[0, 2] > 0.0


def test_identity_distance_between_disjoint_subfamilies():
    names, pockets, groups = _panel()
    analyzer = KLIFSHierarchicalConservation(
        names=names, pockets=pockets, groups=groups, metric="identity"
    )
    # subfamilies share only the invariant column 0 -> 1 of 85 identical
    assert analyzer.distance_matrix[0, 2] == pytest.approx(
        1 - 1 / INT_KLIFS_POCKET_LENGTH
    )


def test_unknown_metric_raises():
    names, pockets, groups = _panel()
    with pytest.raises(ValueError):
        KLIFSHierarchicalConservation(
            names=names, pockets=pockets, groups=groups, metric="nope"
        )


def test_survives_up_prunes_subfamily_specific_columns():
    names, pockets, groups = _panel()
    analyzer = KLIFSHierarchicalConservation(
        names=names, pockets=pockets, groups=groups, metric="identity"
    )
    records = {rec["node_id"]: rec for rec in analyzer.analyze_nodes()}
    root_rec = records[analyzer.tree.id]

    # only the invariant column survives up across the balanced root split
    assert root_rec["survives_up"] == {0: "G"}
    # but each subfamily-specific column is conserved *within* a child clade
    child_recs = [r for r in analyzer.analyze_nodes() if r["n_members"] == 2]
    assert all(len(r["conserved"]) == INT_KLIFS_POCKET_LENGTH for r in child_recs)

    cd = analyzer.critical_depth()
    invariant = cd[cd["column"] == 0].iloc[0]
    assert invariant["critical_depth"] == 0
    assert invariant["consensus_aa"] == "G"


def test_henikoff_downweights_redundant_majority():
    # three identical A-sequences and one divergent D-sequence at every column
    tail = INT_KLIFS_POCKET_LENGTH
    pockets = ["A" * tail, "A" * tail, "A" * tail, "D" * tail]
    names = ["a1", "a2", "a3", "d1"]
    groups = ["A", "A", "A", "D"]

    plain = KLIFSHierarchicalConservation(
        names=names, pockets=pockets, groups=groups, metric="identity"
    )
    henikoff = KLIFSHierarchicalConservation(
        names=names,
        pockets=pockets,
        groups=groups,
        metric="identity",
        weighting="henikoff",
    )
    all_members = list(range(4))
    # unweighted: A is consensus at 3/4 = 0.75
    aa_plain, frac_plain = plain.node_conservation(all_members)[0]
    assert aa_plain == "A"
    assert frac_plain == pytest.approx(0.75)
    # henikoff up-weights the divergent D so the redundant A majority drops to 0.5
    _, frac_h = henikoff.node_conservation(all_members)[0]
    assert frac_h == pytest.approx(0.5)


def test_min_child_members_ignores_singleton_outlier():
    # four identical majority sequences plus one divergent outlier that the tree
    # peels off as a root singleton; column 0 is invariant across all five, columns
    # 1+ are conserved only within the majority and broken by the outlier
    tail = INT_KLIFS_POCKET_LENGTH - 2
    majority = "GA" + "M" * tail
    outlier = "GQ" + "W" * tail
    pockets = [majority, majority, majority, majority, outlier]
    names = ["m1", "m2", "m3", "m4", "out"]
    groups = ["M", "M", "M", "M", "O"]

    def _root_survives(min_child):
        analyzer = KLIFSHierarchicalConservation(
            names=names,
            pockets=pockets,
            groups=groups,
            metric="identity",
            min_child_members=min_child,
        )
        records = {rec["node_id"]: rec for rec in analyzer.analyze_nodes()}
        return records[analyzer.tree.id]["survives_up"]

    # strict (min=1): the singleton outlier vetoes every majority-only column,
    # so only the pan-panel invariant survives at the root
    strict = _root_survives(1)
    assert strict == {0: "G"}
    # min=2: the singleton is ignored, so the majority-conserved columns survive too
    relaxed = _root_survives(2)
    assert 0 in relaxed
    assert 1 in relaxed
    assert len(relaxed) > len(strict)


def test_unknown_weighting_raises():
    names, pockets, groups = _panel()
    analyzer = KLIFSHierarchicalConservation(
        names=names, pockets=pockets, groups=groups, weighting="nope"
    )
    with pytest.raises(ValueError):
        analyzer.node_conservation([0, 1])


def test_build_display_tree_aggregates_singletons():
    # one tight majority clade plus three lone outliers the tree peels off as singletons;
    # with aggregation on, no leaf row may hold a single sequence
    tail = INT_KLIFS_POCKET_LENGTH - 1
    pockets = ["G" + "A" * tail] * 6 + [
        "G" + "C" * tail,
        "G" + "D" * tail,
        "G" + "E" * tail,
    ]
    names = [f"m{i}" for i in range(6)] + ["o1", "o2", "o3"]
    groups = ["M"] * 6 + ["O", "O", "O"]
    analyzer = KLIFSHierarchicalConservation(
        names=names, pockets=pockets, groups=groups, metric="identity"
    )

    tree = analyzer.build_display_tree(min_cluster_size=2, aggregate_singletons=True)
    # every leaf row holds >=2 members and the membership covers all kinases exactly once
    assert all(len(leaf.members) >= 2 for leaf in tree.leaves)
    covered = sorted(m for leaf in tree.leaves for m in leaf.members)
    assert covered == list(range(len(names)))
    # order indexes the leaves list, one slot per leaf row
    assert sorted(tree.order) == list(range(len(tree.leaves)))

    # with aggregation off, the same panel exposes singleton leaf rows
    raw = analyzer.build_display_tree(min_cluster_size=2, aggregate_singletons=False)
    assert any(leaf.kind == "singleton" for leaf in raw.leaves)


def test_members_consensus_matches_threshold():
    names, pockets, groups = _panel()
    analyzer = KLIFSHierarchicalConservation(
        names=names, pockets=pockets, groups=groups, metric="identity"
    )
    # column 0 is invariant ("G") across all four; later columns split by subfamily, so a
    # mixed two-subfamily set has no >=80% consensus there
    cons = analyzer.members_consensus([0, 1, 2, 3])
    assert cons[0] == "G"
    assert cons[1] is None


def test_conservation_tree_figure_builds(tmp_path):
    # the static figure is panel-only, so a synthetic panel suffices; save into tmp_path
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tail = INT_KLIFS_POCKET_LENGTH - 1
    pockets = ["G" + "A" * tail] * 4 + ["G" + "D" * tail] * 4
    names = [f"k{i}" for i in range(8)]
    groups = ["TK"] * 4 + ["CMGC"] * 4
    fig_obj = KLIFSConservationTreeFigure(
        names=names, pockets=pockets, groups=groups, min_cluster_size=2
    )
    fig, ax = fig_obj.build_figure()
    assert fig is not None and ax is not None
    plt.close(fig)

    fig_obj.plot(str(tmp_path), formats=("png",))
    assert (tmp_path / "klifs_conservation_tree.png").exists()


def test_conservation_tree_split_figures(tmp_path):
    # the split supplement = summary + top/bottom detail panels; render + save as PDFs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tail = INT_KLIFS_POCKET_LENGTH - 1
    pockets = ["G" + "A" * tail] * 4 + ["G" + "C" * tail] * 4 + ["G" + "D" * tail] * 4
    names = [f"k{i}" for i in range(12)]
    groups = ["TK"] * 4 + ["CMGC"] * 4 + ["CAMK"] * 4
    obj = KLIFSConservationTreeFigure(
        names=names, pockets=pockets, groups=groups, min_cluster_size=2
    )
    n = len(obj.build_display_tree().order)
    assert n >= 2
    split = max(1, n // 2)

    figs = obj.build_split_figures(split_index=split)
    assert [name for name, _ in figs] == [
        "klifs_conservation_tree_summary",
        "klifs_conservation_tree_top",
        "klifs_conservation_tree_bottom",
    ]
    for _, fig in figs:
        plt.close(fig)

    obj.plot_split(str(tmp_path), formats=("pdf",), split_index=split)
    for base in ("summary", "top", "bottom"):
        assert (tmp_path / f"klifs_conservation_tree_{base}.pdf").exists()


def test_tree_conservation_app_builds_and_saves(tmp_path):
    # the interactive Bokeh app is panel-only, so a synthetic panel suffices
    tail = INT_KLIFS_POCKET_LENGTH - 1
    pockets = ["G" + "A" * tail] * 4 + ["G" + "D" * tail] * 4
    names = [f"k{i}" for i in range(8)]
    groups = ["TK"] * 4 + ["CMGC"] * 4
    app = KLIFSTreeConservationApp(
        names=names, pockets=pockets, groups=groups, min_cluster_size=2
    )
    layout = app.build_layout()
    assert layout is not None

    app.save_app(str(tmp_path))
    out = tmp_path / "conservation_tree_explorer.html"
    assert out.exists()
    html = out.read_text()
    assert "Bokeh" in html and len(html) > 1000
