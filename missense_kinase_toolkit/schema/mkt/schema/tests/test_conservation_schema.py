"""Tests for the scipy-free :class:`KLIFSConservationData` schema.

Covers the pydantic round-trip and the pure-Python reconstruction methods
(:meth:`to_long_form`, :meth:`rebuild_square`) on a tiny hand-built 3-kinase
fixture, so the schema is exercised without numpy/scipy or the panel build.
"""

import pytest
from mkt.schema.conservation_schema import KLIFSConservationData


@pytest.fixture
def sample():
    """Three-kinase fixture with a known upper-triangle distance vector."""
    return KLIFSConservationData(
        metric="blosum",
        blosum_name="BLOSUM62",
        linkage_method="average",
        conservation_threshold=0.8,
        weighting="none",
        exclude_pseudokinases=False,
        gap_chars="-",
        n_kinases=3,
        pocket_length=2,
        names=["A", "B", "C"],
        position_labels=["I:1", "I:2"],
        distances_condensed=[0.1, 0.2, 0.3],
        linkage_matrix=[[0.0, 1.0, 0.1, 2.0], [3.0, 2.0, 0.3, 3.0]],
        leaves_order=[0, 1, 2],
        display_leaves=[
            {"members": [0, 1], "kind": "clade"},
            {"members": [2], "kind": "singleton"},
        ],
        display_order=[0, 1],
    )


def test_model_validate_roundtrip(sample):
    # model_dump -> model_validate must reproduce the object exactly
    assert KLIFSConservationData.model_validate(sample.model_dump()) == sample


def test_to_long_form(sample):
    # the condensed vector expands to (kin1, kin2, distance) upper-triangle triples
    assert sample.to_long_form() == [
        ("A", "B", 0.1),
        ("A", "C", 0.2),
        ("B", "C", 0.3),
    ]


def test_rebuild_square_distance(sample):
    # as_similarity=False returns raw distances with a zero diagonal
    mat = sample.rebuild_square(as_similarity=False)
    assert mat == [
        [0.0, 0.1, 0.2],
        [0.1, 0.0, 0.3],
        [0.2, 0.3, 0.0],
    ]


def test_rebuild_square_similarity_default(sample):
    # the default is a similarity matrix (1 - distance) with a unit diagonal
    mat = sample.rebuild_square()
    assert all(mat[i][i] == 1.0 for i in range(3))
    assert mat[0][1] == pytest.approx(0.9)
    assert mat[1][2] == pytest.approx(0.7)


def test_rebuild_square_reorder_by_leaves(sample):
    # a non-identity leaf order permutes rows and columns consistently
    sample.leaves_order = [2, 0, 1]
    base = sample.rebuild_square(as_similarity=False)
    reordered = sample.rebuild_square(as_similarity=False, reorder_by_leaves=True)
    order = [2, 0, 1]
    assert reordered == [[base[i][j] for j in order] for i in order]
