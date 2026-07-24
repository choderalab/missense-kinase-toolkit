import pytest


def test_molecular_brake_residues_fgfr2(dict_kinase):
    """FGFR2 carries the full canonical N-E-K molecular brake triad."""
    obj = dict_kinase["FGFR2"]
    assert obj.return_molecular_brake_residues() == {
        "b.l:37": "N",
        "hinge:46": "E",
        "VIII:79": "K",
    }
    assert obj.check_molecular_brake_against_canonical() == (True, True, True)


def test_molecular_brake_offset_applied(dict_kinase):
    """The VIII:79 lysine sits one residue N-terminal to its KLIFS-aligned index.

    The DICT_MOLECULAR_BRAKE_OFFSET of -1 must be applied, otherwise the mapped
    index resolves to the non-conserved residue (e.g. FGFR2 idx 642 -> "I", the
    residue after the brake lysine K641).
    """
    obj = dict_kinase["FGFR2"]
    idx = obj.KLIFS2UniProtIdx["VIII:79"]
    seq = obj.uniprot.canonical_seq
    # without the offset the mapped index is not the brake lysine
    assert seq[idx - 1] != "K"
    # the returned residue applies the -1 offset and recovers the lysine
    assert obj.return_molecular_brake_residues()["VIII:79"] == seq[idx - 2] == "K"


def test_molecular_brake_partial_match(dict_kinase):
    """EGFR keeps the conserved VIII:79 lysine but not the b.l/hinge brake residues."""
    obj = dict_kinase["EGFR"]
    assert obj.return_molecular_brake_residues() == {
        "b.l:37": "R",
        "hinge:46": "Q",
        "VIII:79": "K",
    }
    assert obj.check_molecular_brake_against_canonical() == (False, False, True)


def test_molecular_brake_no_klifs_mapping(mutable_kinase):
    """Both methods return None when no KLIFS2UniProt mapping is available."""
    obj = mutable_kinase("FGFR2")
    obj.KLIFS2UniProtIdx = None
    assert obj.return_molecular_brake_residues() is None
    assert obj.check_molecular_brake_against_canonical() is None


def test_molecular_brake_unmapped_position(mutable_kinase):
    """An unmapped brake position yields None and is treated as not matching."""
    obj = mutable_kinase("FGFR2")
    obj.KLIFS2UniProtIdx["b.l:37"] = None
    assert obj.return_molecular_brake_residues()["b.l:37"] is None
    # b.l:37 no longer matches; the other two positions still do
    assert obj.check_molecular_brake_against_canonical() == (False, True, True)


@pytest.mark.parametrize(
    "hgnc_name",
    ["FGFR1", "FGFR3", "FGFR4", "KIT", "PDGFRA"],
)
def test_molecular_brake_canonical_kinases(dict_kinase, hgnc_name):
    """Kinases with the intact molecular brake match all three canonical residues."""
    assert dict_kinase[hgnc_name].check_molecular_brake_against_canonical() == (
        True,
        True,
        True,
    )
