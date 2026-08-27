import logging

import pytest


def test_adjudicate_group(mutable_kinase, caplog):
    """Test kinase group adjudication across data-source priorities.

    Uses ``mutable_kinase`` because the final case sets ``PI4KA.klifs = None``
    to exercise the no-group-found path.
    """
    caplog.set_level(logging.INFO)

    assert mutable_kinase("ABL1").adjudicate_group() == "TK"  # Kincore
    assert mutable_kinase("ADCK1").adjudicate_group() == "Atypical"  # KinHub

    obj_pi4ka = mutable_kinase("PI4KA")
    assert obj_pi4ka.adjudicate_group() == "Atypical"  # KLIFS

    # remove KLIFS so no source can supply a group
    obj_pi4ka.klifs = None
    caplog.clear()
    assert obj_pi4ka.adjudicate_group(bool_verbose=True) is None
    assert "No group found for PI4KA" in caplog.text


def test_adjudicate_kd_clean_bounds(dict_kinase):
    """A kinase whose KLIFS pocket falls inside the KD is returned unchanged."""
    obj = dict_kinase["ABL1"]
    # KLIFS indices 246-385 fall within the adjudicated KD 234-503
    assert obj.adjudicate_kd_start() == 234
    assert obj.adjudicate_kd_end() == 503


def test_adjudicate_kd_no_klifs(mutable_kinase):
    """Missing KLIFS2UniProtIdx leaves the adjudicated bounds untouched."""
    obj = mutable_kinase("ABL1")
    obj.KLIFS2UniProtIdx = None
    assert obj.adjudicate_kd_start() == 234
    assert obj.adjudicate_kd_end() == 503


@pytest.mark.parametrize(
    "hgnc_name, expected_start, expected_end",
    [
        ("NPR1", 532, 803),  # start gap of 1 -> expand start
        ("NPR2", 516, 788),  # start gap of 1 -> expand start
        ("RPS6KL1", 149, 539),  # start gap of 1 -> expand start
        ("BUB1B", 759, 1021),  # start gap of 7 -> expand start
        ("GUCY2F", 536, 811),  # start gap of 11 -> expand start
    ],
)
def test_adjudicate_kd_small_gap_expands(
    dict_kinase, hgnc_name, expected_start, expected_end
):
    """A KLIFS pocket extending past the bound expands it to the KLIFS index."""
    obj = dict_kinase[hgnc_name]
    assert obj.adjudicate_kd_start() == expected_start
    assert obj.adjudicate_kd_end() == expected_end


def test_adjudicate_kd_large_gap_expands_by_default(dict_kinase):
    """With the default infinite cut-off, large gaps expand to the KLIFS index.

    These bounds returned None under the historical finite cut-off; the KLIFS
    pocket is now trusted as the better-annotated bound (large kinase-domain
    inserts missed by Pfam but present in KLIFS).
    """
    # EIF2AK4_2 start gap of 46 expands to the KLIFS minimum
    assert dict_kinase["EIF2AK4_2"].adjudicate_kd_start() == 284
    # ADCK2 end gap (Pfam end 218 -> KLIFS max) expands to the KLIFS maximum
    assert dict_kinase["ADCK2"].adjudicate_kd_end() == 497


def test_adjudicate_kd_finite_cutoff_returns_none(dict_kinase, caplog):
    """An explicit finite int_max_gap still returns None and warns."""
    caplog.set_level(logging.WARNING)

    # EIF2AK4_2 start gap is 46 (> explicit cut-off of 15)
    assert dict_kinase["EIF2AK4_2"].adjudicate_kd_start(int_max_gap=15) is None
    assert "Kinase domain start found for EIF2AK4_2" in caplog.text
    assert "larger than cut-off 15" in caplog.text

    # a large-but-finite cut-off still expands the ADCK2 end bound
    assert dict_kinase["ADCK2"].adjudicate_kd_end(int_max_gap=2000) == 497


def test_adjudicate_kd_verbose_logs_expansion(dict_kinase, caplog):
    """Verbose mode logs an info message when a bound is expanded."""
    caplog.set_level(logging.INFO)
    assert dict_kinase["BUB1B"].adjudicate_kd_start(bool_verbose=True) == 759
    assert "expanding start to 759" in caplog.text


def test_molecular_brake_residues(dict_kinase):
    """Brake residues are read via KLIFS2UniProtIdx with the VIII:79 -1 offset.

    FGFR2 carries the full canonical N-E-K triad; EGFR keeps only the conserved
    VIII:79 lysine. The brake lysine sits one residue N-terminal to its VIII:79
    KLIFS-aligned index, so the -1 offset must be applied -- the raw mapped index
    resolves to a non-conserved residue.
    """
    obj = dict_kinase["FGFR2"]
    assert obj.return_molecular_brake_residues() == {
        "b.l:37": "N",
        "hinge:46": "E",
        "VIII:79": "K",
    }
    # the -1 offset recovers the lysine; the raw mapped index does not
    idx = obj.KLIFS2UniProtIdx["VIII:79"]
    seq = obj.uniprot.canonical_seq
    assert seq[idx - 1] != "K" and seq[idx - 2] == "K"

    assert dict_kinase["EGFR"].return_molecular_brake_residues() == {
        "b.l:37": "R",
        "hinge:46": "Q",
        "VIII:79": "K",
    }


@pytest.mark.parametrize(
    "hgnc_name, expected",
    [
        ("FGFR2", (True, True, True)),  # canonical FGFR brake
        ("FGFR1", (True, True, True)),
        ("KIT", (True, True, True)),
        ("PDGFRA", (True, True, True)),
        ("EGFR", (False, False, True)),  # only the VIII:79 lysine is conserved
    ],
)
def test_molecular_brake_against_canonical(dict_kinase, hgnc_name, expected):
    """Brake residues are compared position-wise against the canonical N-E-K."""
    assert dict_kinase[hgnc_name].check_molecular_brake_against_canonical() == expected


def test_molecular_brake_missing_mapping(mutable_kinase):
    """Absent or unmapped KLIFS positions yield None rather than raising."""
    obj = mutable_kinase("FGFR2")
    obj.KLIFS2UniProtIdx = None
    assert obj.return_molecular_brake_residues() is None
    assert obj.check_molecular_brake_against_canonical() is None

    obj = mutable_kinase("FGFR2")
    obj.KLIFS2UniProtIdx["b.l:37"] = None
    assert obj.return_molecular_brake_residues()["b.l:37"] is None
    # the unmapped position no longer matches; the other two still do
    assert obj.check_molecular_brake_against_canonical() == (False, True, True)


def test_adjudicate_kd_sequence_one_to_one_with_bounds(dict_kinase):
    """adjudicate_kd_sequence is exactly the canonical UniProt slice over the KD bounds.

    Guards the invariant that the KD sequence, the KD start/end, and (by construction) the
    KD-sliced AlphaFold structure are 1-to-1: ``seq == canonical_seq[start - 1 : end]`` and
    ``len(seq) == end - start + 1``.
    """
    for obj in dict_kinase.values():
        seq = obj.adjudicate_kd_sequence()
        start, end = obj.adjudicate_kd_start(), obj.adjudicate_kd_end()
        if seq is None:
            assert start is None or end is None
            continue
        assert len(seq) == end - start + 1
        assert seq == obj.uniprot.canonical_seq[start - 1 : end]
