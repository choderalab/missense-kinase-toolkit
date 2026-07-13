import logging

import pytest


def test_adjudicate_group(mutable_kinase, caplog):
    """Test kinase group adjudication across data-source priorities.

    Uses ``mutable_kinase`` because the final case sets ``ANTXR1.klifs = None``
    to exercise the no-group-found path.
    """
    caplog.set_level(logging.INFO)

    assert mutable_kinase("ABL1").adjudicate_group() == "TK"  # Kincore
    assert mutable_kinase("ABR").adjudicate_group() == "Atypical"  # KinHub

    obj_antxr1 = mutable_kinase("ANTXR1")
    assert obj_antxr1.adjudicate_group() == "Atypical"  # KLIFS

    # remove KLIFS so no source can supply a group
    obj_antxr1.klifs = None
    caplog.clear()
    assert obj_antxr1.adjudicate_group(bool_verbose=True) is None
    assert "No group found for ANTXR1" in caplog.text


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
    """Gaps within the default cut-off expand the bound to the KLIFS index."""
    obj = dict_kinase[hgnc_name]
    assert obj.adjudicate_kd_start() == expected_start
    assert obj.adjudicate_kd_end() == expected_end


def test_adjudicate_kd_large_gap_returns_none(dict_kinase, caplog):
    """Gaps beyond the cut-off return None and warn that the KD exists."""
    caplog.set_level(logging.WARNING)

    # EIF2AK4_2 start gap is 46 (> default 15)
    assert dict_kinase["EIF2AK4_2"].adjudicate_kd_start() is None
    assert "Kinase domain start found for EIF2AK4_2" in caplog.text
    assert "larger than cut-off 15" in caplog.text

    # MTOR KLIFS pocket is disjoint from the KD; caught on the end bound
    caplog.clear()
    assert dict_kinase["MTOR"].adjudicate_kd_end() is None
    assert "Kinase domain end found for MTOR" in caplog.text


def test_adjudicate_kd_cutoff_parameter(dict_kinase):
    """Raising int_max_gap expands bounds that would otherwise return None."""
    # EIF2AK4_2 start gap of 46 expands once the cut-off allows it
    assert dict_kinase["EIF2AK4_2"].adjudicate_kd_start(int_max_gap=50) == 284
    # MTOR end gap of 1337 expands with a large enough cut-off
    assert dict_kinase["MTOR"].adjudicate_kd_end(int_max_gap=2000) == 2361


def test_adjudicate_kd_verbose_logs_expansion(dict_kinase, caplog):
    """Verbose mode logs an info message when a bound is expanded."""
    caplog.set_level(logging.INFO)
    assert dict_kinase["BUB1B"].adjudicate_kd_start(bool_verbose=True) == 759
    assert "expanding start to 759" in caplog.text
