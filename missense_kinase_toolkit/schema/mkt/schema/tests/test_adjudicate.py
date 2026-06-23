import logging


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
