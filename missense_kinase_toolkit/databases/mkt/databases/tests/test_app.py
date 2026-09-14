"""Tests for the Streamlit-backing mkt.databases.app helpers."""

import logging


def test_missing_sources_do_not_log_errors(caplog):
    """Absent KinHub/KLIFS data for a kinase is expected, not an error or info log.

    Regression: PEAK3 (no KinHub, KLIFS, or KLIFS2UniProtIdx) logged ERRORs from
    PropertyTables and INFO/ERROR start/end parse messages from SequenceAlignment.
    """
    from mkt.databases.app.properties import PropertyTables
    from mkt.databases.app.sequences import SequenceAlignment
    from mkt.databases.colors import DICT_COLORS
    from mkt.schema.io_utils import deserialize_kinase_dict

    obj = deserialize_kinase_dict(list_ids=["PEAK3"])["PEAK3"]
    dict_color = next(iter(DICT_COLORS.values()))["DICT_COLORS"]

    caplog.set_level(logging.INFO, logger="mkt.databases.app")
    table = PropertyTables(obj)
    SequenceAlignment(str_kinase="PEAK3", dict_color=dict_color, obj_kinase=obj)

    assert table.df_kinhub is None and table.df_klifs is None
    assert table.df_computed is not None
    assert [r for r in caplog.records if r.name.startswith("mkt.databases.app")] == []
