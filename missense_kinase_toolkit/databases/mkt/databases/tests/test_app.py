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


def test_computed_table_motifs_name_index_source():
    """Catalytic motifs follow return_catalytic_residues (KLIFS, else MSA) and name it.

    Regression: KLIFS-less kinases such as PEAK3 dropped the catalytic Lys/HRD/DFG rows.
    """
    from mkt.databases.app.properties import PropertyTables
    from mkt.schema.io_utils import deserialize_kinase_dict

    dict_obj = deserialize_kinase_dict(list_ids=["ABL1", "PEAK3", "ALPK1"])

    ser_abl1 = PropertyTables(dict_obj["ABL1"]).df_computed["Property"]
    assert ser_abl1["CATALYTIC LYS (KLIFS)"] == "K271"
    assert ser_abl1["HRD MOTIF (KLIFS)"] == "H361-R362-D363"
    assert ser_abl1["DFG MOTIF (KLIFS)"] == "D381-F382-G383"

    ser_peak3 = PropertyTables(dict_obj["PEAK3"]).df_computed["Property"]
    assert ser_peak3["CATALYTIC LYS (MSA)"] == "K204"
    assert ser_peak3["HRD MOTIF (MSA)"] == "L302-V303-E304"
    assert ser_peak3["DFG MOTIF (MSA)"] == "D330-F331-G332"
    # the molecular brake is KLIFS-only, so it stays unresolved without a KLIFS mapping
    assert ser_peak3["MOLECULAR BRAKE N-E-K (KLIFS)"] == "None"

    # neither alignment: rows are kept (no source suffix) and read "None"
    ser_alpk1 = PropertyTables(dict_obj["ALPK1"]).df_computed["Property"]
    assert ser_alpk1["HRD MOTIF"] == "None"
    assert ser_alpk1["DFG MOTIF"] == "None"


def test_computed_table_reversed_hrd_and_help():
    """PIK/PIKK HRD reads c.l:72-71-70, and every computed row has a base property key."""
    from mkt.databases.app.properties import PropertyTables
    from mkt.schema.io_utils import deserialize_kinase_dict

    dict_obj = deserialize_kinase_dict(list_ids=["ATM", "PI4K2A"])

    table = PropertyTables(dict_obj["ATM"])
    assert table.df_computed["Property"]["HRD MOTIF (KLIFS)"] == "H2872-R2871-D2870"
    assert set(table.dict_computed_keys) == set(table.df_computed.index)
    assert table.dict_computed_keys["HRD MOTIF (KLIFS)"] == "HRD motif"

    ser_pi4k2a = PropertyTables(dict_obj["PI4K2A"]).df_computed["Property"]
    assert ser_pi4k2a["HRD MOTIF (KLIFS)"] == "G310-R309-D308"
