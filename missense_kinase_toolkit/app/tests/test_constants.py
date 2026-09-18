"""Unit tests for the app-level display constants in ``constants.py``."""

from constants import DICT_COMPUTED_HELP
from mkt.databases.app.properties import PropertyTables
from mkt.schema.io_utils import deserialize_kinase_dict


def test_every_computed_row_has_help():
    """Every computed-property row keys into DICT_COMPUTED_HELP."""
    table = PropertyTables(deserialize_kinase_dict(list_ids=["ATM"])["ATM"])
    assert set(table.dict_computed_keys.values()) <= set(DICT_COMPUTED_HELP)
