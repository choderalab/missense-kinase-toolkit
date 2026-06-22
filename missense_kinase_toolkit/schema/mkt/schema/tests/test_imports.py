def test_mkt_schema_imported():
    """Test that the mkt.schema module is importable."""
    import sys

    import mkt.schema  # noqa F401

    assert "mkt.schema" in sys.modules
