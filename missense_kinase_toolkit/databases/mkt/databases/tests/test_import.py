class TestImport:
    def test_missense_kinase_toolkit_database_imported(self):
        """Test if module is imported."""
        import sys

        import mkt.databases  # noqa F401

        assert "mkt.databases" in sys.modules
