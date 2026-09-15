import pytest
from mkt.databases import scrapers


@pytest.fixture(scope="module")
def kinhub_df():
    """Scrape KinHub once."""
    return scrapers.kinhub()


@pytest.mark.network
class TestKinHub:
    def test_row_count(self, kinhub_df):
        assert kinhub_df.shape[0] == 536

    def test_column_count(self, kinhub_df):
        assert kinhub_df.shape[1] == 8

    def test_has_hgnc_column(self, kinhub_df):
        assert "HGNC Name" in kinhub_df.columns

    def test_has_uniprot_column(self, kinhub_df):
        assert "UniprotID" in kinhub_df.columns
