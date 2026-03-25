import pytest
import requests as req_lib
from mkt.databases import pfam


@pytest.fixture(scope="module")
def pfam_abl1():
    """Fetch Pfam data for ABL1 (P00519) once."""
    try:
        obj = pfam.Pfam("P00519")
        return obj._pfam
    except req_lib.exceptions.RetryError as e:
        if "500 error responses" in str(e):
            pytest.skip("Pfam API returned 500 errors - skipping test")
        raise


@pytest.mark.network
class TestPfam:
    def test_row_count(self, pfam_abl1):
        assert pfam_abl1.shape[0] == 4

    def test_column_count(self, pfam_abl1):
        # allow for 18 or 19 columns depending on Pfam database version
        assert pfam_abl1.shape[1] == 18 or pfam_abl1.shape[1] == 19

    def test_has_uniprot_column(self, pfam_abl1):
        assert "uniprot" in pfam_abl1.columns

    def test_has_start_column(self, pfam_abl1):
        assert "start" in pfam_abl1.columns

    def test_has_end_column(self, pfam_abl1):
        assert "end" in pfam_abl1.columns

    def test_has_name_column(self, pfam_abl1):
        assert "name" in pfam_abl1.columns

    def test_kinase_domain_start(self, pfam_abl1):
        assert (
            pfam_abl1.loc[
                pfam_abl1["name"] == "Protein tyrosine and serine/threonine kinase",
                "start",
            ].values[0]
            == 242
        )

    def test_kinase_domain_end(self, pfam_abl1):
        assert (
            pfam_abl1.loc[
                pfam_abl1["name"] == "Protein tyrosine and serine/threonine kinase",
                "end",
            ].values[0]
            == 492
        )

    def test_find_pfam_domain(self, pfam_abl1):
        assert (
            pfam.find_pfam_domain(
                input_id="p00519",
                input_position=350,
                df_ref=pfam_abl1,
                col_ref_id="uniprot",
            )
            == "Protein tyrosine and serine/threonine kinase"
        )
