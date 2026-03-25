import pytest


@pytest.mark.network
class TestKLIFSKinaseInfo:
    def test_single_kinase_result(self, egfr_klifs_info):
        assert len(egfr_klifs_info._kinase_info) == 1

    def test_family(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["family"] == "EGFR"

    def test_full_name(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert (
            egfr_klifs_info.get_kinase_info()[0]["full_name"]
            == "epidermal growth factor receptor"
        )

    def test_gene_name(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["gene_name"] == "EGFR"

    def test_group(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["group"] == "TK"

    def test_iuphar(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["iuphar"] == 1797

    def test_kinase_id(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["kinase_ID"] == 406

    def test_name(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["name"] == "EGFR"

    def test_pocket_sequence(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert (
            egfr_klifs_info.get_kinase_info()[0]["pocket"]
            == "KVLGSGAFGTVYKVAIKELEILDEAYVMASVDPHVCRLLGIQLITQLMPFGCLLDYVREYLEDRRLVHRDLAARNVLVITDFGLA"
        )

    def test_species(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["species"] == "Human"

    def test_uniprot_id(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["uniprot"] == "P00533"

    def test_server_error_returns_none(self, egfr_klifs_info):
        """When KLIFS returns 5xx, get_kinase_info()[0] should be None."""
        if 500 <= egfr_klifs_info.status_code < 600:
            assert egfr_klifs_info.get_kinase_info()[0] is None
