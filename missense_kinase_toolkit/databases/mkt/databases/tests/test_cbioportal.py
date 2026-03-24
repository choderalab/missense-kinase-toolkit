import pytest
from mkt.databases import cbioportal, config


@pytest.fixture(scope="module")
def cbioportal_instance():
    """Create a cBioPortal client once for this module."""
    config.set_cbioportal_instance("www.cbioportal.org")
    config.set_output_dir(".")
    return cbioportal.cBioPortal()


@pytest.fixture(scope="module")
def mutations_instance():
    """Query MSK-IMPACT 2017 mutations once."""
    config.set_cbioportal_instance("www.cbioportal.org")
    config.set_output_dir(".")
    return cbioportal.Mutations(study_id="msk_impact_2017")


@pytest.fixture(scope="module")
def gene_panel_instance():
    """Query IMPACT341 gene panel once."""
    config.set_cbioportal_instance("www.cbioportal.org")
    config.set_output_dir(".")
    return cbioportal.GenePanel(panel_id="IMPACT341")


@pytest.mark.network
class TestCBioPortalClient:
    def test_instance_url(self, cbioportal_instance):
        assert cbioportal_instance.get_instance() == "www.cbioportal.org"

    def test_api_docs_url(self, cbioportal_instance):
        assert (
            cbioportal_instance.get_url()
            == "https://www.cbioportal.org/api/v2/api-docs"
        )

    def test_client_not_none(self, cbioportal_instance):
        assert cbioportal_instance._cbioportal is not None

    def test_server_status_up(self, cbioportal_instance):
        status = (
            cbioportal_instance._cbioportal.Server_running_status.getServerStatusUsingGET()
            .response()
            .result["status"]
        )
        assert status == "UP"


@pytest.mark.network
class TestMutations:
    def test_entity_id_exists(self, mutations_instance):
        assert mutations_instance.check_entity_id() is True

    def test_entity_id_value(self, mutations_instance):
        assert mutations_instance.get_entity_id() == "msk_impact_2017"

    def test_mutation_count(self, mutations_instance):
        assert mutations_instance._df.shape[0] == 78142


@pytest.mark.network
class TestGenePanel:
    def test_panel_entity_id_exists(self, gene_panel_instance):
        assert gene_panel_instance.check_entity_id() is True

    def test_panel_row_count(self, gene_panel_instance):
        assert gene_panel_instance._df.shape[0] == 341

    def test_panel_column_count(self, gene_panel_instance):
        assert gene_panel_instance._df.shape[1] == 2
