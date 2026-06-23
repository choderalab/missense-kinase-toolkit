import pytest
from mkt.databases.alphafold import AlphaFoldPrediction, AlphaFoldStructure

# ---------------------------------------------------------------------------
# module-scoped fixtures – one API call per query
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def af_prediction_abl1():
    """Fetch AlphaFold prediction for ABL1 (P00519) once."""
    return AlphaFoldPrediction(uniprot_id="P00519")


@pytest.fixture(scope="module")
def af_prediction_egfr():
    """Fetch AlphaFold prediction for EGFR (P00533) once."""
    return AlphaFoldPrediction(uniprot_id="P00533")


@pytest.fixture(scope="module")
def af_prediction_invalid():
    """Fetch AlphaFold prediction for invalid ID once."""
    return AlphaFoldPrediction(uniprot_id="INVALID_ID_12345")


@pytest.fixture(scope="module")
def af_structure_abl1():
    """Fetch AlphaFold structure for ABL1 (P00519) once."""
    return AlphaFoldStructure(uniprot_id="P00519")


@pytest.fixture(scope="module")
def af_structure_invalid():
    """Fetch AlphaFold structure for invalid ID once."""
    return AlphaFoldStructure(uniprot_id="INVALID_ID_12345")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestAlphaFoldPredictionSingleIsoform:
    def test_json_not_none(self, af_prediction_abl1):
        assert af_prediction_abl1._json is not None

    def test_uniprot_accession(self, af_prediction_abl1):
        assert af_prediction_abl1._json["uniprotAccession"] == "P00519"

    def test_has_cif_url(self, af_prediction_abl1):
        assert "cifUrl" in af_prediction_abl1._json


@pytest.mark.network
class TestAlphaFoldPredictionMultipleIsoforms:
    def test_json_not_none(self, af_prediction_egfr):
        assert af_prediction_egfr._json is not None

    def test_uniprot_accession(self, af_prediction_egfr):
        assert af_prediction_egfr._json["uniprotAccession"] == "P00533"

    def test_has_cif_url(self, af_prediction_egfr):
        assert "cifUrl" in af_prediction_egfr._json


@pytest.mark.network
class TestAlphaFoldPredictionInvalid:
    def test_json_is_none(self, af_prediction_invalid):
        assert af_prediction_invalid._json is None


@pytest.mark.network
class TestAlphaFoldStructureDownload:
    def test_json_not_none(self, af_structure_abl1):
        assert af_structure_abl1._json is not None

    def test_cif_not_none(self, af_structure_abl1):
        assert af_structure_abl1._cif is not None

    def test_cif_starts_with_data(self, af_structure_abl1):
        assert af_structure_abl1._cif.startswith("data_")


@pytest.mark.network
class TestAlphaFoldStructureInvalid:
    def test_json_is_none(self, af_structure_invalid):
        assert af_structure_invalid._json is None

    def test_cif_is_none(self, af_structure_invalid):
        assert af_structure_invalid._cif is None
