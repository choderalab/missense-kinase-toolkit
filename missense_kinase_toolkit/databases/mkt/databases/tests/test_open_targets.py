import pytest
from mkt.databases import open_targets


@pytest.fixture(scope="module")
def drug_moa_erlotinib():
    """Query OpenTargets drug MoA for CHEMBL1079742 once."""
    return open_targets.OpenTargetsDrugMoA(chembl_id="CHEMBL1079742")


@pytest.fixture(scope="module")
def drug_moa_invalid():
    """Query OpenTargets drug MoA for invalid ChEMBL ID once."""
    return open_targets.OpenTargetsDrugMoA(chembl_id="TEST")


@pytest.mark.network
class TestOpenTargetsDrugMoA:
    def test_erlotinib_targets_egfr(self, drug_moa_erlotinib):
        assert drug_moa_erlotinib.get_moa() == {"EGFR"}

    def test_invalid_id_returns_none(self, drug_moa_invalid):
        assert drug_moa_invalid.get_moa() is None
