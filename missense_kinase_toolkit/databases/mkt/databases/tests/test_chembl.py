import pytest
from mkt.databases import chembl

# ---------------------------------------------------------------------------
# module-scoped fixtures – one API call per query type
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chembl_search_erlotinib():
    """ChEMBLMoleculeSearch for erlotinib (1 API call)."""
    return chembl.ChEMBLMoleculeSearch(id="erlotinib")


@pytest.fixture(scope="module")
def chembl_exact_erlotinib():
    """ChEMBLMoleculeExact for erlotinib (1 API call)."""
    return chembl.ChEMBLMoleculeExact(id="erlotinib")


@pytest.fixture(scope="module")
def chembl_preferred_erlotinib():
    """ChEMBLMoleculePreferred for erlotinib (1 API call)."""
    return chembl.ChEMBLMoleculePreferred(id="erlotinib")


@pytest.fixture(scope="module")
def chembl_search_invalid():
    """ChEMBLMoleculeSearch for non-existent drug (1 API call)."""
    return chembl.ChEMBLMoleculeSearch(id="TESTTESTTEST")


@pytest.fixture(scope="module")
def chembl_exact_invalid():
    """ChEMBLMoleculeExact for non-existent drug (1 API call)."""
    return chembl.ChEMBLMoleculeExact(id="TESTTESTTEST")


@pytest.fixture(scope="module")
def chembl_preferred_invalid():
    """ChEMBLMoleculePreferred for non-existent drug (1 API call)."""
    return chembl.ChEMBLMoleculePreferred(id="TESTTESTTEST")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestChEMBLErlotinib:
    def test_search_ids(self, chembl_search_erlotinib):
        assert set(chembl_search_erlotinib.get_chembl_id()) == {
            "CHEMBL1079742",
            "CHEMBL3186743",
            "CHEMBL5220042",
            "CHEMBL5220676",
            "CHEMBL553",
            "CHEMBL5965928",
        }

    def test_exact_id(self, chembl_exact_erlotinib):
        assert chembl_exact_erlotinib.get_chembl_id() == ["CHEMBL553"]

    def test_preferred_id(self, chembl_preferred_erlotinib):
        assert chembl_preferred_erlotinib.get_chembl_id() == ["CHEMBL553"]


@pytest.mark.network
class TestChEMBLInvalidDrug:
    def test_search_empty(self, chembl_search_invalid):
        assert chembl_search_invalid.get_chembl_id() == []

    def test_exact_empty(self, chembl_exact_invalid):
        assert chembl_exact_invalid.get_chembl_id() == []

    def test_preferred_empty(self, chembl_preferred_invalid):
        assert chembl_preferred_invalid.get_chembl_id() == []
