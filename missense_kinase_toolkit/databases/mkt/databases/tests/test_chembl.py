import pytest
import requests as req_lib
from mkt.databases import chembl

STR_CHEMBL_STATUS_URL = "https://www.ebi.ac.uk/chembl/api/data/status.json"
"""str: ChEMBL API health endpoint, probed once before any query."""

# ---------------------------------------------------------------------------
# module-scoped fixtures – one API call per query type
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def chembl_available():
    """Skip the module up front if ChEMBL is down, instead of retrying each query."""
    # plain requests: no retries or caching, so an outage costs one short request
    try:
        res = req_lib.get(STR_CHEMBL_STATUS_URL, timeout=(5, 15))
    except req_lib.exceptions.RequestException as e:
        pytest.skip(f"ChEMBL API unreachable - skipping tests ({e})")
    if not res.ok:
        pytest.skip(f"ChEMBL API unhealthy (HTTP {res.status_code}) - skipping tests")


def _build_chembl_query(query_class, query_id):
    """Build ChEMBL query object, skipping on transient request failures."""
    try:
        return query_class(id=query_id)
    except req_lib.exceptions.RequestException as e:
        if "500 error responses" in str(e) or "Failed to resolve" in str(e):
            pytest.skip("ChEMBL API request failed - skipping test")
        raise


@pytest.fixture(scope="module")
def chembl_search_erlotinib():
    """ChEMBLMoleculeSearch for erlotinib (1 API call)."""
    return _build_chembl_query(chembl.ChEMBLMoleculeSearch, "erlotinib")


@pytest.fixture(scope="module")
def chembl_exact_erlotinib():
    """ChEMBLMoleculeExact for erlotinib (1 API call)."""
    return _build_chembl_query(chembl.ChEMBLMoleculeExact, "erlotinib")


@pytest.fixture(scope="module")
def chembl_preferred_erlotinib():
    """ChEMBLMoleculePreferred for erlotinib (1 API call)."""
    return _build_chembl_query(chembl.ChEMBLMoleculePreferred, "erlotinib")


@pytest.fixture(scope="module")
def chembl_search_invalid():
    """ChEMBLMoleculeSearch for non-existent drug (1 API call)."""
    return _build_chembl_query(chembl.ChEMBLMoleculeSearch, "TESTTESTTEST")


@pytest.fixture(scope="module")
def chembl_exact_invalid():
    """ChEMBLMoleculeExact for non-existent drug (1 API call)."""
    return _build_chembl_query(chembl.ChEMBLMoleculeExact, "TESTTESTTEST")


@pytest.fixture(scope="module")
def chembl_preferred_invalid():
    """ChEMBLMoleculePreferred for non-existent drug (1 API call)."""
    return _build_chembl_query(chembl.ChEMBLMoleculePreferred, "TESTTESTTEST")


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
