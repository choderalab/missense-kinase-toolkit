import pytest
from mkt.databases import hgnc

# ---------------------------------------------------------------------------
# module-scoped fixtures – one API call per HGNC query
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hgnc_abl1_by_name():
    """Search HGNC for ABL1 by gene name (1 search + 1 fetch call)."""
    obj = hgnc.HGNC("Abl1", True)
    obj.maybe_get_symbol_from_hgnc_search()
    return {"obj": obj, "fetch": obj.maybe_get_info_from_hgnc_fetch()}


@pytest.fixture(scope="module")
def hgnc_abl1_by_ensembl():
    """Search HGNC for ABL1 by Ensembl ID (1 search + 1 fetch call)."""
    obj = hgnc.HGNC("ENSG00000097007", False)
    obj.maybe_get_symbol_from_hgnc_search()
    return {"obj": obj, "fetch": obj.maybe_get_info_from_hgnc_fetch()}


@pytest.fixture(scope="module")
def hgnc_invalid_by_name():
    """Search HGNC with invalid gene name (1 search + 1 fetch call)."""
    obj = hgnc.HGNC("test", True)
    obj.maybe_get_symbol_from_hgnc_search()
    return {"obj": obj, "fetch": obj.maybe_get_info_from_hgnc_fetch()}


@pytest.fixture(scope="module")
def hgnc_invalid_by_ensembl():
    """Search HGNC with invalid Ensembl ID (1 search + 1 fetch call)."""
    obj = hgnc.HGNC("test", False)
    obj.maybe_get_symbol_from_hgnc_search()
    return {"obj": obj, "fetch": obj.maybe_get_info_from_hgnc_fetch()}


@pytest.fixture(scope="module")
def hgnc_abl1_by_custom_field():
    """Search HGNC using custom mane_select field (1 search call)."""
    obj = hgnc.HGNC("temp")
    obj.maybe_get_symbol_from_hgnc_search(
        custom_field="mane_select", custom_term="ENST00000318560.6"
    )
    return obj


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestHGNCSearchByName:
    def test_symbol(self, hgnc_abl1_by_name):
        assert hgnc_abl1_by_name["obj"].hgnc == "ABL1"

    def test_locus_type(self, hgnc_abl1_by_name):
        assert (
            hgnc_abl1_by_name["fetch"]["locus_type"][0] == "gene with protein product"
        )


@pytest.mark.network
class TestHGNCSearchByEnsembl:
    def test_symbol(self, hgnc_abl1_by_ensembl):
        assert hgnc_abl1_by_ensembl["obj"].hgnc == "ABL1"

    def test_locus_type(self, hgnc_abl1_by_ensembl):
        assert (
            hgnc_abl1_by_ensembl["fetch"]["locus_type"][0]
            == "gene with protein product"
        )


@pytest.mark.network
class TestHGNCInvalidName:
    def test_ensembl_is_none(self, hgnc_invalid_by_name):
        assert hgnc_invalid_by_name["obj"].ensembl is None

    def test_locus_type_is_none(self, hgnc_invalid_by_name):
        assert hgnc_invalid_by_name["fetch"]["locus_type"] is None


@pytest.mark.network
class TestHGNCInvalidEnsembl:
    def test_hgnc_is_none(self, hgnc_invalid_by_ensembl):
        assert hgnc_invalid_by_ensembl["obj"].hgnc is None

    def test_fetch_is_none(self, hgnc_invalid_by_ensembl):
        assert hgnc_invalid_by_ensembl["fetch"] is None


@pytest.mark.network
class TestHGNCCustomFieldSearch:
    def test_custom_mane_select_finds_abl1(self, hgnc_abl1_by_custom_field):
        assert hgnc_abl1_by_custom_field.hgnc == "ABL1"
