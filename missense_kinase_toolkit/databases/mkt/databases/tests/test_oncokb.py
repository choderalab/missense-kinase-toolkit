import pytest
from mkt.databases import oncokb
from mkt.databases.config import maybe_get_oncokb_token
from mkt.databases.oncokb import OncoKBCancerGeneList

# the OncoKB levels-of-evidence endpoint requires an access token; the skip below
# gates only the token-dependent class. to run those tests in CI, add the token as
# a GitHub Actions secret (e.g. ONCOKB_TOKEN) and export it before pytest runs.
_requires_oncokb_token = pytest.mark.skipif(
    maybe_get_oncokb_token() is None,
    reason="OncoKB API token not set (ONCOKB_TOKEN); skipping live OncoKB tests",
)


@pytest.fixture(scope="module")
def oncokb_levels():
    """Query the OncoKB API for the levels of evidence once."""
    return oncokb.get_oncokb_levels()


@pytest.fixture(scope="module")
def gene_list():
    """Query the OncoKB cancer gene list once."""
    return OncoKBCancerGeneList()


@_requires_oncokb_token
@pytest.mark.network
class TestOncoKBLevels:
    def test_levels_retrieved(self, oncokb_levels):
        assert oncokb_levels is not None
        assert isinstance(oncokb_levels, dict)
        assert len(oncokb_levels) > 0

    def test_levels_are_named_descriptions(self, oncokb_levels):
        # every level is a "LEVEL_*" key mapped to a non-empty description string
        for level, description in oncokb_levels.items():
            assert level.startswith("LEVEL_")
            assert isinstance(description, str) and description

    @pytest.mark.parametrize(
        "level",
        ["LEVEL_1", "LEVEL_R1", "LEVEL_Dx1", "LEVEL_Px1", "LEVEL_Fda1"],
    )
    def test_expected_levels_present(self, oncokb_levels, level):
        # one representative level from each category (Tx, R, Dx, Px, FDA)
        assert level in oncokb_levels


# the cancer gene list endpoint (/utils/cancerGeneList) is public and needs no
# token, so these tests are gated only on the `network` marker.
@pytest.mark.network
class TestOncoKBCancerGeneList:
    def test_records_populated(self, gene_list):
        assert gene_list._json is not None
        assert len(gene_list.df) == len(gene_list._json)

    def test_expected_columns(self, gene_list):
        assert {"hugoSymbol", "geneType", "geneAliases"}.issubset(gene_list.df.columns)

    def test_get_gene_braf(self, gene_list):
        braf = gene_list.get_gene("BRAF")
        assert len(braf) == 1
        assert braf.iloc[0]["geneType"] == "ONCOGENE"

    def test_get_gene_missing_returns_empty(self, gene_list):
        assert gene_list.get_gene("NOTAGENE").empty

    def test_csv_roundtrip_preserves_list_columns(self, gene_list, tmp_path):
        path = tmp_path / "oncokb_cancer_genes.csv"
        gene_list.to_csv(str(path))

        reloaded = OncoKBCancerGeneList.from_csv(str(path))
        assert len(reloaded.df) == len(gene_list.df)
        # geneAliases survives the round-trip as a list, not a json string
        braf = reloaded.get_gene("BRAF").iloc[0]
        assert isinstance(braf["geneAliases"], list)
        assert "BRAF1" in braf["geneAliases"]

    def test_from_dataframe_skips_query(self, gene_list):
        offline = OncoKBCancerGeneList.from_dataframe(gene_list.df)
        assert offline._json is None
        assert offline.query_datetime is None
        assert not offline.get_gene("BRAF").empty
