import pytest
from mkt.databases.oncokb import OncoKBCancerGeneList

# the cancer gene list endpoint (/utils/cancerGeneList) is public and needs no
# token, so these tests are gated only on the `network` marker.


@pytest.fixture(scope="module")
def gene_list():
    """Query the OncoKB cancer gene list once."""
    return OncoKBCancerGeneList()


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
