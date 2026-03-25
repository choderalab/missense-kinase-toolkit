import pytest
from mkt.databases.protvar import ProtvarScore


@pytest.fixture(scope="module")
def protvar_abl1():
    """Query ProtVar for ABL1 P00519 pos 292 D once."""
    return ProtvarScore(database="AM", uniprot_id="P00519", pos=292, mut="D")


@pytest.mark.network
class TestProtvarScore:
    def test_single_result(self, protvar_abl1):
        assert len(protvar_abl1._protvar_score) == 1

    def test_pathogenicity_score(self, protvar_abl1):
        assert protvar_abl1._protvar_score[0]["amPathogenicity"] == 0.4217

    def test_pathogenicity_class(self, protvar_abl1):
        assert protvar_abl1._protvar_score[0]["amClass"] == "AMBIGUOUS"
