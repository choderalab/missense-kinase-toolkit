import pytest
from mkt.databases.protvar import ProtvarScoreQuery, ProtvarVariant, ScoreDatabase


@pytest.fixture(scope="module")
def protvar_abl1_single():
    """Query ProtVar for ABL1 P00519 pos 292 mut D once."""
    return ProtvarScoreQuery(uniprot_id="P00519", pos=292, mut="D")


@pytest.fixture(scope="module")
def protvar_abl1_all():
    """Query ProtVar for ABL1 P00519 pos 292, all substitutions, once."""
    return ProtvarScoreQuery(uniprot_id="P00519", pos=292, mut=None)


@pytest.mark.network
class TestProtvarSingleVariant:
    def test_single_variant_built(self, protvar_abl1_single):
        assert set(protvar_abl1_single.variants) == {"D"}

    def test_get_variant_no_arg(self, protvar_abl1_single):
        variant = protvar_abl1_single.get_variant()
        assert isinstance(variant, ProtvarVariant)
        assert variant.mt == "D"

    def test_get_score_scalar(self, protvar_abl1_single):
        variant = protvar_abl1_single.get_variant()
        assert variant.get_score(ScoreDatabase.Conservation) == 0.653
        assert variant.get_score(ScoreDatabase.EVE) == 0.59
        assert variant.get_score(ScoreDatabase.ESM1b) == -7.494
        assert variant.get_score("AM") == 0.4217

    def test_get_score_none(self, protvar_abl1_single):
        variant = protvar_abl1_single.get_variant()
        # popEVE has no scalar score; unknown databases return None
        assert variant.get_score(ScoreDatabase.popEVE) is None
        assert variant.get_score("NOT_A_DB") is None

    def test_get_classification(self, protvar_abl1_single):
        variant = protvar_abl1_single.get_variant()
        assert variant.get_classification(ScoreDatabase.EVE) == "UNCERTAIN"
        assert variant.get_classification("AM") == "AMBIGUOUS"
        # Conservation, ESM1b, popEVE carry no classification
        assert variant.get_classification(ScoreDatabase.Conservation) is None
        assert variant.get_classification(ScoreDatabase.ESM1b) is None
        assert variant.get_classification(ScoreDatabase.popEVE) is None

    def test_get_popeve(self, protvar_abl1_single):
        popeve = protvar_abl1_single.get_variant().get_popeve()
        assert isinstance(popeve, dict)
        assert popeve["popeve"] == -5.024


@pytest.mark.network
class TestProtvarAllVariants:
    def test_one_variant_per_substitution(self, protvar_abl1_all):
        # 19 substitutions: the 20 amino acids minus wild-type L at pos 292
        expected = set("ACDEFGHIKLMNPQRSTVWY") - {"L"}
        assert set(protvar_abl1_all.variants) == expected

    def test_residue_labels_recovered(self, protvar_abl1_all):
        # variant-level scores are correctly tied back to their residue
        assert protvar_abl1_all.get_variant("A").get_score("AM") == 0.6943
        assert protvar_abl1_all.get_variant("D").get_score("AM") == 0.4217

    def test_conservation_shared(self, protvar_abl1_all):
        # conservation is residue-level: identical across every variant
        scores = {
            v.get_score(ScoreDatabase.Conservation)
            for v in protvar_abl1_all.variants.values()
        }
        assert scores == {0.653}

    def test_get_variant_no_arg_ambiguous(self, protvar_abl1_all):
        # no single variant to return when many were parsed
        assert protvar_abl1_all.get_variant() is None
