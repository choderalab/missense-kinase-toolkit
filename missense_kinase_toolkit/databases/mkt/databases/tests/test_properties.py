import pytest
from mkt.databases.properties import (
    DICT_AA_PROPERTIES,
    classify_aa_change,
    get_aa_property,
)

STR_AA20 = "ACDEFGHIKLMNPQRSTVWY"


class TestAAProperties:
    def test_covers_the_twenty_standard_amino_acids(self):
        assert set(DICT_AA_PROPERTIES) == set(STR_AA20)

    @pytest.mark.parametrize("str_property", ["charge", "volume", "hydrophobicity"])
    def test_every_residue_has_every_property(self, str_property):
        assert all(str_property in d for d in DICT_AA_PROPERTIES.values())

    def test_hydrophobicity_is_the_eisenberg_consensus_scale(self):
        # spot checks against Eisenberg et al. (1984): the extremes and a midpoint
        assert DICT_AA_PROPERTIES["I"]["hydrophobicity"] == pytest.approx(1.38)
        assert DICT_AA_PROPERTIES["R"]["hydrophobicity"] == pytest.approx(-2.53)
        assert DICT_AA_PROPERTIES["G"]["hydrophobicity"] == pytest.approx(0.48)

    def test_hydrophobicity_ranks_hydrophobic_above_charged(self):
        hydrophobic = [DICT_AA_PROPERTIES[aa]["hydrophobicity"] for aa in "ILVF"]
        charged = [DICT_AA_PROPERTIES[aa]["hydrophobicity"] for aa in "DEKR"]
        assert min(hydrophobic) > max(charged)

    def test_hydrophobicity_is_normalized(self):
        # a normalized consensus scale is centred near zero with unit-order spread
        values = [d["hydrophobicity"] for d in DICT_AA_PROPERTIES.values()]
        assert abs(sum(values) / len(values)) < 0.1
        assert max(values) - min(values) == pytest.approx(3.91)


class TestGetAAProperty:
    def test_returns_the_value_case_insensitively(self):
        assert get_aa_property("l", "hydrophobicity") == pytest.approx(1.06)
        assert get_aa_property("K", "charge") == "positive"

    def test_unknown_residue_or_property_is_none(self):
        assert get_aa_property("X", "volume") is None
        assert get_aa_property("A", "not_a_property") is None


class TestClassifyAAChange:
    def test_charge_reversal_and_large_volume_loss(self):
        # R (positive, 173.4) -> D (negative, 111.1): -62.3 A^3 is a modest loss
        assert classify_aa_change("R", "D") == {
            "charge": "charge change, positive to negative",
            "polarity": None,
            "volume": "volume loss, modest",
        }

    def test_polarity_gain_and_large_volume_gain(self):
        # G (nonpolar, 60.1) -> R (positive, 173.4): +113.3 A^3
        assert classify_aa_change("G", "R") == {
            "charge": "charge gain, neutral to positive",
            "polarity": "polarity gain",
            "volume": "volume gain, large",
        }

    def test_invalid_residue_is_none(self):
        assert classify_aa_change("A", "X") is None
