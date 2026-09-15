import pytest
from mkt.databases import colors


class TestMapAAToSingleLetterCode:
    def test_three_letter_code(self):
        assert colors.map_aa_to_single_letter_code("Ala") == "A"

    def test_three_letter_code_uppercase(self):
        assert colors.map_aa_to_single_letter_code("ALA") == "A"

    def test_full_name(self):
        assert colors.map_aa_to_single_letter_code("alanine") == "A"

    def test_full_name_uppercase(self):
        assert colors.map_aa_to_single_letter_code("ALANINE") == "A"

    def test_single_letter(self):
        assert colors.map_aa_to_single_letter_code("A") == "A"

    def test_single_letter_lowercase(self):
        assert colors.map_aa_to_single_letter_code("a") == "A"

    def test_invalid_single_letter(self, capsys):
        assert colors.map_aa_to_single_letter_code("X") is None
        out, _ = capsys.readouterr()
        assert out == "Invalid single-letter amino acid: X\n"

    def test_invalid_two_letter(self, capsys):
        assert colors.map_aa_to_single_letter_code("AL") is None
        out, _ = capsys.readouterr()
        assert out == "Length error and invalid amino acid: AL\n"

    def test_invalid_three_letter(self, capsys):
        assert colors.map_aa_to_single_letter_code("XYZ") is None
        out, _ = capsys.readouterr()
        assert out == "Invalid 3-letter amino acid: XYZ\n"

    def test_invalid_full_name(self, capsys):
        assert colors.map_aa_to_single_letter_code("TEST") is None
        out, _ = capsys.readouterr()
        assert out == "Invalid amino acid name: test\n"


class TestMapSingleLetterAAToColor:
    @pytest.fixture()
    def dict_colors(self):
        return colors.DICT_COLORS

    def test_alphabet_project(self, dict_colors):
        assert (
            colors.map_single_letter_aa_to_color(
                "A", dict_colors["ALPHABET_PROJECT"]["DICT_COLORS"]
            )
            == "#F0A3FF"
        )

    def test_asap(self, dict_colors):
        assert (
            colors.map_single_letter_aa_to_color(
                "A", dict_colors["ASAP"]["DICT_COLORS"]
            )
            == "red"
        )

    def test_rasmol(self, dict_colors):
        assert (
            colors.map_single_letter_aa_to_color(
                "A", dict_colors["RASMOL"]["DICT_COLORS"]
            )
            == "#C8C8C8"
        )

    def test_shapely(self, dict_colors):
        assert (
            colors.map_single_letter_aa_to_color(
                "A", dict_colors["SHAPELY"]["DICT_COLORS"]
            )
            == "#8CFF8C"
        )

    def test_clustalx(self, dict_colors):
        assert (
            colors.map_single_letter_aa_to_color(
                "A", dict_colors["CLUSTALX"]["DICT_COLORS"]
            )
            == "blue"
        )

    def test_zappo(self, dict_colors):
        assert (
            colors.map_single_letter_aa_to_color(
                "A", dict_colors["ZAPPO"]["DICT_COLORS"]
            )
            == "#ffafaf"
        )
