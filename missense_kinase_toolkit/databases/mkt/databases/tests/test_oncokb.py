import pytest
from mkt.databases import oncokb
from mkt.databases.config import maybe_get_oncokb_token

# the OncoKB API requires an access token, so these tests are skipped unless
# ONCOKB_TOKEN is set in the environment. to run them in CI, add the token as a
# GitHub Actions secret (e.g. ONCOKB_TOKEN) and export it before pytest runs.
pytestmark = pytest.mark.skipif(
    maybe_get_oncokb_token() is None,
    reason="OncoKB API token not set (ONCOKB_TOKEN); skipping live OncoKB tests",
)


@pytest.fixture(scope="module")
def oncokb_levels():
    """Query the OncoKB API for the levels of evidence once."""
    return oncokb.get_oncokb_levels()


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
