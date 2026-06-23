import pytest
from mkt.databases.cancer_hotspots import (
    CancerHotspots,
    HotspotVersion,
    first_occurrence_map,
)


@pytest.fixture(scope="module")
def hotspots_v3():
    """Query cancerhotspots.org for the Bandlamudi 2026 (v3) tier once."""
    return CancerHotspots(version=HotspotVersion.BANDLAMUDI)


@pytest.fixture(scope="module")
def hotspots_v2():
    """Query cancerhotspots.org for the Chang (v2) tier once."""
    return CancerHotspots(version=HotspotVersion.CHANG)


@pytest.mark.network
class TestCancerHotspots:
    def test_default_version_is_bandlamudi(self):
        assert CancerHotspots.version is HotspotVersion.BANDLAMUDI

    def test_records_populated(self, hotspots_v3):
        assert hotspots_v3._json is not None
        assert len(hotspots_v3._df) == len(hotspots_v3._json)

    def test_position_flattened_to_int(self, hotspots_v3):
        assert {"positionStart", "positionEnd"}.issubset(hotspots_v3._df.columns)
        assert "aminoAcidPosition" not in hotspots_v3._df.columns
        assert str(hotspots_v3._df["positionStart"].dtype) == "Int64"

    def test_get_gene_braf_v600(self, hotspots_v3):
        braf = hotspots_v3.get_gene("BRAF")
        assert not braf.empty
        assert (braf["positionStart"] == 600).any()

    def test_v2_subset_of_v3(self, hotspots_v2, hotspots_v3):
        # the API notes invariant: v2 is a strict subset of v3
        s2 = set(zip(hotspots_v2._df["hugoSymbol"], hotspots_v2._df["residue"]))
        s3 = set(zip(hotspots_v3._df["hugoSymbol"], hotspots_v3._df["residue"]))
        assert s2 <= s3

    def test_first_occurrence_map_braf_is_chang(self):
        occurrence = first_occurrence_map()
        assert occurrence[("BRAF", 600)] == "Chang"
        assert set(occurrence.values()) <= {"Chang", "Bandlamudi 2026"}
