import pytest
from mkt.databases.cancer_hotspots import (
    TIER_COLUMN,
    CancerHotspots,
    CancerHotspotsQuery,
    HotspotTier,
    HotspotVersion,
)


@pytest.fixture(scope="module")
def query_v3():
    """Query cancerhotspots.org for the Bandlamudi 2026 (v3) tier once."""
    return CancerHotspotsQuery(version=HotspotVersion.BANDLAMUDI)


@pytest.fixture(scope="module")
def hotspots():
    """Build the harmonized, tier-annotated table once."""
    return CancerHotspots()


@pytest.mark.network
class TestCancerHotspotsQuery:
    def test_default_version_is_bandlamudi(self):
        assert CancerHotspotsQuery.version is HotspotVersion.BANDLAMUDI

    def test_records_populated(self, query_v3):
        assert query_v3._json is not None
        assert len(query_v3._df) == len(query_v3._json)

    def test_position_flattened_to_int(self, query_v3):
        assert {"positionStart", "positionEnd"}.issubset(query_v3._df.columns)
        assert "aminoAcidPosition" not in query_v3._df.columns
        assert str(query_v3._df["positionStart"].dtype) == "Int64"

    def test_get_gene_braf_v600(self, query_v3):
        braf = query_v3.get_gene("BRAF")
        assert not braf.empty
        assert (braf["positionStart"] == 600).any()


@pytest.mark.network
class TestCancerHotspots:
    def test_tier_column_present(self, hotspots):
        assert TIER_COLUMN in hotspots.df.columns
        assert set(hotspots.df[TIER_COLUMN].unique()) <= {
            HotspotTier.CHANG.value,
            HotspotTier.BANDLAMUDI.value,
        }

    def test_harmonized_is_v3_superset(self, hotspots):
        # one annotated row per v3 record
        assert len(hotspots.df) == len(hotspots.query_bandlamudi._df)

    def test_chang_subset_of_v3(self, hotspots):
        n_chang_rows = (hotspots.df[TIER_COLUMN] == HotspotTier.CHANG.value).sum()
        assert n_chang_rows == len(hotspots.query_chang._df)

    def test_braf_v600_is_chang(self, hotspots):
        braf = hotspots.get_gene("BRAF")
        v600 = braf[braf["positionStart"] == 600]
        assert (v600[TIER_COLUMN] == HotspotTier.CHANG.value).all()

    def test_first_occurrence_map_position_keyed(self, hotspots):
        occurrence = hotspots.first_occurrence_map()
        assert occurrence[("BRAF", 600)] == HotspotTier.CHANG.value
        assert set(occurrence.values()) <= {
            HotspotTier.CHANG.value,
            HotspotTier.BANDLAMUDI.value,
        }

    def test_position_collapse_fewer_than_residue(self, hotspots):
        # residue-level new count (df) should exceed position-level new count (map)
        n_new_rows = (hotspots.df[TIER_COLUMN] == HotspotTier.BANDLAMUDI.value).sum()
        occurrence = hotspots.first_occurrence_map()
        n_new_positions = sum(
            1 for v in occurrence.values() if v == HotspotTier.BANDLAMUDI.value
        )
        assert n_new_positions <= n_new_rows

    def test_single_residue_only_excludes_indels(self, hotspots):
        full = hotspots.first_occurrence_map()
        single = hotspots.first_occurrence_map(single_residue_only=True)
        # every single-residue position is a position present in the full map
        assert set(single) <= set(full)
        # dropping indel positions yields a strictly smaller keyset
        assert len(single) < len(full)
        # BRAF V600 (single residue) survives the filter and stays Chang
        assert single[("BRAF", 600)] == HotspotTier.CHANG.value
