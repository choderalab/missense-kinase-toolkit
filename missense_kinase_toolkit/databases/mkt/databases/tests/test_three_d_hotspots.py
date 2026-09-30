import pytest
from mkt.databases.three_d_hotspots import (
    COLUMN_RENAME,
    HotspotClass,
    ThreeDHotspots,
)


@pytest.fixture(scope="module")
def hotspots():
    """Load the bundled ``<repo_root>/data/3d_hotspots.xls`` workbook once."""
    return ThreeDHotspots()


class TestThreeDHotspots:
    def test_columns_renamed_and_ordered(self, hotspots):
        assert list(hotspots.df.columns) == [
            "hugoSymbol",
            "residue",
            "positionStart",
            "mutationCount",
            "pValue",
            "hotspotClass",
        ]
        # no raw upstream header survives the rename
        assert not set(COLUMN_RENAME).intersection(hotspots.df.columns)

    def test_license_column_dropped(self, hotspots):
        assert not hotspots.df.columns.str.startswith("Data available").any()

    def test_position_parsed_to_int(self, hotspots):
        assert str(hotspots.df["positionStart"].dtype) == "Int64"
        assert hotspots.df["positionStart"].notna().all()

    def test_class_values_are_enumerated(self, hotspots):
        assert set(hotspots.df["hotspotClass"].unique()) <= {
            c.value for c in HotspotClass
        }

    def test_get_gene_braf_v600(self, hotspots):
        braf = hotspots.get_gene("BRAF")
        assert not braf.empty
        v600 = braf[braf["positionStart"] == 600]
        assert (v600["hotspotClass"] == HotspotClass.HOTSPOT.value).all()

    def test_get_gene_missing_is_empty(self, hotspots):
        assert hotspots.get_gene("NOT_A_GENE").empty

    def test_csv_roundtrip(self, hotspots, tmp_path):
        path = tmp_path / "3d_hotspots.csv"
        hotspots.to_csv(str(path))
        reloaded = ThreeDHotspots.from_csv(str(path))
        assert reloaded.df.equals(hotspots.df)
        assert str(reloaded.df["positionStart"].dtype) == "Int64"

    def test_from_dataframe_skips_load(self, hotspots):
        obj = ThreeDHotspots.from_dataframe(hotspots.df)
        assert len(obj.get_gene("BRAF")) == len(hotspots.get_gene("BRAF"))
