"""Tests for OncoTree roll-up navigation (offline, fixture-backed)."""

import pytest
from mkt.databases.oncotree import (
    DICT_ONCOTREE_LEGACY_ALIAS,
    OncoTree,
    fetch_oncotree_rename_map,
    return_nearest_shared_ancestor,
    return_rename_map_from_nodes,
)

# minimal ragged OncoTree TSV: level_1..level_3 + the five metadata columns.
# two lineages, Lung (LUNG -> NSCLC -> LUAD) and Breast (BREAST -> BRCA -> IDC),
# with the shallow rows omitting trailing level cells (ragged, like upstream).
_FIXTURE_TSV = "\n".join(
    [
        "level_1\tlevel_2\tlevel_3\tmetamaintype\tmetacolor\tmetanci\tmetaumls\thistory",
        "Lung (LUNG)\tmeta\tRed\t\t\t",
        "Lung (LUNG)\tNon-Small Cell Lung Cancer (NSCLC)\tmeta\tRed\t\t\t",
        "Lung (LUNG)\tNon-Small Cell Lung Cancer (NSCLC)\tLung Adenocarcinoma (LUAD)\tmeta\tRed\t\t\t",
        "Breast (BREAST)\tmeta\tHotPink\t\t\t",
        "Breast (BREAST)\tInvasive Breast Carcinoma (BRCA)\tmeta\tHotPink\t\t\t",
        "Breast (BREAST)\tInvasive Breast Carcinoma (BRCA)\tBreast Invasive Ductal Carcinoma (IDC)\tmeta\tHotPink\t\t\t",
    ]
)


@pytest.fixture
def oncotree(tmp_path):
    """An OncoTree built from the offline fixture TSV."""
    path = tmp_path / "tumor_types.txt"
    path.write_text(_FIXTURE_TSV)
    return OncoTree(filepath=str(path))


class TestOncoTreeRollUp:
    def test_ancestor_code_at_level_walks_lineage(self, oncotree):
        """A deep code rolls up to the ancestor code at each level."""
        assert oncotree.ancestor_code_at_level("LUAD", 1) == "LUNG"
        assert oncotree.ancestor_code_at_level("LUAD", 2) == "NSCLC"
        assert oncotree.ancestor_code_at_level("LUAD", 3) == "LUAD"

    def test_shallow_node_returns_itself(self, oncotree):
        """A node shallower than the target level returns its own code."""
        assert oncotree.ancestor_code_at_level("LUNG", 2) == "LUNG"
        assert oncotree.ancestor_code_at_level("NSCLC", 3) == "NSCLC"

    def test_unknown_code_returns_none(self, oncotree):
        assert oncotree.ancestor_code_at_level("NOPE", 2) is None

    def test_level_below_one_raises(self, oncotree):
        with pytest.raises(ValueError):
            oncotree.ancestor_code_at_level("LUAD", 0)

    def test_roll_up_map(self, oncotree):
        assert oncotree.roll_up_map(2, ["LUAD", "IDC"]) == {
            "LUAD": "NSCLC",
            "IDC": "BRCA",
        }

    def test_roll_up_map_all_codes(self, oncotree):
        """Without an explicit list, every known code is rolled up."""
        rolled = oncotree.roll_up_map(1)
        assert rolled["LUAD"] == "LUNG"
        assert rolled["IDC"] == "BREAST"
        assert set(rolled) == {"LUNG", "NSCLC", "LUAD", "BREAST", "BRCA", "IDC"}

    def test_dict_code_name_resolves_internal_nodes(self, oncotree):
        """Internal (rolled-up) codes resolve to a display name, not just leaves."""
        names = oncotree.dict_code_name
        assert names["NSCLC"] == "Non-Small Cell Lung Cancer"
        assert names["LUAD"] == "Lung Adenocarcinoma"

    def test_known_codes(self, oncotree):
        assert oncotree.known_codes == {
            "LUNG",
            "NSCLC",
            "LUAD",
            "BREAST",
            "BRCA",
            "IDC",
        }


class TestOncoTreeResolveCode:
    def test_legacy_alias_is_applied(self, oncotree):
        """Retired / truncated codes map to their current equivalent."""
        assert oncotree.resolve_code("GBM") == "GB"
        assert oncotree.resolve_code("DIPG") == "DMG"
        assert oncotree.resolve_code("BLLETV6RUN") == "BLLETV6RUNX1"

    def test_split_code_resolves_to_shared_parent(self, oncotree):
        """ALL was split into BLL and TLL, so it resolves to their shared parent."""
        assert oncotree.resolve_code("ALL") == "LNM"

    def test_current_code_unchanged(self, oncotree):
        assert oncotree.resolve_code("LUAD") == "LUAD"
        assert oncotree.resolve_code("NOPE") == "NOPE"


_LIST_NODES = [
    {"code": "TISSUE", "parent": None},
    {"code": "LNM", "parent": "TISSUE"},
    {"code": "BLL", "parent": "LNM", "revocations": ["ALL"]},
    {"code": "TLL", "parent": "LNM", "revocations": ["ALL"], "precursors": ["TALL"]},
    {"code": "GB", "parent": "TISSUE", "history": ["GBM"]},
]
"""list[dict]: Minimal OncoTree API nodes: ALL split into BLL/TLL under LNM."""

_DICT_PARENT = {node["code"]: node["parent"] for node in _LIST_NODES} | {"X": None}
"""dict[str, str | None]: Parent links for :data:`_LIST_NODES` plus a disjoint root."""


class TestNearestSharedAncestor:
    def test_siblings_share_parent(self):
        assert return_nearest_shared_ancestor(["BLL", "TLL"], _DICT_PARENT) == "LNM"

    def test_ancestor_of_the_others_is_returned(self):
        assert return_nearest_shared_ancestor(["LNM", "BLL"], _DICT_PARENT) == "LNM"

    def test_single_code_is_its_own_ancestor(self):
        assert return_nearest_shared_ancestor(["BLL"], _DICT_PARENT) == "BLL"

    def test_disjoint_codes_return_none(self):
        assert return_nearest_shared_ancestor(["BLL", "X"], _DICT_PARENT) is None


class TestRenameMapFromNodes:
    def test_single_successor_renames(self):
        rename = return_rename_map_from_nodes(_LIST_NODES)
        assert rename["GBM"] == "GB"
        assert rename["TALL"] == "TLL"

    def test_split_code_maps_to_nearest_shared_ancestor(self):
        assert return_rename_map_from_nodes(_LIST_NODES)["ALL"] == "LNM"

    def test_independent_of_node_order(self):
        """The API returns nodes in varying order; the map must not depend on it."""
        assert return_rename_map_from_nodes(
            list(reversed(_LIST_NODES))
        ) == return_rename_map_from_nodes(_LIST_NODES)


class TestFetchRenameMap:
    @pytest.mark.network
    def test_fetch_covers_curated_renames(self):
        """The live API rename map agrees with the curated non-truncation aliases."""
        api = fetch_oncotree_rename_map()
        assert api.get("GBM") == "GB"
        assert api.get("ALL") == "LNM"
        # every curated alias the API also knows about should match (truncation
        # fixes like AMLMLLT3KM are client-side and absent from the API)
        for old, current in DICT_ONCOTREE_LEGACY_ALIAS.items():
            if old in api:
                assert api[old] == current
