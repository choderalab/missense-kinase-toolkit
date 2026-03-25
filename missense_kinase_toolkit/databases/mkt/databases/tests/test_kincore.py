import tarfile
from itertools import chain

import pytest
from mkt.databases.kincore import PATH_ORIG_CIF


@pytest.mark.network
@pytest.mark.xdist_group("kincore")
class TestKinCoreHarmonization:
    def test_cif_hgnc_count_matches_cif_file_count(self, kincore_harmonized_dict):
        """Number of non-None CIF entries matches .cif count in tar.gz archive."""
        list_dict_cif_hgnc = [
            [entry.cif.hgnc for entry in v if entry.cif is not None]
            for v in kincore_harmonized_dict.values()
        ]
        list_dict_cif_hgnc = list(chain(*list_dict_cif_hgnc))
        # count .cif members from the archive index (no extraction);
        # exclude macOS resource fork entries (._*.cif)
        with tarfile.open(PATH_ORIG_CIF, "r:gz") as tar:
            n_cif_files = sum(
                1
                for m in tar.getmembers()
                if m.name.endswith(".cif") and "/._" not in m.name
            )
        assert len(list_dict_cif_hgnc) == n_cif_files

    def test_egfr_has_single_entry(self, kincore_harmonized_dict):
        """EGFR (P00533) has exactly one KinCore entry."""
        assert len(kincore_harmonized_dict["P00533"]) == 1


@pytest.mark.network
@pytest.mark.xdist_group("kincore")
class TestKinCoreAlignment:
    def test_aligned_sequence(self, egfr_kincore_alignment):
        assert (
            egfr_kincore_alignment["seq"]
            == "LRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRY"
        )

    def test_alignment_start(self, egfr_kincore_alignment):
        assert egfr_kincore_alignment["start"] == 704

    def test_alignment_end(self, egfr_kincore_alignment):
        assert egfr_kincore_alignment["end"] == 978

    def test_no_mismatches(self, egfr_kincore_alignment):
        assert egfr_kincore_alignment["mismatch"] is None
