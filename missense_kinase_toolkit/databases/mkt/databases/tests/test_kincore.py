from itertools import chain

import pytest
from mkt.databases.kincore import extract_pk_cif_files_as_list


@pytest.mark.network
class TestKinCoreHarmonization:
    def test_cif_hgnc_count_matches_cif_file_list(self, kincore_harmonized_dict):
        """Number of non-None CIF entries matches extract_pk_cif_files_as_list."""
        list_dict_cif_hgnc = [
            [entry.cif.hgnc for entry in v if entry.cif is not None]
            for v in kincore_harmonized_dict.values()
        ]
        list_dict_cif_hgnc = list(chain(*list_dict_cif_hgnc))
        list_kincore_cif = extract_pk_cif_files_as_list()
        assert len(list_dict_cif_hgnc) == len(list_kincore_cif)

    def test_egfr_has_single_entry(self, kincore_harmonized_dict):
        """EGFR (P00533) has exactly one KinCore entry."""
        assert len(kincore_harmonized_dict["P00533"]) == 1


@pytest.mark.network
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
