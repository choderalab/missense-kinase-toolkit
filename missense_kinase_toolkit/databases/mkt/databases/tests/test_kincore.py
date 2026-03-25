import tarfile
from itertools import chain

import pytest
from mkt.databases.kincore import PATH_ORIG_CIF


@pytest.mark.network
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


@pytest.mark.network
class TestKLIFSPocketAlignment:
    """KLIFS pocket alignment tests (depend on KinCore fixtures)."""

    def test_klifs_substr_actual(self, egfr_klifs_pocket):
        assert egfr_klifs_pocket.list_klifs_substr_actual == [
            "KVL",
            "GSGAFG",
            "TVYK",
            "VAIKEL",
            "EILDEAYVMAS",
            "VDPHVCR",
            "LLGI",
            "QLI",
            "T",
            "QLM",
            "PFGC",
            "LLDYVRE",
            "YLEDR",
            "RLV",
            "HRDLAARN",
            "VLV",
            "I",
            "TDFG",
            "LA",
        ]

    def test_klifs_substr_match(self, egfr_klifs_pocket):
        assert egfr_klifs_pocket.list_klifs_substr_match == [
            "KVL",
            "GSGAFG",
            "TVYK",
            "VAIKEL",
            "EILDEAYVMAS",
            "VDPHVCR",
            "LLGI",
            "QLI",
            "TQLM",
            "QLM",
            "PFGC",
            "LLDYVRE",
            "YLEDR",
            "RLV",
            "HRDLAARN",
            "VLV",
            "ITDFG",
            "TDFG",
            "TDFGLA",
        ]

    def test_klifs2uniprot_idx(self, egfr_klifs_pocket):
        assert egfr_klifs_pocket.KLIFS2UniProtIdx == {
            "I:1": 716,
            "I:2": 717,
            "I:3": 718,
            "g.l:4": 719,
            "g.l:5": 720,
            "g.l:6": 721,
            "g.l:7": 722,
            "g.l:8": 723,
            "g.l:9": 724,
            "II:10": 725,
            "II:11": 726,
            "II:12": 727,
            "II:13": 728,
            "III:14": 742,
            "III:15": 743,
            "III:16": 744,
            "III:17": 745,
            "III:18": 746,
            "III:19": 747,
            "αC:20": 758,
            "αC:21": 759,
            "αC:22": 760,
            "αC:23": 761,
            "αC:24": 762,
            "αC:25": 763,
            "αC:26": 764,
            "αC:27": 765,
            "αC:28": 766,
            "αC:29": 767,
            "αC:30": 768,
            "b.l:31": 769,
            "b.l:32": 770,
            "b.l:33": 772,
            "b.l:34": 773,
            "b.l:35": 774,
            "b.l:36": 775,
            "b.l:37": 776,
            "IV:38": 777,
            "IV:39": 778,
            "IV:40": 779,
            "IV:41": 780,
            "V:42": 787,
            "V:43": 788,
            "V:44": 789,
            "GK:45": 790,
            "hinge:46": 791,
            "hinge:47": 792,
            "hinge:48": 793,
            "linker:49": 794,
            "linker:50": 795,
            "linker:51": 796,
            "linker:52": 797,
            "αD:53": 798,
            "αD:54": 799,
            "αD:55": 800,
            "αD:56": 801,
            "αD:57": 802,
            "αD:58": 803,
            "αD:59": 804,
            "αE:60": 827,
            "αE:61": 828,
            "αE:62": 829,
            "αE:63": 830,
            "αE:64": 831,
            "VI:65": 832,
            "VI:66": 833,
            "VI:67": 834,
            "c.l:68": 835,
            "c.l:69": 836,
            "c.l:70": 837,
            "c.l:71": 838,
            "c.l:72": 839,
            "c.l:73": 840,
            "c.l:74": 841,
            "c.l:75": 842,
            "VII:76": 843,
            "VII:77": 844,
            "VII:78": 845,
            "VIII:79": 853,
            "xDFG:80": 854,
            "xDFG:81": 855,
            "xDFG:82": 856,
            "xDFG:83": 857,
            "a.l:84": 858,
            "a.l:85": 859,
        }

    def test_klifs2uniprot_seq(self, egfr_klifs_pocket):
        assert egfr_klifs_pocket.KLIFS2UniProtSeq == {
            "I": "KVL",
            "g.l": "GSGAFG",
            "II": "TVYK",
            "II:III": "GLWIPEGEKVKIP",
            "III": "VAIKEL",
            "III:αC": "REATSPKANK",
            "αC": "EILDEAYVMAS",
            "b.l_1": "VD",
            "b.l_intra": "N",
            "b.l_2": "PHVCR",
            "IV": "LLGI",
            "IV:V": "CLTSTV",
            "V": "QLI",
            "GK": "T",
            "hinge": "QLM",
            "hinge:linker": None,
            "linker_1": "P",
            "linker_intra": None,
            "linker_2": "FGC",
            "αD": "LLDYVRE",
            "αD:αE": "HKDNIGSQYLLNWCVQIAKGMN",
            "αE": "YLEDR",
            "αE:VI": None,
            "VI": "RLV",
            "c.l": "HRDLAARN",
            "VII": "VLV",
            "VII:VIII": "KTPQHVK",
            "VIII": "I",
            "xDFG": "TDFG",
            "a.l": "LA",
        }
