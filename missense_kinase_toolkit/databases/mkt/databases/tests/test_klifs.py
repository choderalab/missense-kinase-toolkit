import pytest


@pytest.mark.network
class TestKLIFSKinaseInfo:
    def test_single_kinase_result(self, egfr_klifs_info):
        assert len(egfr_klifs_info._kinase_info) == 1

    def test_family(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["family"] == "EGFR"

    def test_full_name(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert (
            egfr_klifs_info.get_kinase_info()[0]["full_name"]
            == "epidermal growth factor receptor"
        )

    def test_gene_name(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["gene_name"] == "EGFR"

    def test_group(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["group"] == "TK"

    def test_iuphar(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["iuphar"] == 1797

    def test_kinase_id(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["kinase_ID"] == 406

    def test_name(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["name"] == "EGFR"

    def test_pocket_sequence(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert (
            egfr_klifs_info.get_kinase_info()[0]["pocket"]
            == "KVLGSGAFGTVYKVAIKELEILDEAYVMASVDPHVCRLLGIQLITQLMPFGCLLDYVREYLEDRRLVHRDLAARNVLVITDFGLA"
        )

    def test_species(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["species"] == "Human"

    def test_uniprot_id(self, egfr_klifs_info):
        if egfr_klifs_info.status_code != 200:
            pytest.skip("KLIFS API returned non-200 status")
        assert egfr_klifs_info.get_kinase_info()[0]["uniprot"] == "P00533"

    def test_server_error_returns_none(self, egfr_klifs_info):
        """When KLIFS returns 5xx, get_kinase_info()[0] should be None."""
        if 500 <= egfr_klifs_info.status_code < 600:
            assert egfr_klifs_info.get_kinase_info()[0] is None


@pytest.mark.network
@pytest.mark.xdist_group("kincore")
class TestKLIFSPocketAlignment:
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
