import logging
import os


class TestSchema:
    def test_missense_kinase_toolkit_database_imported(self):
        """Test if module is imported."""
        import sys

        import mkt.schema  # noqa F401

        assert "mkt.schema" in sys.modules

    def test_dict_kinase(self, caplog):
        """Test if the kinase dictionary is correctly deserialized."""
        import copy

        from mkt.schema.io_utils import deserialize_kinase_dict

        DICT_KINASE = deserialize_kinase_dict()
        DICT_KINASE_COPY = copy.deepcopy(DICT_KINASE)

        # check that deserializing again gives the same object
        # (i.e. from cache, not re-reading the file)
        DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE")
        assert DICT_KINASE == DICT_KINASE_COPY

        caplog.set_level(logging.INFO)

        assert len(DICT_KINASE) == 566
        assert (
            sum(["_" in i for i in DICT_KINASE.keys()]) == 28
        )  # 14 proteins with multiple KDs

        # missing data
        n_klifs = len(
            [i.hgnc_name for i in DICT_KINASE.values() if i.klifs is not None]
        )
        assert n_klifs == 555

        n_pocket = len(
            [
                i.hgnc_name
                for i in DICT_KINASE.values()
                if i.klifs is not None and i.klifs.pocket_seq is not None
            ]
        )
        assert n_pocket == 519

        n_kincore = len(
            [i.hgnc_name for i in DICT_KINASE.values() if i.kincore is not None]
        )
        assert n_kincore == 492

        n_pfam = len([i.hgnc_name for i in DICT_KINASE.values() if i.pfam is not None])
        assert n_pfam == 490

        n_klif2uniprot = len(
            [
                i.hgnc_name
                for i in DICT_KINASE.values()
                if i.KLIFS2UniProtIdx is not None
            ]
        )
        assert n_klif2uniprot == 519

        # check ABL1 entries
        obj_abl1 = DICT_KINASE["ABL1"]

        assert obj_abl1.hgnc_name == "ABL1"

        assert obj_abl1.uniprot_id == "P00519"

        assert obj_abl1.kinhub.hgnc_name == "ABL1"
        assert obj_abl1.kinhub.kinase_name == "Tyrosine-protein kinase ABL1"
        assert obj_abl1.kinhub.manning_name == "ABL"
        assert obj_abl1.kinhub.xname == "ABL1"
        assert obj_abl1.kinhub.group == "TK"
        assert obj_abl1.kinhub.family == "Other"

        assert (
            obj_abl1.uniprot.canonical_seq
            == "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR"
        )
        assert obj_abl1.uniprot.phospho_sites == [
            50,
            70,
            115,
            128,
            139,
            172,
            185,
            215,
            226,
            229,
            253,
            257,
            393,
            413,
            446,
            559,
            569,
            618,
            619,
            620,
            659,
            683,
            718,
            735,
            751,
            781,
            814,
            823,
            844,
            852,
            855,
            917,
            977,
        ]
        assert (
            sum([i.startswith("Phospho") for i in obj_abl1.uniprot.phospho_description])
            == 33
        )
        assert len(obj_abl1.uniprot.phospho_evidence) == 33

        assert obj_abl1.klifs.gene_name == "ABL1"
        assert obj_abl1.klifs.name == "ABL1"
        assert (
            obj_abl1.klifs.full_name
            == "ABL proto-oncogene 1, non-receptor tyrosine kinase"
        )
        assert obj_abl1.klifs.group == "TK"
        assert obj_abl1.klifs.family == "Other"
        assert obj_abl1.klifs.iuphar == 1923
        assert obj_abl1.klifs.kinase_id == 392
        assert (
            obj_abl1.klifs.pocket_seq
            == "HKLGGGQYGEVYEVAVKTLEFLKEAAVMKEIKPNLVQLLGVYIITEFMTYGNLLDYLREYLEKKNFIHRDLAARNCLVVADFGLS"
        )

        assert (
            obj_abl1.pfam.domain_name == "Protein tyrosine and serine/threonine kinase"
        )
        assert obj_abl1.pfam.start == 242
        assert obj_abl1.pfam.end == 492
        assert obj_abl1.pfam.pfam_accession == "PF07714"
        assert obj_abl1.pfam.in_alphafold is True

        assert (
            obj_abl1.kincore.fasta.seq
            == "KWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSIS"
        )
        assert obj_abl1.kincore.fasta.start == 234
        assert obj_abl1.kincore.fasta.end == 503
        assert obj_abl1.kincore.mismatch is None
        assert obj_abl1.kincore.start == 1
        assert obj_abl1.kincore.end == 270

        str_dict = "".join(
            [
                v
                for k, v in obj_abl1.KLIFS2UniProtSeq.items()
                if v is not None and ":" not in k and "_intra" not in k
            ]
        )
        assert obj_abl1.klifs.pocket_seq == str_dict

        assert min(obj_abl1.KLIFS2UniProtIdx.values()) == 246
        assert max(obj_abl1.KLIFS2UniProtIdx.values()) == 385

        assert (
            DICT_KINASE["ABL1"].extract_sequence_from_cif()
            == "KWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSIS"
        )

        # test logger messages for CIF extraction failures
        caplog.clear()
        assert (
            DICT_KINASE["BUB1B"].extract_sequence_from_cif() is None
        )  # Kincore but no cif
        assert "No CIF sequence for BUB1" in caplog.text

        caplog.clear()
        assert DICT_KINASE["ABR"].extract_sequence_from_cif() is None  # no Kincore
        assert "No CIF sequence for ABR" in caplog.text

        assert (
            DICT_KINASE["ABL1"].adjudicate_kd_sequence()
            == "KWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSIS"
        )
        assert (
            DICT_KINASE["BUB1B"].adjudicate_kd_sequence()
            == "YCIKREYLICEDYKLFWVAPRNSAELTVIKVSSQPVPWDFYINLKLKERLNEDFDHFCSCYQYQDGCIVWHQYINCFTLQDLLQHSEYITHEITVLIIYNLLTIVEMLHKAEIVHGDLSPRCLILRNRIHDPYDCNKNNQALKIVDFSYSVDLRVQLDVFTLSGFRTVQILEGQKILANCSSPYQVDLFGIADLAHLLLFKEHLQVFWDGSFWKLSQNISELKDGELWNKFFVRILNANDEATVSVLGELAAEMNG"
        )
        assert (
            DICT_KINASE["MTOR"].adjudicate_kd_sequence()
            == "VVEPYRKYPTLLEVLLNFLKTEQNQGTRREAIRVLGLLGALDPYKHKVNIGMIDQSRDASAVSLSESKSSQDSSDYSTSEMLVNMGNLPLDEFYPAVSMVALMRIFRDQSLSHHHTMVVQAITFIFKSLGLKCVQFLPQVMPTFLNVIRVCDGAIREFLFQQLGMLVSFVK"
        )
        caplog.clear()
        assert DICT_KINASE["ABR"].adjudicate_kd_sequence() is None
        assert "No kinase domain sequence found for ABR" in caplog.text

        # do this last since changing the DICT_KINASE object
        assert DICT_KINASE["ABL1"].adjudicate_group() == "TK"  # Kincore
        assert DICT_KINASE["ABR"].adjudicate_group() == "Atypical"  # KinHub
        assert DICT_KINASE["ANTXR1"].adjudicate_group() == "Atypical"  # KLIFS
        DICT_KINASE["ANTXR1"].klifs = None
        caplog.clear()
        assert DICT_KINASE["ANTXR1"].adjudicate_group() is None  # None
        assert "No group found for ANTXR1" in caplog.text

    # TODO: downsample toml files to speed up test
    # TODO: add .tar.gz test for yaml/toml - currently only presumed to work
    def test_serde(self):
        """Test if the kinase dictionary is correctly serialized and deserialized."""
        from mkt.schema import io_utils

        yaml_test = "CDK2"

        DICT_KINASE = io_utils.deserialize_kinase_dict(str_name="DICT_KINASE")

        for suffix in io_utils.DICT_FUNCS.keys():
            print(f"Format: {suffix}")
            if suffix == "yaml":
                # only check a single entry given time
                # ~4 hours to check all (done on 4/4/25)
                io_utils.serialize_kinase_dict(
                    {yaml_test: DICT_KINASE[yaml_test]},
                    suffix=suffix,
                    str_path=f"./{suffix}",
                )
            else:
                io_utils.serialize_kinase_dict(
                    DICT_KINASE,
                    suffix=suffix,
                    str_path=f"./{suffix}",
                )
            if os.name == "nt" and suffix == "toml":
                pass
            else:
                dict_temp = io_utils.deserialize_kinase_dict(
                    suffix=suffix, str_path=f"./{suffix}"
                )
                if suffix == "yaml":
                    assert DICT_KINASE[yaml_test] == dict_temp[yaml_test]
                else:
                    assert DICT_KINASE == dict_temp

    def test_utils(self):
        """Test utility functions."""
        import random

        from mkt.schema import utils
        from mkt.schema.io_utils import deserialize_kinase_dict

        DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE")

        # test rgetattr
        obj = DICT_KINASE["ABL1"]
        assert utils.rgetattr(obj, attr="hgnc_name") == "ABL1"
        assert utils.rgetattr(obj, attr="uniprot_id") == "P00519"
        assert utils.rgetattr(obj, attr="non_existent") is None

        # test rsetattr
        utils.rsetattr(obj=obj, attr="hgnc_name", val="ABL2")
        assert obj.hgnc_name == "ABL2"
        utils.rsetattr(obj=obj, attr="kincore.fasta.seq", val=None)
        assert obj.kincore.fasta.seq is None

        random.seed(42)
        uuid = utils.random_uuid()
        assert str(uuid) == "a31c06bd-463e-4923-bc1a-adbde48b1697"
