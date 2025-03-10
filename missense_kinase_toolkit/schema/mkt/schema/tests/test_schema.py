import os


class TestSchema:
    def test_missense_kinase_toolkit_database_imported(self):
        """Test if module is imported."""
        import sys

        import mkt.schema  # noqa F401

        assert "mkt.schema" in sys.modules

    def test_dict_kinase(self):
        from mkt.schema import io_utils

        dict_kinase = io_utils.deserialize_kinase_dict()
        assert len(dict_kinase) == 517

        # missing data
        n_klifs = len(
            [i.hgnc_name for i in dict_kinase.values() if i.klifs is not None]
        )
        assert n_klifs == 510

        n_pocket = len(
            [
                i.hgnc_name
                for i in dict_kinase.values()
                if i.klifs is not None and i.klifs.pocket_seq is not None
            ]
        )
        assert n_pocket == 487

        n_kincore = len(
            [i.hgnc_name for i in dict_kinase.values() if i.kincore is not None]
        )
        assert n_kincore == 474

        n_pfam = len([i.hgnc_name for i in dict_kinase.values() if i.pfam is not None])
        assert n_pfam == 468

        n_klif2uniprot = len(
            [
                i.hgnc_name
                for i in dict_kinase.values()
                if i.KLIFS2UniProtIdx is not None
            ]
        )
        assert n_klif2uniprot == 487

        # check ABL1 entries
        obj_abl1 = dict_kinase["ABL1"]

        assert obj_abl1.hgnc_name == "ABL1"

        assert obj_abl1.uniprot_id == "P00519"

        assert obj_abl1.kinhub.kinase_name == "Tyrosine-protein kinase ABL1"
        assert obj_abl1.kinhub.manning_name == ["ABL"]
        assert obj_abl1.kinhub.xname == ["ABL1"]
        assert obj_abl1.kinhub.group == ["TK"]
        assert obj_abl1.kinhub.family == ["Other"]

        assert (
            obj_abl1.uniprot.canonical_seq
            == "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR"
        )

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
            obj_abl1.kincore.seq
            == "ITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFET"
        )
        assert obj_abl1.kincore.start == 242
        assert obj_abl1.kincore.end == 495
        assert obj_abl1.kincore.mismatch is None

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

    def test_serialization(self, caplog):
        import shutil

        from mkt.schema import io_utils

        dict_kinase = io_utils.deserialize_kinase_dict()

        for suffix in io_utils.DICT_FUNCS.keys():
            print(f"Format: {suffix}")
            io_utils.serialize_kinase_dict(
                dict_kinase, suffix=suffix, str_path=f"./{suffix}"
            )
            if os.name == "nt" and suffix == "toml":
                pass
            else:
                dict_temp = io_utils.deserialize_kinase_dict(
                    suffix=suffix, str_path=f"./{suffix}"
                )
                assert dict_kinase == dict_temp
                print()
                shutil.rmtree(f"./{suffix}")

        # TODO: Fix this test
        # # move to data subdir in Github repo
        # path_original = io_utils.return_str_path()
        # path_gitroot = io_utils.get_repo_root()
        # path_new = os.path.join(path_gitroot, "data/KinaseInfo")
        # print(path_new)
        # if not os.path.exists(path_new):
        #     shutil.copytree(path_original, path_new)
        # shutil.rmtree(path_original)

        # # check that this produces a warning
        # with caplog.at_level(logging.WARNING):
        #     path_temp = io_utils.return_str_path()
        #     # assert caplog.records[0].levelname == "WARNING"
        #     # warn_msg = (
        #     #     "Could not find KinaseInfo directory within package: FileNotFoundError\n"
        #     #     "Please provide a path to the KinaseInfo directory."
        #     # )
        #     # assert warn_msg in caplog.records[0].message
        #     assert path_temp == path_new

        # # check that deserialization still works
        # dict_temp = io_utils.deserialize_kinase_dict()
        # assert dict_kinase == dict_temp

        # # delete all KinaseInfo files
        # shutil.rmtree(path_new)

        # # check that this oproduces an error
        # with caplog.at_level(logging.ERROR):
        #     path_temp = io_utils.return_str_path()
        #     assert caplog.records[0].levelname == "ERROR"
        #     warn_msg = (
        #         "Could not find KinaseInfo directory within package: FileNotFoundError\n"
        #         "Please provide a path to the KinaseInfo directory."
        #     )
        #     assert warn_msg in caplog.records[0].message
        #     assert path_temp is None
