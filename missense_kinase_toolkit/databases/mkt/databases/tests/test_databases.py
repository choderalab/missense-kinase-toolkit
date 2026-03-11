import pytest


class TestDatabases:
    def test_missense_kinase_toolkit_database_imported(self):
        """Test if module is imported."""
        import sys

        import mkt.databases  # noqa F401

        assert "mkt.databases" in sys.modules

    def test_config(self):
        from mkt.databases import config

        # test that the function to set the output directory works
        with pytest.raises(SystemExit) as sample:
            config.get_output_dir()
        assert sample.type == SystemExit
        assert sample.value.code == 1
        config.set_output_dir("test")
        assert config.get_output_dir() == "test"

        # test that the function to set the request cache works
        assert config.maybe_get_request_cache() is None
        config.set_request_cache("test")
        assert config.maybe_get_request_cache() == "test"

        # test that the function to set the cBioPortal instance works
        with pytest.raises(SystemExit) as sample:
            config.get_cbioportal_instance()
        assert sample.type == SystemExit
        assert sample.value.code == 1
        config.set_cbioportal_instance("test")
        assert config.get_cbioportal_instance() == "test"

        # test that the function to set the cBioPortal token works
        assert config.maybe_get_cbioportal_token() is None
        config.set_cbioportal_token("test")
        assert config.maybe_get_cbioportal_token() == "test"

    def test_io_utils(self):
        import os

        import pandas as pd
        from mkt.databases import config, io_utils

        # os.environ["OUTPUT_DIR"] = "."
        config.set_output_dir(".")

        # test that the functions to save and load dataframes work
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        io_utils.save_dataframe_to_csv(df, "test1.csv")
        df_read = io_utils.load_csv_to_dataframe("test1.csv")
        assert df.equals(df_read)

        # test that the function to concatenate csv files with glob works
        io_utils.save_dataframe_to_csv(df, "test2.csv")
        df_concat = io_utils.concatenate_csv_files_with_glob("*test*.csv")
        assert df_concat.equals(pd.concat([df, df]))

        # remove the files created
        os.remove("test1.csv")
        os.remove("test2.csv")

        # test that the function to convert a string to a list works
        assert io_utils.convert_str2list("a,b,c") == ["a", "b", "c"]
        assert io_utils.convert_str2list("a, b, c") == ["a", "b", "c"]

    def test_utils_requests(self, capsys):
        import requests
        from mkt.databases import utils_requests
        from mkt.databases.uniprot import UniProtFASTA

        # conform with SwissProt ID pattern
        uniprot_id = "L91119"
        UniProtFASTA(uniprot_id)
        out, _ = capsys.readouterr()
        assert out == f"Error code: 400 (Bad request)\nUniProt ID: {uniprot_id}\n\n"

        utils_requests.print_status_code_if_res_not_ok(
            requests.get("https://rest.uniprot.org/uniprotkb/TEST"),
            dict_status_code={400: "TEST"},
        )
        out, _ = capsys.readouterr()
        assert out == "Error code: 400 (TEST)\n"

        utils_requests.print_status_code_if_res_not_ok(
            requests.get("https://rest.uniprot.org/uniprotkb/TEST"),
            dict_status_code={200: "TEST"},
        )
        out, _ = capsys.readouterr()
        assert out == "Error code: 400\n"

    def test_cbioportal(self):
        from mkt.databases import cbioportal, config

        config.set_cbioportal_instance("www.cbioportal.org")
        config.set_output_dir(".")

        # test that the function to query the cBioPortal API works
        cbioportal_instance = cbioportal.cBioPortal()
        assert cbioportal_instance.get_instance() == "www.cbioportal.org"
        assert (
            cbioportal_instance.get_url()
            == "https://www.cbioportal.org/api/v2/api-docs"
        )
        assert cbioportal_instance._cbioportal is not None

        # test that server status is up
        assert (
            cbioportal_instance._cbioportal.Server_running_status.getServerStatusUsingGET()
            .response()
            .result["status"]
            == "UP"
        )

        # test that Zehir cohort is available
        study = "msk_impact_2017"
        mutations_instance = cbioportal.Mutations(study_id=study)
        assert mutations_instance.check_entity_id() is True
        assert mutations_instance.get_entity_id() == study
        assert mutations_instance._df.shape[0] == 78142

        # make sure save works - no longer saving to file
        # import os
        # mutations_instance = cbioportal.Mutations(study)
        # mutations_instance.get_cbioportal_cohort_mutations(bool_save=True)
        # assert os.path.isfile(f"{mutations_instance.study_id}_mutations.csv") is True
        # os.remove(f"{mutations_instance.study_id}_mutations.csv")

        panel = "IMPACT341"
        panel_instance = cbioportal.GenePanel(panel_id=panel)
        assert panel_instance.check_entity_id() is True
        assert panel_instance._df.shape[0] == 341
        assert panel_instance._df.shape[1] == 2

    def test_hgnc(self):
        from mkt.databases import hgnc

        abl1 = hgnc.HGNC("Abl1", True)
        abl1.maybe_get_symbol_from_hgnc_search()
        assert abl1.hgnc == "ABL1"
        assert (
            abl1.maybe_get_info_from_hgnc_fetch()["locus_type"][0]
            == "gene with protein product"
        )

        abl1 = hgnc.HGNC("ENSG00000097007", False)
        abl1.maybe_get_symbol_from_hgnc_search()
        assert abl1.hgnc == "ABL1"
        assert (
            abl1.maybe_get_info_from_hgnc_fetch()["locus_type"][0]
            == "gene with protein product"
        )

        test = hgnc.HGNC("test", True)
        test.maybe_get_symbol_from_hgnc_search()
        assert test.ensembl is None
        assert test.maybe_get_info_from_hgnc_fetch()["locus_type"] is None

        test = hgnc.HGNC("test", False)
        test.maybe_get_symbol_from_hgnc_search()
        assert test.hgnc is None
        assert test.maybe_get_info_from_hgnc_fetch() is None

        test = hgnc.HGNC("temp")
        test.maybe_get_symbol_from_hgnc_search(
            custom_field="mane_select", custom_term="ENST00000318560.6"
        )
        assert test.hgnc == "ABL1"

    def test_kincore_klifs(self):
        from itertools import chain

        from mkt.databases import klifs
        from mkt.databases.kincore import (
            align_kincore2uniprot,
            extract_pk_cif_files_as_list,
            harmonize_kincore_fasta_cif,
        )
        from mkt.databases.uniprot import UniProtFASTA

        # test KinCore
        uniprot_id = "P00533"
        egfr_uniprot = UniProtFASTA(uniprot_id)
        dict_kincore = harmonize_kincore_fasta_cif()

        # make sure the number of non-None cif files is correct
        list_dict_cif_hgnc = [
            [entry.cif.hgnc for entry in v if entry.cif is not None]
            for v in dict_kincore.values()
        ]
        list_dict_cif_hgnc = list(chain(*list_dict_cif_hgnc))
        list_kincore_cif = extract_pk_cif_files_as_list()
        assert len(list_dict_cif_hgnc) == len(list_kincore_cif)

        assert len(dict_kincore[uniprot_id]) == 1

        egfr_align = align_kincore2uniprot(
            str_kincore=dict_kincore[uniprot_id][0].fasta.seq,
            str_uniprot=egfr_uniprot._sequence,
        )

        assert (
            egfr_align["seq"]
            == "LRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRY"
        )
        assert egfr_align["start"] == 704
        assert egfr_align["end"] == 978
        assert egfr_align["mismatch"] is None

        # test KLIFS
        temp_obj = klifs.KinaseInfo("EGFR")
        assert len(temp_obj._kinase_info) == 1
        dict_egfr = temp_obj.get_kinase_info()[0]
        if temp_obj.status_code == 200:
            assert dict_egfr["family"] == "EGFR"
            assert dict_egfr["full_name"] == "epidermal growth factor receptor"
            assert dict_egfr["gene_name"] == "EGFR"
            assert dict_egfr["group"] == "TK"
            assert dict_egfr["iuphar"] == 1797
            assert dict_egfr["kinase_ID"] == 406
            assert dict_egfr["name"] == "EGFR"
            assert (
                dict_egfr["pocket"]
                == "KVLGSGAFGTVYKVAIKELEILDEAYVMASVDPHVCRLLGIQLITQLMPFGCLLDYVREYLEDRRLVHRDLAARNVLVITDFGLA"
            )
            assert dict_egfr["species"] == "Human"
            assert dict_egfr["uniprot"] == "P00533"

            egfr_uniprot = UniProtFASTA(dict_egfr["uniprot"])

            # check KLIFS pocket alignment to UniProt sequence
            egfr_pocket = klifs.KLIFSPocket(
                uniprotSeq=egfr_uniprot._sequence,
                klifsSeq=dict_egfr["pocket"],
                idx_kd=(egfr_align["start"] - 1, egfr_align["end"] - 1),
            )

            assert egfr_pocket.list_klifs_substr_actual == [
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
            assert egfr_pocket.list_klifs_substr_match == [
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
            assert egfr_pocket.KLIFS2UniProtIdx == {
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

            assert egfr_pocket.KLIFS2UniProtSeq == {
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

        if 500 <= temp_obj.status_code < 600:
            assert dict_egfr is None

    def test_scrapers(self):
        from mkt.databases import scrapers

        # test that the function to scrape the KinHub database works
        df_kinhub = scrapers.kinhub()
        assert df_kinhub.shape[0] == 536
        assert df_kinhub.shape[1] == 8
        assert "HGNC Name" in df_kinhub.columns
        assert "UniprotID" in df_kinhub.columns

    def test_colors(self, capsys):
        from mkt.databases import colors

        # correct mappings
        assert colors.map_aa_to_single_letter_code("Ala") == "A"
        assert colors.map_aa_to_single_letter_code("ALA") == "A"
        assert colors.map_aa_to_single_letter_code("alanine") == "A"
        assert colors.map_aa_to_single_letter_code("ALANINE") == "A"
        assert colors.map_aa_to_single_letter_code("A") == "A"
        assert colors.map_aa_to_single_letter_code("a") == "A"

        # incorrect mappings
        TEST1 = "X"
        assert colors.map_aa_to_single_letter_code(TEST1) is None
        out, _ = capsys.readouterr()
        assert out == f"Invalid single-letter amino acid: {TEST1.upper()}\n"
        TEST2 = "AL"
        assert colors.map_aa_to_single_letter_code(TEST2) is None
        out, _ = capsys.readouterr()
        assert out == f"Length error and invalid amino acid: {TEST2}\n"
        TEST3 = "XYZ"
        assert colors.map_aa_to_single_letter_code(TEST3) is None
        out, _ = capsys.readouterr()
        assert out == f"Invalid 3-letter amino acid: {TEST3.upper()}\n"
        TEST4 = "TEST"
        assert colors.map_aa_to_single_letter_code(TEST4) is None
        out, _ = capsys.readouterr()
        assert out == f"Invalid amino acid name: {TEST4.lower()}\n"

        # color mapping
        DICT_COLORS = colors.DICT_COLORS
        assert (
            colors.map_single_letter_aa_to_color(
                "A", DICT_COLORS["ALPHABET_PROJECT"]["DICT_COLORS"]
            )
            == "#F0A3FF"
        )
        assert (
            colors.map_single_letter_aa_to_color(
                "A", DICT_COLORS["ASAP"]["DICT_COLORS"]
            )
            == "red"
        )
        assert (
            colors.map_single_letter_aa_to_color(
                "A", DICT_COLORS["RASMOL"]["DICT_COLORS"]
            )
            == "#C8C8C8"
        )
        assert (
            colors.map_single_letter_aa_to_color(
                "A", DICT_COLORS["SHAPELY"]["DICT_COLORS"]
            )
            == "#8CFF8C"
        )
        assert (
            colors.map_single_letter_aa_to_color(
                "A", DICT_COLORS["CLUSTALX"]["DICT_COLORS"]
            )
            == "blue"
        )
        assert (
            colors.map_single_letter_aa_to_color(
                "A", DICT_COLORS["ZAPPO"]["DICT_COLORS"]
            )
            == "#ffafaf"
        )

    def test_uniprot(self):
        from mkt.databases.uniprot import UniProtFASTA

        abl1 = UniProtFASTA("P00519")
        assert (
            abl1._sequence
            == "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR"  # noqa E501
        )

    def test_pfam(self):
        import requests
        from mkt.databases import pfam

        # test that the function to find Pfam domain for a given HGNC symbol and position works
        try:
            df_pfam = pfam.Pfam("P00519")._pfam
            assert df_pfam.shape[0] == 4
            # allow for 18 or 19 columns, depending on the version of the Pfam database
            assert df_pfam.shape[1] == 18 or df_pfam.shape[1] == 19
            assert "uniprot" in df_pfam.columns
            assert "start" in df_pfam.columns
            assert "end" in df_pfam.columns
            assert "name" in df_pfam.columns
            assert (
                df_pfam.loc[
                    df_pfam["name"] == "Protein tyrosine and serine/threonine kinase",
                    "start",
                ].values[0]
                == 242
            )
            assert (
                df_pfam.loc[
                    df_pfam["name"] == "Protein tyrosine and serine/threonine kinase",
                    "end",
                ].values[0]
                == 492
            )
            assert (
                pfam.find_pfam_domain(
                    input_id="p00519",
                    input_position=350,
                    df_ref=df_pfam,
                    col_ref_id="uniprot",
                )
                == "Protein tyrosine and serine/threonine kinase"
            )
        except requests.exceptions.RetryError as e:
            # Allow test to pass if API returns 500 errors (common in CI environments)
            if "500 error responses" in str(e):
                pytest.skip("Pfam API returned 500 errors - skipping test")
            else:
                raise

    def test_protvar(self):
        from mkt.databases.protvar import ProtvarScore

        temp_obj = ProtvarScore(database="AM", uniprot_id="P00519", pos=292, mut="D")
        assert len(temp_obj._protvar_score) == 1
        assert temp_obj._protvar_score[0]["amPathogenicity"] == 0.4217
        assert temp_obj._protvar_score[0]["amClass"] == "AMBIGUOUS"

    # @pytest.mark.skip(
    #     reason="NCBI API currently returning 404 error as of 3/26 - see if this improves."
    # )
    def test_ncbi(self):
        import requests
        from mkt.databases import ncbi

        try:
            seq_obj = ncbi.ProteinNCBI(accession="EAX02438.1")
            assert seq_obj.list_headers == [
                "EAX02438.1 BR serine/threonine kinase 2, isoform CRA_c [Homo sapiens]"
            ]
            assert seq_obj.list_seq == [
                "MTSTGKDGGAQHAQYVGPYRLEKTLGKGQTGLVKLGVHCVTCQKVAIKIVNREKLSESVLMKVEREIAILKLIEHPHVLKLHDVYENKKYLYLVLEHVSGGELFDYLVKKGRLTPKEARKFFRQIISALDFCHSHSICHRDLKPENLLLDEKNNIRIADFGMASLQVGDSLLETSCGSPHYACPEVIRGEKYDGRKADVWSCGVILFALLVGALPFDDDNLRQLLEKVKRGVFHMPHFIPPDCQSLLRGMIEVDAARRLTLEHIQKHIWYIGGKNEPEPEQPIPRKVQIRSLPSLEDIDPDVLDSMHSLGCFRDRNKLLQDLLSEEENQEKMIYFLLLDRKERYPSQEDEDLPPRNEIDPPRKRVDSPMLNRHGKRRPERKSMEVLSVTDGGSPVPARRAIEMAQHGQRSRSISGASSGLSTSPLSSPRVTPHPSPRGSPLPTPKGTPVHTPKESPAGTPNPTPPSSPSVGGVPWRARLNSIKNSFLGSPRFHRRKLQVPTPEEMSNLTPESSPELAKKSWFGNFISLEKEEQIFVVIKDKPLSSIKADIVHAFLSIPSLSHSVISQTSFRAEYKATGGPAVFQKPVKFQVDITYTEGGEAQKENGIYSVTFTLLSGPSRRFKRVVETIQAQLLSTHDPPAAQHLSEPPPPAPGLSWGAGLKGQKVATSYESSL"
            ]
        except requests.exceptions.RetryError as e:
            # Allow test to pass if API returns 500 errors (common in CI environments)
            if "500 error responses" in str(e):
                pytest.skip("NCBI API returned 500 errors - skipping test")
            else:
                raise

    def test_chembl(self):
        from mkt.databases import chembl

        # drug present
        drug = "erlotinib"
        # ChEMBLMoleculeSearch
        chembl_query = chembl.ChEMBLMoleculeSearch(id=drug)
        set_id = set(chembl_query.get_chembl_id())
        assert {
            "CHEMBL1079742",
            "CHEMBL3186743",
            "CHEMBL5220042",
            "CHEMBL5220676",
            "CHEMBL553",
            "CHEMBL5965928",
        } == set_id
        # ChEMBLMoleculeExact
        assert chembl.ChEMBLMoleculeExact(id=drug).get_chembl_id() == ["CHEMBL553"]
        # ChEMBLMoleculePreferred
        assert chembl.ChEMBLMoleculePreferred(id=drug).get_chembl_id() == ["CHEMBL553"]

        # drug not present
        drug = "TESTTESTTEST"
        assert chembl.ChEMBLMoleculeSearch(id=drug).get_chembl_id() == []
        assert chembl.ChEMBLMoleculeExact(id=drug).get_chembl_id() == []
        assert chembl.ChEMBLMoleculePreferred(id=drug).get_chembl_id() == []

    def test_opentargets(self):
        from mkt.databases import open_targets

        # test that the function to get drug mechanism of action works
        drug_moa = open_targets.OpenTargetsDrugMoA(chembl_id="CHEMBL1079742")
        set_moa = drug_moa.get_moa()
        assert set_moa == {"EGFR"}

        test = open_targets.OpenTargetsDrugMoA(chembl_id="TEST")
        assert test.get_moa() is None

    # ------------------------------------------------------------------
    # plot_config tests
    # ------------------------------------------------------------------

    def test_plot_config_defaults(self):
        """Test that all config dataclasses instantiate with expected defaults."""
        from mkt.databases.plot_config import (
            ColKinaseColorConfig,
            DataSourceConfig,
            DynamicRangePlotConfig,
            FamilyColorConfig,
            MatplotlibRCConfig,
            MetricsBoxplotConfig,
            OutputConfig,
            PlotDatasetConfig,
            RidgelinePlotConfig,
            SequenceSchematicConfig,
            StackedBarchartConfig,
            VennDiagramConfig,
        )

        rc = MatplotlibRCConfig()
        assert rc.svg_fonttype == "path"
        assert rc.pdf_fonttype == 42
        assert rc.text_usetex is False

        assert FamilyColorConfig().use_kinase_group_colors is True
        assert ColKinaseColorConfig().construct_unaligned == [242, 101, 41]
        assert DynamicRangePlotConfig().bins == 100
        assert RidgelinePlotConfig().overlap == 0.1
        assert StackedBarchartConfig().figsize_height == 7
        assert VennDiagramConfig().circle_alpha == 0.6
        assert MetricsBoxplotConfig().box_widths == 0.6
        assert SequenceSchematicConfig().n_show_start == 40
        assert DataSourceConfig().davis_csv == "data/davis_data_processed.csv"
        assert OutputConfig().bool_svg is True

        # top-level aggregator
        cfg = PlotDatasetConfig()
        assert isinstance(cfg.matplotlib_rc, MatplotlibRCConfig)
        assert isinstance(cfg.dynamic_range, DynamicRangePlotConfig)
        assert isinstance(cfg.data_sources, DataSourceConfig)
        assert cfg.output.bool_png is True

    def test_family_color_config_get_colors(self):
        """Test FamilyColorConfig.get_colors in both modes."""
        from mkt.databases.plot_config import FamilyColorConfig

        # default mode uses curated DICT_KINASE_GROUP_COLORS
        cfg = FamilyColorConfig()
        colors = cfg.get_colors()
        assert isinstance(colors, dict)
        assert len(colors) > 0
        assert "TK" in colors

        # filtered + reordered families
        cfg2 = FamilyColorConfig(families=["TK", "Other"])
        colors2 = cfg2.get_colors()
        assert list(colors2.keys()) == ["TK", "Other"]

        # seaborn palette mode
        cfg3 = FamilyColorConfig(use_kinase_group_colors=False)
        colors3 = cfg3.get_colors()
        assert isinstance(colors3, dict)
        assert len(colors3) > 0

    def test_col_kinase_color_config_as_rgb_dict(self):
        """Test ColKinaseColorConfig.as_rgb_dict returns 0-1 scaled RGB tuples."""
        from mkt.databases.plot_config import ColKinaseColorConfig

        cfg = ColKinaseColorConfig()
        rgb = cfg.as_rgb_dict()
        assert set(rgb.keys()) == {
            "construct_unaligned",
            "klifs_region_aligned",
            "klifs_residues_only",
        }
        # values should be 0-1 scaled
        for v in rgb.values():
            assert len(v) == 3
            assert all(0 <= c <= 1 for c in v)
        # check specific default: construct_unaligned = [242, 101, 41]
        assert abs(rgb["construct_unaligned"][0] - 242 / 255) < 1e-6
        assert abs(rgb["construct_unaligned"][1] - 101 / 255) < 1e-6
        assert abs(rgb["construct_unaligned"][2] - 41 / 255) < 1e-6

    def test_plot_config_from_yaml(self, tmp_path):
        """Test PlotDatasetConfig.from_yaml loads overrides and keeps defaults."""
        from mkt.databases.plot_config import PlotDatasetConfig

        yaml_content = (
            "matplotlib_rc:\n"
            '  svg_fonttype: "none"\n'
            "  pdf_fonttype: 3\n"
            "dynamic_range:\n"
            "  bins: 50\n"
            "  alpha: 0.5\n"
        )
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        cfg = PlotDatasetConfig.from_yaml(yaml_file)
        # overridden values
        assert cfg.matplotlib_rc.svg_fonttype == "none"
        assert cfg.matplotlib_rc.pdf_fonttype == 3
        assert cfg.dynamic_range.bins == 50
        assert cfg.dynamic_range.alpha == 0.5
        # unspecified fields keep defaults
        assert cfg.ridgeline.overlap == 0.1
        assert cfg.output.bool_svg is True

    # ------------------------------------------------------------------
    # plot pure-computation tests
    # ------------------------------------------------------------------

    def test_convert_percentile_functions(self):
        """Test convert_to_percentile and convert_from_percentile round-trip."""
        from mkt.databases.plot import convert_from_percentile, convert_to_percentile

        assert convert_to_percentile(5, orig_max=10) == 50.0
        assert convert_to_percentile(10, orig_max=10) == 100.0
        assert convert_to_percentile(0, orig_max=10) == 0.0

        assert convert_from_percentile(50, orig_max=10) == 5.0
        assert convert_from_percentile(100, orig_max=10) == 10.0
        assert convert_from_percentile(0, orig_max=10) == 0.0

        # round-trip
        val = 7.5
        assert convert_from_percentile(convert_to_percentile(val)) == val

    def test_generate_venn_diagram_dict(self):
        """Test generate_venn_diagram_dict with synthetic DataFrame."""
        import numpy as np
        import pandas as pd
        from mkt.databases.plot import generate_venn_diagram_dict

        df = pd.DataFrame(
            {
                "kinase_name": ["EGFR", "ABL1", "BRAF", "CDK2"],
                "seq_construct_unaligned": ["ACGT", np.nan, "TGCA", "AAAA"],
                "seq_klifs_region_aligned": ["ACGT", "TTTT", np.nan, "AAAA"],
                "seq_klifs_residues_only": [np.nan, "TTTT", "TGCA", "AAAA"],
            }
        )
        result = generate_venn_diagram_dict(df)
        assert set(result.keys()) == {
            "Construct Unaligned",
            "KLIFS Region Aligned",
            "Klifs Residues Only",
        }
        assert set(result["Construct Unaligned"]) == {"EGFR", "BRAF", "CDK2"}
        assert set(result["KLIFS Region Aligned"]) == {"EGFR", "ABL1", "CDK2"}
        assert set(result["Klifs Residues Only"]) == {"ABL1", "BRAF", "CDK2"}

    def test_get_klifs_position_colors(self):
        """Test _get_klifs_position_colors returns 85 (region, color) tuples."""
        from mkt.databases.plot import _get_klifs_position_colors

        colors = _get_klifs_position_colors()
        assert len(colors) == 85
        assert all(isinstance(c, tuple) and len(c) == 2 for c in colors)
        assert all(isinstance(c[0], str) and isinstance(c[1], str) for c in colors)

    def test_map_aligned_to_klifs_colors(self):
        """Test _map_aligned_to_klifs_colors with a small synthetic example."""
        from mkt.databases.plot import _map_aligned_to_klifs_colors

        # aligned: A - B C   (4 chars)
        # klifs:   A - B     (3 chars, 2 non-gap)
        # klifs positions: A matches pocket pos 0 (yellow), B matches pos 2 (orange)
        # C is not in KLIFS -> gets alphabet color
        seq_aligned = "A-BC"
        seq_klifs_only = "A-B"
        dict_aa_colors = {"A": "red", "B": "blue", "C": "green"}
        klifs_pos_colors = [
            ("I", "yellow"),
            ("g.l", "purple"),
            ("II", "orange"),
        ]

        colors = _map_aligned_to_klifs_colors(
            seq_aligned, seq_klifs_only, dict_aa_colors, klifs_pos_colors
        )
        assert len(colors) == 4
        assert colors[0] == "yellow"  # A matched to KLIFS pocket pos 0
        assert colors[1] == "white"  # gap
        assert colors[2] == "orange"  # B matched to KLIFS pocket pos 2
        assert colors[3] == "green"  # C not in KLIFS -> alphabet color

    def test_sequence_alignment_get_colors(self):
        """Test SequenceAlignment.get_colors static method."""
        from mkt.databases.plot import SequenceAlignment

        colors = SequenceAlignment.get_colors(
            ["A", "B", "C"],
            {"A": "red", "B": "green", "C": "blue"},
        )
        assert colors == ["red", "green", "blue"]

    def test_apply_matplotlib_rc(self):
        """Test apply_matplotlib_rc sets rcParams correctly."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mkt.databases.plot import apply_matplotlib_rc
        from mkt.databases.plot_config import MatplotlibRCConfig

        rc = MatplotlibRCConfig(svg_fonttype="none", pdf_fonttype=3, text_usetex=False)
        apply_matplotlib_rc(rc)
        assert plt.rcParams["svg.fonttype"] == "none"
        assert plt.rcParams["pdf.fonttype"] == 3
        assert plt.rcParams["text.usetex"] is False

    # ------------------------------------------------------------------
    # plot smoke tests (verify execution, not visual output)
    # ------------------------------------------------------------------

    def test_plot_dynamic_range_smoke(self, tmp_path):
        """Smoke test: plot_dynamic_range runs without error on synthetic data."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from mkt.databases.plot import plot_dynamic_range

        # y column = -log10(Kd) values; function applies 10^(-y)
        df_davis = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0] * 20})
        # y column = percent inhibition values
        df_pkis2 = pd.DataFrame({"y": [10.0, 50.0, 90.0, 30.0, 70.0] * 20})

        output_path = str(tmp_path / "dynamic_range")
        plot_dynamic_range(df_davis, df_pkis2, output_path)
        plt.close("all")

        saved = list(tmp_path.glob("dynamic_range.*"))
        assert len(saved) >= 1

    def test_plot_ridgeline_smoke(self, tmp_path):
        """Smoke test: plot_ridgeline runs without error on synthetic data."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from mkt.databases.plot import plot_ridgeline
        from mkt.databases.plot_config import FamilyColorConfig

        rng = np.random.default_rng(42)
        families = ["TK", "TKL", "STE", "Other"]
        rows = []
        for fam in families:
            for i in range(20):
                rows.append(
                    {
                        "kinase_name": f"{fam}_{i}",
                        "family": fam,
                        "fraction_construct": rng.uniform(0.3, 1.0),
                        "source": "Davis",
                    }
                )
        df = pd.DataFrame(rows)

        output_path = str(tmp_path / "ridgeline")
        plot_ridgeline(df, output_path, family_cfg=FamilyColorConfig())
        plt.close("all")

        saved = list(tmp_path.glob("ridgeline.*"))
        assert len(saved) >= 1

    def test_plot_stacked_barchart_smoke(self, tmp_path):
        """Smoke test: plot_stacked_barchart runs without error on synthetic data."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        from mkt.databases.plot import plot_stacked_barchart
        from mkt.databases.plot_config import FamilyColorConfig

        df = pd.DataFrame(
            {
                "family": ["TK", "TK", "TKL", "TKL"],
                "bool_uniprot2refseq": [True, False, True, False],
                "count": [30, 10, 20, 5],
                "source": ["Davis", "Davis", "Davis", "Davis"],
            }
        )

        output_path = str(tmp_path / "stacked_barchart")
        plot_stacked_barchart(df, output_path, family_cfg=FamilyColorConfig())
        plt.close("all")

        saved = list(tmp_path.glob("stacked_barchart.*"))
        assert len(saved) >= 1

    def test_plot_venn_diagram_smoke(self, tmp_path):
        """Smoke test: plot_venn_diagram runs without error on synthetic data."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from mkt.databases.plot import plot_venn_diagram

        df = pd.DataFrame(
            {
                "kinase_name": ["EGFR", "ABL1", "BRAF", "CDK2", "SRC"],
                "seq_construct_unaligned": ["ACGT", "TTTT", np.nan, "AAAA", "CCCC"],
                "seq_klifs_region_aligned": ["ACGT", np.nan, "TGCA", "AAAA", "CCCC"],
                "seq_klifs_residues_only": [np.nan, "TTTT", "TGCA", "AAAA", "CCCC"],
            }
        )

        output_path = str(tmp_path / "venn_diagram")
        plot_venn_diagram(df, output_path, source_name="Test")
        plt.close("all")

        saved = list(tmp_path.glob("venn_diagram.*"))
        assert len(saved) >= 1

    def test_plot_metrics_boxplot_smoke(self, tmp_path):
        """Smoke test: plot_metrics_boxplot runs without error on synthetic data."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from mkt.databases.plot import plot_metrics_boxplot

        rng = np.random.default_rng(42)
        rows = []
        for col_kinase in [
            "construct_unaligned",
            "klifs_region_aligned",
            "klifs_residues_only",
        ]:
            for fold in range(5):
                rows.append(
                    {
                        "col_kinase": col_kinase,
                        "source": "davis",
                        "fold": fold,
                        "avg_stable_epoch": 10,
                        "mse": rng.normal(0.5, 0.1),
                    }
                )
        df = pd.DataFrame(rows)

        output_path = str(tmp_path / "boxplot")
        plot_metrics_boxplot(df, output_path)
        plt.close("all")

        saved = list(tmp_path.glob("boxplot.*"))
        assert len(saved) >= 1
