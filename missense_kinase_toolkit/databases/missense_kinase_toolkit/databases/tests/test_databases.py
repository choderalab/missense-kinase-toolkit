import pytest


class TestDatabases:
    def test_missense_kinase_toolkit_database_imported(self):
        """Test if module is imported."""
        import sys

        import missense_kinase_toolkit.databases  # noqa F401

        assert "missense_kinase_toolkit.databases" in sys.modules

    def test_config(self):
        from missense_kinase_toolkit.databases import config

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

        from missense_kinase_toolkit.databases import config, io_utils

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

    def test_requests_wrapper(self, capsys):
        import requests

        from missense_kinase_toolkit.databases import uniprot, utils_requests

        uniprot.UniProt("TEST")
        out, _ = capsys.readouterr()
        assert out == "Error code: 400 (Bad request)\n"

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
        import os

        from missense_kinase_toolkit.databases import cbioportal, config

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
        list_studies = (
            cbioportal_instance._cbioportal.Studies.getAllStudiesUsingGET().result()
        )
        list_study_ids = [study.studyId for study in list_studies]
        assert study in list_study_ids

        # test that the function to get all mutations by study works
        df = cbioportal.Mutations(study).get_cbioportal_cohort_mutations()
        assert df.shape[0] == 78142

        # make sure save works
        mutations_instance = cbioportal.Mutations(study)
        mutations_instance.get_cbioportal_cohort_mutations(bool_save=True)
        assert os.path.isfile(f"{mutations_instance.study_id}_mutations.csv") is True
        os.remove(f"{mutations_instance.study_id}_mutations.csv")

        assert mutations_instance.get_study_id() == study
        assert mutations_instance._mutations is not None

    def test_hgnc(self):
        from missense_kinase_toolkit.databases import hgnc

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

    def test_klifs(self):
        from missense_kinase_toolkit.databases import klifs

        dict_egfr = klifs.KinaseInfo("EGFR")._kinase_info

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

    def test_scrapers(self):
        from missense_kinase_toolkit.databases import scrapers

        # test that the function to scrape the KinHub database works
        df_kinhub = scrapers.kinhub()
        assert df_kinhub.shape[0] == 517
        assert df_kinhub.shape[1] == 8
        assert "HGNC Name" in df_kinhub.columns
        assert "UniprotID" in df_kinhub.columns

    def test_colors(self, capsys):
        from missense_kinase_toolkit.databases import colors

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
        from missense_kinase_toolkit.databases import uniprot

        abl1 = uniprot.UniProt("P00519")
        assert (
            abl1._sequence
            == "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR"  # noqa E501
        )

    def test_pfam(self):
        from missense_kinase_toolkit.databases import pfam

        # test that the function to find Pfam domain for a given HGNC symbol and position works
        df_pfam = pfam.Pfam("P00519")._pfam
        assert df_pfam.shape[0] == 4
        assert df_pfam.shape[1] == 18
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
                df_pfam["name"] == "Protein tyrosine and serine/threonine kinase", "end"
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

def test_ncbi():
    from missense_kinase_toolkit.databases import ncbi

    seq_obj = ncbi.ProteinNCBI(accession="EAX02438.1")
    assert seq_obj.list_headers == ['EAX02438.1 BR serine/threonine kinase 2, isoform CRA_c [Homo sapiens]']
    assert seq_obj.list_seq == ['MTSTGKDGGAQHAQYVGPYRLEKTLGKGQTGLVKLGVHCVTCQKVAIKIVNREKLSESVLMKVEREIAILKLIEHPHVLKLHDVYENKKYLYLVLEHVSGGELFDYLVKKGRLTPKEARKFFRQIISALDFCHSHSICHRDLKPENLLLDEKNNIRIADFGMASLQVGDSLLETSCGSPHYACPEVIRGEKYDGRKADVWSCGVILFALLVGALPFDDDNLRQLLEKVKRGVFHMPHFIPPDCQSLLRGMIEVDAARRLTLEHIQKHIWYIGGKNEPEPEQPIPRKVQIRSLPSLEDIDPDVLDSMHSLGCFRDRNKLLQDLLSEEENQEKMIYFLLLDRKERYPSQEDEDLPPRNEIDPPRKRVDSPMLNRHGKRRPERKSMEVLSVTDGGSPVPARRAIEMAQHGQRSRSISGASSGLSTSPLSSPRVTPHPSPRGSPLPTPKGTPVHTPKESPAGTPNPTPPSSPSVGGVPWRARLNSIKNSFLGSPRFHRRKLQVPTPEEMSNLTPESSPELAKKSWFGNFISLEKEEQIFVVIKDKPLSSIKADIVHAFLSIPSLSHSVISQTSFRAEYKATGGPAVFQKPVKFQVDITYTEGGEAQKENGIYSVTFTLLSGPSRRFKRVVETIQAQLLSTHDPPAAQHLSEPPPPAPGLSWGAGLKGQKVATSYESSL']
