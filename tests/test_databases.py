"""
Unit and regression test for the missense_kinase_toolkit package.
"""

# Import package, test suite, and other packages as needed
import pytest


def test_missense_kinase_toolkit_database_imported():
    """Test if module is imported."""
    import sys
    import missense_kinase_toolkit.databases
    
    assert "missense_kinase_toolkit.databases" in sys.modules


def test_config():
    from missense_kinase_toolkit.databases import config

    # test that the function to set the output directory works
    config.set_output_dir("test")
    assert config.get_output_dir() == "test"

    # test that the function to set the request cache works
    config.set_request_cache("test")
    assert config.maybe_get_request_cache() == "test"

    # test that the function to set the cBioPortal instance works
    config.set_cbioportal_instance("test")
    assert config.get_cbioportal_instance() == "test"

    # test that the function to set the cBioPortal token works
    config.set_cbioportal_token("test")
    assert config.maybe_get_cbioportal_token() == "test"


def test_cbioportal():
    from missense_kinase_toolkit.databases import cbioportal

    # test that the function to set the API key for cBioPortal works
    cbioportal.cBioPortal()._set_api_key()

    # test that the function to query the cBioPortal API works
    cbioportal.cBioPortal().query_cbioportal_api()


def test_io_utils():
    from missense_kinase_toolkit.databases import io_utils
    import pandas as pd
    import os

    os.environ["OUTPUT_DIR"] = "."

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


def test_kinhub_scraper():
    from missense_kinase_toolkit.databases import scrapers

    df_kinhub = scrapers.kinhub()

    assert df_kinhub.shape[0] == 517
    assert df_kinhub.shape[1] == 8
    assert "HGNC Name" in df_kinhub.columns
    assert "UniprotID" in df_kinhub.columns


def test_klifs_KinaseInfo():
    from missense_kinase_toolkit.databases import klifs

    dict_egfr = klifs.KinaseInfo("EGFR")._kinase_info

    assert dict_egfr["family"]      ==      "EGFR"
    assert dict_egfr["full_name"]   ==      "epidermal growth factor receptor"
    assert dict_egfr["gene_name"]   ==      "EGFR"
    assert dict_egfr["group"]       ==      "TK"
    assert dict_egfr["iuphar"]      ==      1797
    assert dict_egfr["kinase_ID"]   ==      406
    assert dict_egfr["name"]        ==      "EGFR"
    assert dict_egfr["pocket"]      ==      "KVLGSGAFGTVYKVAIKELEILDEAYVMASVDPHVCRLLGIQLITQLMPFGCLLDYVREYLEDRRLVHRDLAARNVLVITDFGLA"
    assert dict_egfr["species"]     ==      "Human"
    assert dict_egfr["uniprot"]     ==      "P00533"
