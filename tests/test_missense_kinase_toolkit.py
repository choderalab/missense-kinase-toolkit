"""
Unit and regression test for the missense_kinase_toolkit package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import missense_kinase_toolkit


def test_missense_kinase_toolkit_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "missense_kinase_toolkit" in sys.modules


def test_kinhub_scraper():
    from missense_kinase_toolkit import scrapers

    df_kinhub = scrapers.kinhub()

    assert df_kinhub.shape[0] == 517
    assert df_kinhub.shape[1] == 8
    assert "HGNC Name" in df_kinhub.columns
    assert "UniprotID" in df_kinhub.columns


def test_klifs_KinaseInfo():
    from missense_kinase_toolkit import klifs

    dict_egfr = klifs.KinaseInfo("EGFR")._kinase_info

    assert dict_egfr["family"] == "EGFR"
    assert dict_egfr["full_name"] == "epidermal growth factor receptor"
    assert dict_egfr["gene_name"] == "EGFR"
    assert dict_egfr["group"] == "TK"
    assert dict_egfr["iuphar"] == 1797
    assert dict_egfr["kinase_ID"] == 406
    assert dict_egfr["name"] == "EGFR"
    assert dict_egfr["pocket"] == "KVLGSGAFGTVYKVAIKELEILDEAYVMASVDPHVCRLLGIQLITQLMPFGCLLDYVREYLEDRRLVHRDLAARNVLVITDFGLA"
    assert dict_egfr["species"] == "Human"
    assert dict_egfr["uniprot"] == "P00533"


def test_io_utils():
    from missense_kinase_toolkit import io_utils
    import pandas as pd
    import os

    os.environ["OUTPUT_DIR"] = "."
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    io_utils.save_dataframe_to_csv(df, "test1.csv")
    df_read = io_utils.load_csv_to_dataframe("test1.csv")
    assert df.equals(df_read)

    io_utils.save_dataframe_to_csv(df, "test2.csv")
    df_concat = io_utils.concatenate_csv_files_with_glob("*test*.csv")
    assert df_concat.equals(pd.concat([df, df]))

    os.remove("test1.csv")
    os.remove("test2.csv")

    assert io_utils.convert_str2list("a,b,c") == ["a", "b", "c"]
    assert io_utils.convert_str2list("a, b, c") == ["a", "b", "c"]
