import os

import pandas as pd
import pytest
from mkt.databases import config, io_utils


@pytest.fixture(autouse=True)
def _set_output_dir():
    """Ensure OUTPUT_DIR is set for all tests in this module."""
    config.set_output_dir(".")


class TestSaveLoadDataframe:
    def test_save_and_load_csv_roundtrip(self, tmp_path):
        """save_dataframe_to_csv / load_csv_to_dataframe round-trip."""
        config.set_output_dir(str(tmp_path))
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        io_utils.save_dataframe_to_csv(df, "test1.csv")
        df_read = io_utils.load_csv_to_dataframe("test1.csv")
        assert df.equals(df_read)

    def test_concatenate_csv_files_with_glob(self, tmp_path):
        """concatenate_csv_files_with_glob merges matching CSV files."""
        config.set_output_dir(str(tmp_path))
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        io_utils.save_dataframe_to_csv(df, "test1.csv")
        io_utils.save_dataframe_to_csv(df, "test2.csv")
        # glob from inside tmp_path
        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            df_concat = io_utils.concatenate_csv_files_with_glob("*test*.csv")
        finally:
            os.chdir(orig_dir)
        assert df_concat.equals(pd.concat([df, df]))


class TestConvertStr2List:
    def test_comma_separated(self):
        assert io_utils.convert_str2list("a,b,c") == ["a", "b", "c"]

    def test_comma_space_separated(self):
        assert io_utils.convert_str2list("a, b, c") == ["a", "b", "c"]
