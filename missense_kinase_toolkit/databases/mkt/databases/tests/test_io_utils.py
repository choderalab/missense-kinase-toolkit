import os
import tarfile

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


class TestCreateTarWithoutMetadata:
    @staticmethod
    def _write_tree(path):
        """Write two files (one nested) plus a macOS AppleDouble file under ``path``."""
        (path / "b").mkdir(parents=True)
        (path / "a.json").write_text('{"x": 1}')
        (path / "b" / "c.json").write_text('{"y": 2}')
        (path / "._a.json").write_text("appledouble")
        return path

    def test_reproducible_and_stripped(self, tmp_path):
        """Rebuilding identical files gives identical bytes with member metadata cleared."""
        src = self._write_tree(tmp_path / "src")
        tar_one, tar_two = tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"
        io_utils.create_tar_without_metadata(str(src), str(tar_one))
        # a new checkout changes file mtimes
        os.utime(src / "a.json", (1_000_000, 1_000_000))
        io_utils.create_tar_without_metadata(str(src), str(tar_two))
        assert tar_one.read_bytes() == tar_two.read_bytes()

        with tarfile.open(tar_one) as tar:
            members = tar.getmembers()
        assert [member.name for member in members] == ["a.json", "b/c.json"]
        for member in members:
            assert (member.mtime, member.uid, member.gid) == (0, 0, 0)
            assert (member.uname, member.gname) == ("", "")

    def test_missing_source_raises(self, tmp_path):
        """A missing source directory raises instead of writing an empty archive."""
        with pytest.raises(NotADirectoryError):
            io_utils.create_tar_without_metadata(
                str(tmp_path / "absent"), str(tmp_path / "out.tar.gz")
            )


class TestConvertStr2List:
    def test_comma_separated(self):
        assert io_utils.convert_str2list("a,b,c") == ["a", "b", "c"]

    def test_comma_space_separated(self):
        assert io_utils.convert_str2list("a, b, c") == ["a", "b", "c"]
