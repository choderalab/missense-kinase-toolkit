import logging
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


@pytest.fixture
def _empty_sources(monkeypatch):
    """Isolate the module-level sources lookup and download log from other tests."""
    from mkt.schema import kinase_schema

    monkeypatch.setattr(kinase_schema, "_DICT_SOURCES", {})
    monkeypatch.setattr(io_utils, "_SET_DOWNLOADED", set())


class TestDataSourceSha256:
    def test_provenance_is_sha256_only_and_registers_full_entry(
        self, tmp_path, _empty_sources
    ):
        """provenance() returns just the file's SHA-256; the full entry resolves from it."""
        import hashlib

        path_src = tmp_path / "source.txt"
        path_src.write_bytes(b"kinase data")
        source = io_utils.DataSource(
            name="source.txt",
            path=str(path_src),
            version="v1",
            citation="Someone, 2026.",
        )

        prov = source.provenance()
        str_sha = hashlib.sha256(b"kinase data").hexdigest()
        assert prov.model_dump() == {"sha256": str_sha}
        full = prov.resolve()
        assert (full.name, full.version, full.citation, full.sha256) == (
            "source.txt",
            "v1",
            "Someone, 2026.",
            str_sha,
        )

    def test_query_date_rules(self, tmp_path, _empty_sources, caplog):
        """An unchanged file keeps its previous date; otherwise it's dated today, noting
        when the file was neither downloaded nor previously recorded."""
        from datetime import date

        from mkt.schema.kinase_schema import Provenance, register_sources

        path_src = tmp_path / "source.txt"
        path_src.write_bytes(b"kinase data")
        source = io_utils.DataSource(name="source.txt", path=str(path_src))
        str_sha = source.sha256()
        str_today = date.today().isoformat()

        caplog.set_level(logging.INFO, logger=io_utils.__name__)
        assert source.query_date(str_sha) == str_today
        assert "not downloaded this run" in caplog.text

        caplog.clear()
        io_utils._SET_DOWNLOADED.add(str(path_src))
        assert source.query_date(str_sha) == str_today
        assert "not downloaded this run" not in caplog.text

        register_sources(
            {
                str_sha: Provenance(
                    name="source.txt", query_date="2026-01-02", sha256=str_sha
                )
            }
        )
        assert source.query_date(str_sha) == "2026-01-02"

    def test_changed_file_rehashes(self, tmp_path):
        """A changed file (new mtime/size) gets a new hash despite the cache."""
        path_src = tmp_path / "source.txt"
        path_src.write_bytes(b"v1")
        source = io_utils.DataSource(name="source.txt", path=str(path_src))
        sha_v1 = source.sha256()

        path_src.write_bytes(b"version 2")
        assert source.sha256() != sha_v1

    def test_missing_file_has_inline_provenance(self, tmp_path, _empty_sources):
        """A source file that isn't present yields inline provenance rather than raising."""
        source = io_utils.DataSource(name="absent", path=str(tmp_path / "absent"))
        assert source.sha256() is None
        prov = source.provenance()
        assert prov.sha256 is None and prov.name == "absent"


class TestConvertStr2List:
    def test_comma_separated(self):
        assert io_utils.convert_str2list("a,b,c") == ["a", "b", "c"]

    def test_comma_space_separated(self):
        assert io_utils.convert_str2list("a, b, c") == ["a", "b", "c"]
