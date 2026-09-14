import logging
import tarfile

import pytest
from mkt.schema import io_utils
from mkt.schema.utils import (
    LIST_MANIFEST_EXTRA_PATHS,
    return_manifest_tallies,
    return_submodel_paths,
    rgetattr,
)

DICT_CORPUS_COUNTS = {
    "uniprot": 543,
    "kinhub": 517,
    "klifs": 539,
    "klifs.pocket_seq": 519,
    "pfam": 533,
    "kincore": 497,
    "kincore.fasta": 497,
    "kincore.cif": 437,
    "kincore.cif.sasa": 436,
    "kincore.cif.superposition": 437,
    "kincore.msa": 497,
    "alphafold": 530,
    "alphafold.sasa": 509,
    "alphafold.superposition": 530,
    "exon": 523,
    "KLIFS2UniProtIdx": 519,
    "KLIFS2UniProtSeq": 519,
}
"""dict[str, int]: Expected non-None counts per path in the packaged KinaseInfo.tar.gz."""

DICT_CORPUS_SOURCE_VERSIONS = {
    "kincore.fasta": {"v1": 60, "v3": 437},
    "kincore.cif": {"v2": 437},
    "alphafold": {"v6": 530},
}
"""dict[str, dict[str, int]]: Expected source-version tallies in the packaged tar."""


@pytest.fixture(scope="module")
def dict_sample(dict_kinase):
    """Two-entry subset (KinCoRe CIF + AlphaFold-only) for archive round-trips."""
    return {key: dict_kinase[key] for key in ("ABL1", "BUB1B")}


def _write_dir(tmp_path, dict_entries, manifest=None):
    """Serialize entries (and an optional manifest) into ``tmp_path/KinaseInfo``."""
    path_dir = tmp_path / "KinaseInfo"
    io_utils.serialize_kinase_dict(dict_entries, str_path=str(path_dir))
    if manifest is not None:
        (path_dir / io_utils.STR_MANIFEST_FILENAME).write_text(
            manifest.model_dump_json(indent=4)
        )
    return path_dir


def _write_tar(tmp_path, dict_entries, manifest=None):
    """Write entries (and an optional manifest) to ``tmp_path/KinaseInfo.tar.gz``."""
    path_dir = _write_dir(tmp_path, dict_entries, manifest)
    path_tar = tmp_path / "KinaseInfo.tar.gz"
    with tarfile.open(path_tar, "w:gz") as tar:
        for path in sorted(path_dir.iterdir()):
            tar.add(path, arcname=path.name)
    return str(path_tar)


def test_manifest_tallies_match_corpus(dict_kinase):
    """Tallies on the packaged dict agree with the hardcoded ``test_dict_counts``."""
    counts, source_versions = return_manifest_tallies(dict_kinase)
    assert counts == DICT_CORPUS_COUNTS
    assert source_versions == DICT_CORPUS_SOURCE_VERSIONS


def test_manifest_extra_paths_resolve(dict_kinase):
    """Every explicit extra path resolves on some entry (``rgetattr`` hides typos)."""
    for path in LIST_MANIFEST_EXTRA_PATHS:
        assert any(rgetattr(obj, path) is not None for obj in dict_kinase.values())
    assert not set(LIST_MANIFEST_EXTRA_PATHS) & set(return_submodel_paths())


def test_untar_skips_manifest(tmp_path, dict_sample):
    """The manifest is neither an entry ID nor an extracted kinase file."""
    manifest = io_utils.Manifest.from_kinase_dict(dict_sample)
    str_tar = _write_tar(tmp_path, dict_sample, manifest)

    list_entries, dict_str = io_utils.untar_files_in_memory(str_tar)
    assert sorted(list_entries) == sorted(dict_sample)
    assert io_utils.STR_MANIFEST_FILENAME not in dict_str

    list_entries, _ = io_utils.untar_files_in_memory(str_tar, bool_extract=False)
    assert "manifest" not in list_entries


def test_load_manifest(tmp_path, dict_sample):
    """``load_manifest`` reads from a tar or directory and returns None when absent."""
    manifest = io_utils.Manifest.from_kinase_dict(dict_sample)
    str_tar = _write_tar(tmp_path, dict_sample, manifest)

    assert io_utils.load_manifest(str_tar) == manifest
    assert io_utils.load_manifest(str(tmp_path / "KinaseInfo")) == manifest

    path_bare = tmp_path / "bare"
    _write_tar(path_bare, dict_sample)
    assert io_utils.load_manifest(str(path_bare / "KinaseInfo.tar.gz")) is None
    assert io_utils.load_manifest(str(path_bare / "KinaseInfo")) is None


def test_manifest_summary(tmp_path, dict_kinase, capsys):
    """The summary nests children under parents and prints from an archive."""
    manifest = io_utils.Manifest.from_kinase_dict(
        dict_kinase,
        git={"sha": "0123456789abcdef", "dirty": True},
        packages={"mkt-schema": "0.1.0"},
    )
    list_lines = manifest.return_summary().splitlines()

    def _row(str_label):
        return next(
            i for i, line in enumerate(list_lines) if line.startswith(str_label)
        )

    assert "  git        0123456789ab (dirty)" in list_lines
    assert "  entries    543" in list_lines
    # extras sit directly under their parent rather than at the end
    assert _row("  klifs ") + 1 == _row("    pocket_seq ")
    assert _row("    pocket_seq ") < _row("  pfam ")
    assert "v1 60 · v3 437" in list_lines[_row("    fasta ")]

    str_tar = _write_tar(tmp_path, {"ABL1": dict_kinase["ABL1"]}, manifest)
    io_utils.print_manifest_summary(str_tar)
    assert "KinaseInfo manifest v1" in capsys.readouterr().out


def test_load_with_matching_manifest(tmp_path, dict_sample, caplog):
    """A consistent archive loads without a missing-manifest warning."""
    caplog.set_level(logging.WARNING)
    manifest = io_utils.Manifest.from_kinase_dict(dict_sample)
    str_tar = _write_tar(tmp_path, dict_sample, manifest)

    dict_loaded = io_utils.deserialize_kinase_dict(str_path=str_tar)
    assert list(dict_loaded) == sorted(dict_sample)
    assert io_utils.STR_MANIFEST_FILENAME not in caplog.text


def test_tampered_count_raises(tmp_path, dict_sample):
    """A count that disagrees with the loaded dict raises with the offending path."""
    manifest = io_utils.Manifest.from_kinase_dict(dict_sample)
    manifest.counts["kincore.cif"] += 1
    str_tar = _write_tar(tmp_path, dict_sample, manifest)

    with pytest.raises(ValueError, match=r"counts\[kincore.cif\]"):
        io_utils.deserialize_kinase_dict(str_path=str_tar)


def test_stale_entry_raises(tmp_path, dict_kinase, dict_sample):
    """An extra file not recorded in the manifest raises on ``n_entries``."""
    manifest = io_utils.Manifest.from_kinase_dict(dict_sample)
    dict_stale = {**dict_sample, "CDK2": dict_kinase["CDK2"]}
    str_tar = _write_tar(tmp_path, dict_stale, manifest)

    with pytest.raises(ValueError, match="n_entries"):
        io_utils.deserialize_kinase_dict(str_path=str_tar)


def test_missing_manifest_warns(tmp_path, dict_sample, caplog):
    """A manifest-less tar still loads, with a warning."""
    caplog.set_level(logging.WARNING)
    str_tar = _write_tar(tmp_path, dict_sample)

    dict_loaded = io_utils.deserialize_kinase_dict(str_path=str_tar)
    assert len(dict_loaded) == len(dict_sample)
    assert f"No {io_utils.STR_MANIFEST_FILENAME}" in caplog.text


def test_list_ids_skips_check(tmp_path, dict_sample):
    """A subset load skips the check even against a mismatched manifest."""
    manifest = io_utils.Manifest.from_kinase_dict(dict_sample)
    manifest.n_entries += 1
    str_tar = _write_tar(tmp_path, dict_sample, manifest)

    dict_loaded = io_utils.deserialize_kinase_dict(str_path=str_tar, list_ids=["ABL1"])
    assert list(dict_loaded) == ["ABL1"]


def test_directory_load_checks_manifest(tmp_path, dict_sample):
    """Directory loads skip the manifest file and check it when present."""
    manifest = io_utils.Manifest.from_kinase_dict(dict_sample)
    path_dir = _write_dir(tmp_path, dict_sample, manifest)
    dict_loaded = io_utils.deserialize_kinase_dict(
        str_path=str(path_dir), bool_remove=False
    )
    assert list(dict_loaded) == sorted(dict_sample)

    manifest.counts["alphafold"] -= 1
    (path_dir / io_utils.STR_MANIFEST_FILENAME).write_text(manifest.model_dump_json())
    with pytest.raises(ValueError, match=r"counts\[alphafold\]"):
        io_utils.deserialize_kinase_dict(str_path=str(path_dir), bool_remove=False)
