import logging
import tarfile

import pytest
from mkt.schema import io_utils, kinase_schema
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
    "pfam": 518,
    "kincore": 497,
    "kincore.fasta": 497,
    "kincore.cif": 437,
    "kincore.cif.sasa": 436,
    "kincore.cif.superposition": 437,
    "kincore.msa": 497,
    "alphafold": 529,
    "alphafold.sasa": 509,
    "alphafold.superposition": 529,
    "exon": 523,
    "KLIFS2UniProtIdx": 519,
    "KLIFS2UniProtSeq": 519,
}
"""dict[str, int]: Expected non-None counts per path in the packaged KinaseInfo.tar.gz."""

DICT_CORPUS_SOURCE_VERSIONS = {
    "kincore.fasta": {"v1": 60, "v3": 437},
    "kincore.cif": {"v2": 437},
    "alphafold": {"v6": 529},
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


def _write_hashed_tar(tmp_path, dict_entries, str_tamper=None):
    """Write a tar whose manifest records entry SHA-256s; optionally edit one file after.

    The edit appends a newline, so the JSON stays valid and the stem still matches:
    only the hash check can catch it.
    """
    path_dir = _write_dir(tmp_path, dict_entries)
    manifest = io_utils.Manifest.from_kinase_dict(
        dict_entries, entry_sha256=io_utils.return_dir_entry_sha256(str(path_dir))
    )
    (path_dir / io_utils.STR_MANIFEST_FILENAME).write_text(manifest.model_dump_json())
    if str_tamper is not None:
        path_file = path_dir / str_tamper
        path_file.write_bytes(path_file.read_bytes() + b"\n")
    path_tar = tmp_path / "KinaseInfo.tar.gz"
    with tarfile.open(path_tar, "w:gz") as tar:
        for path in sorted(path_dir.iterdir()):
            tar.add(path, arcname=path.name)
    return str(path_tar), path_dir


def test_manifest_tallies_match_corpus(dict_kinase):
    """Tallies on the packaged dict agree with the hardcoded ``test_dict_counts``."""
    counts, source_versions, _ = return_manifest_tallies(dict_kinase)
    assert counts == DICT_CORPUS_COUNTS
    assert source_versions == DICT_CORPUS_SOURCE_VERSIONS


def test_packaged_manifest_matches_expected(dict_kinase):
    """The shipped archive's manifest matches the loaded dict and the expected counts."""
    manifest = io_utils.load_manifest(io_utils.return_str_path_from_pkg_data())
    assert manifest is not None
    assert manifest.return_mismatches(dict_kinase) == []
    assert manifest.n_entries == len(dict_kinase)
    assert manifest.counts == DICT_CORPUS_COUNTS
    assert manifest.source_versions == DICT_CORPUS_SOURCE_VERSIONS


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
    assert "KinaseInfo manifest (mkt-schema 0.1.0)" in capsys.readouterr().out


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


def test_missing_entry_raises(tmp_path, dict_sample):
    """An entry dropped from the archive (e.g. a truncated tar) raises on ``n_entries``."""
    manifest = io_utils.Manifest.from_kinase_dict(dict_sample)
    str_tar = _write_tar(tmp_path, {"ABL1": dict_sample["ABL1"]}, manifest)

    with pytest.raises(ValueError, match="n_entries: expected 2, got 1"):
        io_utils.deserialize_kinase_dict(str_path=str_tar)


def test_serialize_rejects_key_hgnc_mismatch(tmp_path, dict_sample):
    """Serializing under a key other than ``hgnc_name`` raises before writing files."""
    path_out = tmp_path / "out"
    with pytest.raises(ValueError, match="dict keys must equal hgnc_name"):
        io_utils.serialize_kinase_dict(
            {"NOT_ABL1": dict_sample["ABL1"]}, str_path=str(path_out)
        )
    assert not path_out.exists()


def test_deserialize_rejects_renamed_file(tmp_path, dict_sample):
    """A file whose name differs from its ``hgnc_name`` raises, from a tar or directory."""
    path_dir = _write_dir(tmp_path, dict_sample)
    (path_dir / "ABL1.json").rename(path_dir / "NOT_ABL1.json")

    path_tar = tmp_path / "renamed.tar.gz"
    with tarfile.open(path_tar, "w:gz") as tar:
        for path in sorted(path_dir.iterdir()):
            tar.add(path, arcname=path.name)
    with pytest.raises(ValueError, match="filenames must match hgnc_name"):
        io_utils.deserialize_kinase_dict(str_path=str(path_tar))

    with pytest.raises(ValueError, match="filenames must match hgnc_name"):
        io_utils.deserialize_kinase_dict(str_path=str(path_dir), bool_remove=False)


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


def test_entry_sha256_roundtrip(tmp_path, dict_sample):
    """An archive whose entries match their recorded SHA-256s loads, full or subset."""
    str_tar, _ = _write_hashed_tar(tmp_path, dict_sample)
    manifest = io_utils.load_manifest(str_tar)
    assert sorted(manifest.entry_sha256) == ["ABL1.json", "BUB1B.json"]
    assert "sha256     2 entries" in manifest.return_summary()

    assert list(io_utils.deserialize_kinase_dict(str_path=str_tar)) == ["ABL1", "BUB1B"]
    assert list(
        io_utils.deserialize_kinase_dict(str_path=str_tar, list_ids=["ABL1"])
    ) == ["ABL1"]


def test_tampered_entry_raises(tmp_path, dict_sample):
    """A byte-level edit that still parses raises on the entry's SHA-256."""
    str_tar, _ = _write_hashed_tar(tmp_path, dict_sample, str_tamper="ABL1.json")

    with pytest.raises(ValueError, match=r"ABL1\.json: SHA-256"):
        io_utils.deserialize_kinase_dict(str_path=str_tar)


def test_subset_load_checks_entry_sha256(tmp_path, dict_sample):
    """Subset loads skip the count check but still verify the entries they read."""
    str_tar, _ = _write_hashed_tar(tmp_path, dict_sample, str_tamper="ABL1.json")

    with pytest.raises(ValueError, match=r"ABL1\.json: SHA-256"):
        io_utils.deserialize_kinase_dict(str_path=str_tar, list_ids=["ABL1"])
    # an untouched entry in the same archive still loads
    assert list(
        io_utils.deserialize_kinase_dict(str_path=str_tar, list_ids=["BUB1B"])
    ) == ["BUB1B"]


def test_directory_load_checks_entry_sha256(tmp_path, dict_sample):
    """Directory loads verify entry SHA-256s too."""
    _, path_dir = _write_hashed_tar(tmp_path, dict_sample, str_tamper="BUB1B.json")

    with pytest.raises(ValueError, match=r"BUB1B\.json: SHA-256"):
        io_utils.deserialize_kinase_dict(str_path=str(path_dir), bool_remove=False)


def test_unlisted_entry_raises(tmp_path, dict_kinase, dict_sample):
    """An entry absent from the manifest's hashes raises, even on a subset load."""
    str_tar, path_dir = _write_hashed_tar(tmp_path, dict_sample)
    io_utils.serialize_kinase_dict(
        {"CDK2": dict_kinase["CDK2"]}, str_path=str(path_dir)
    )
    with tarfile.open(str_tar, "w:gz") as tar:
        for path in sorted(path_dir.iterdir()):
            tar.add(path, arcname=path.name)

    with pytest.raises(ValueError, match="CDK2.json: not listed"):
        io_utils.deserialize_kinase_dict(str_path=str_tar, list_ids=["ABL1", "CDK2"])


def test_manifest_cached_until_archive_changes(tmp_path, dict_sample):
    """A repeat load reuses the parsed manifest; a rebuilt archive is re-read."""
    import os

    manifest = io_utils.Manifest.from_kinase_dict(dict_sample)
    str_tar = _write_tar(tmp_path, dict_sample, manifest)
    first = io_utils.load_manifest(str_tar)
    assert io_utils.load_manifest(str_tar) is first
    io_utils.deserialize_kinase_dict(str_path=str_tar, list_ids=["ABL1"])
    assert io_utils.load_manifest(str_tar) is first

    manifest.counts["kincore"] += 1
    _write_tar(tmp_path, dict_sample, manifest)
    os.utime(str_tar, ns=(0, os.stat(str_tar).st_mtime_ns + 1))
    rebuilt = io_utils.load_manifest(str_tar)
    assert rebuilt is not first
    assert rebuilt.counts["kincore"] == first.counts["kincore"] + 1


def test_manifest_without_hashes_loads(tmp_path, dict_sample):
    """A manifest without entry hashes skips the SHA-256 check."""
    manifest = io_utils.Manifest.from_kinase_dict(dict_sample)
    assert manifest.entry_sha256 == {}
    str_tar = _write_tar(tmp_path, dict_sample, manifest)

    assert len(io_utils.deserialize_kinase_dict(str_path=str_tar)) == 2


def test_source_sha256_tallied_and_checked(tmp_path, mutable_kinase):
    """Provenance SHA-256s are tallied by source name and checked on full loads."""
    abl1 = mutable_kinase("ABL1")
    abl1.kincore.cif.source.sha256 = "a" * 64
    dict_entries = {"ABL1": abl1}

    manifest = io_utils.Manifest.from_kinase_dict(dict_entries)
    str_name = abl1.kincore.cif.source.name
    assert manifest.source_sha256[str_name] == {"a" * 64: 1}

    manifest.source_sha256[str_name] = {"b" * 64: 1}
    str_tar = _write_tar(tmp_path, dict_entries, manifest)
    with pytest.raises(ValueError, match=r"source_sha256\["):
        io_utils.deserialize_kinase_dict(str_path=str_tar)


STR_SOURCE_SHA256 = "f" * 64
"""str: Stand-in SHA-256 for a file source stored only by hash."""


@pytest.fixture
def _empty_sources(monkeypatch):
    """Isolate the module-level sources lookup so registrations don't leak across tests."""
    monkeypatch.setattr(kinase_schema, "_DICT_SOURCES", {})


def _abl1_by_sha256(mutable_kinase):
    """ABL1 whose KinCoRe CIF source is stored by SHA-256, plus the matching table entry."""
    abl1 = mutable_kinase("ABL1")
    full = abl1.kincore.cif.source.model_copy(update={"sha256": STR_SOURCE_SHA256})
    abl1.kincore.cif.source = kinase_schema.Provenance(sha256=STR_SOURCE_SHA256)
    return abl1, {STR_SOURCE_SHA256: full}


def test_provenance_sha256_only_is_compact_and_needs_an_identifier():
    """A hash-only entry serializes as just its SHA-256; an empty one is rejected."""
    from pydantic import ValidationError

    prov = kinase_schema.Provenance(sha256=STR_SOURCE_SHA256)
    assert prov.model_dump() == {"sha256": STR_SOURCE_SHA256}
    assert kinase_schema.Provenance(name="AlphaFold DB").model_dump() == {
        "name": "AlphaFold DB"
    }
    with pytest.raises(ValidationError, match="name or a sha256"):
        kinase_schema.Provenance()


def test_provenance_resolve_and_str(_empty_sources):
    """resolve() looks hash-only entries up (registered or via a manifest); str() prints it."""
    full = kinase_schema.Provenance(
        name="AF2_Active_Models_v2.zip",
        version="v2",
        citation="Gizzio et al., 2026.",
        doi="https://doi.org/10.1042/BCJ20260137",
        query_date="2026-08-26",
        sha256=STR_SOURCE_SHA256,
    )
    prov = kinase_schema.Provenance(sha256=STR_SOURCE_SHA256)
    assert prov.resolve() is None
    assert "unresolved" in str(prov)

    manifest = io_utils.Manifest(
        generated_at="2026-10-06T00:00:00Z",
        n_entries=0,
        counts={},
        sources={STR_SOURCE_SHA256: full},
    )
    assert prov.resolve(manifest) == full

    kinase_schema.register_sources({STR_SOURCE_SHA256: full})
    assert prov.resolve() == full
    assert str(prov) == (
        "AF2_Active_Models_v2.zip (v2) · Gizzio et al., 2026. · "
        "https://doi.org/10.1042/BCJ20260137\n"
        "  queried 2026-08-26 · sha256 ffffffffffff…"
    )
    # inline provenance resolves to itself
    assert full.resolve() is full


def test_sources_table_resolves_after_load(tmp_path, mutable_kinase, _empty_sources):
    """Hash-only record sources resolve after a full or subset load, and tallies use them."""
    abl1, dict_sources = _abl1_by_sha256(mutable_kinase)
    dict_entries = {"ABL1": abl1}
    kinase_schema.register_sources(dict_sources)  # as a build would
    manifest = io_utils.Manifest.from_kinase_dict(dict_entries, sources=dict_sources)
    assert manifest.source_versions["kincore.cif"] == {"v2": 1}
    str_tar = _write_tar(tmp_path, dict_entries, manifest)

    for list_ids in (None, ["ABL1"]):
        kinase_schema._DICT_SOURCES.clear()  # a fresh session knows nothing yet
        loaded = io_utils.deserialize_kinase_dict(str_path=str_tar, list_ids=list_ids)
        source = loaded["ABL1"].kincore.cif.source
        assert source.model_dump() == {"sha256": STR_SOURCE_SHA256}
        assert source.resolve().name == "AF2_Active_Models_v2.zip"


def test_missing_sources_entry_raises(tmp_path, mutable_kinase, _empty_sources):
    """A hash-only record source absent from the sources table raises, subset loads too."""
    abl1, _ = _abl1_by_sha256(mutable_kinase)
    manifest = io_utils.Manifest.from_kinase_dict({"ABL1": abl1})
    str_tar = _write_tar(tmp_path, {"ABL1": abl1}, manifest)

    for list_ids in (None, ["ABL1"]):
        with pytest.raises(ValueError, match="missing from its manifest.json sources"):
            io_utils.deserialize_kinase_dict(str_path=str_tar, list_ids=list_ids)
