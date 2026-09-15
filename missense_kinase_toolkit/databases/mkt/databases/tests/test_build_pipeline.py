"""Tests for the compositional KinaseInfo build pipeline (:mod:`mkt.databases.generator`).

Covers the deterministic, network-free logic: multi-kinase-domain suffix stripping,
enrichment-step selection/validation (``--only``/``--skip``), HGNC->UniProt target
resolution, and the ``--kinase`` splice round-trip (with the base build stubbed so no
API calls are made and the committed archive is never touched).
"""

import copy

import pytest
from mkt.databases.generator import pipeline
from mkt.databases.generator import steps as build_steps
from mkt.databases.io_utils import create_tar_without_metadata
from mkt.schema.io_utils import (
    deserialize_kinase_dict,
    load_manifest,
    serialize_kinase_dict,
)


@pytest.mark.parametrize(
    "str_in,expected",
    [
        ("EGFR", "EGFR"),
        ("JAK1_1", "JAK1"),
        ("JAK1_2", "JAK1"),
        ("O60674_2", "O60674"),
        ("SGK223", "SGK223"),  # trailing digits without underscore are kept
        ("A_B_1", "A_B"),
    ],
)
def test_strip_kd_suffix(str_in, expected):
    assert pipeline._strip_kd_suffix(str_in) == expected


def test_resolve_step_names_defaults_run_all():
    """A full regen runs every enrichment step unless skipped."""
    list_all = list(build_steps._ENRICH_STEPS)
    assert build_steps.resolve_step_names() == list_all
    assert build_steps.resolve_step_names(skip=[]) == list_all
    assert "alphafold" not in build_steps.resolve_step_names(skip=["alphafold"])


def test_resolve_step_names_unknown_raises():
    with pytest.raises(ValueError, match="unknown enrichment step"):
        build_steps.resolve_step_names(only=["does_not_exist"])


def test_resolve_step_names_mutual_exclusion_raises():
    with pytest.raises(ValueError, match="not both"):
        build_steps.resolve_step_names(only=["a"], skip=["b"])


def test_resolve_step_names_only_and_skip_order(monkeypatch):
    """--only/--skip return steps in registry order regardless of arg order."""
    fake_registry = {name: (lambda ctx: None) for name in ("alpha", "beta", "gamma")}
    monkeypatch.setattr(build_steps, "_ENRICH_STEPS", fake_registry)

    # --only preserves registry order, not the order supplied
    assert build_steps.resolve_step_names(only=["gamma", "alpha"]) == ["alpha", "gamma"]
    # --skip removes named steps, keeps the rest in registry order
    assert build_steps.resolve_step_names(skip=["beta"]) == ["alpha", "gamma"]
    # neither runs all steps
    assert build_steps.resolve_step_names() == ["alpha", "beta", "gamma"]


class _FakeKinase:
    """Minimal stand-in exposing the ``uniprot_id`` attribute used for resolution."""

    def __init__(self, uniprot_id):
        self.uniprot_id = uniprot_id


def test_resolve_targets_matches_base_and_suffixed():
    dict_existing = {
        "EGFR": _FakeKinase("P00533"),
        "JAK1_1": _FakeKinase("P23458_1"),
        "JAK1_2": _FakeKinase("P23458_2"),
    }
    # a base HGNC name resolves all its multi-domain variants to one base UniProt id
    subset, unresolved = pipeline._resolve_targets(["EGFR", "JAK1"], dict_existing)
    assert subset == {"P00533", "P23458"}
    assert unresolved == set()


def test_resolve_targets_reports_unresolved():
    dict_existing = {"EGFR": _FakeKinase("P00533")}
    subset, unresolved = pipeline._resolve_targets(
        ["EGFR", "NOTAKINASE"], dict_existing
    )
    assert subset == {"P00533"}
    assert unresolved == {"NOTAKINASE"}


def test_run_update_splices_targeted_entry(tmp_path, monkeypatch):
    """--kinase rebuilds only the targeted entry and splices it into the archive.

    The base build is stubbed (no network); a seed archive is built from two real
    packaged entries in ``tmp_path`` so the committed KinaseInfo.tar.gz is untouched.
    """
    # two real entries read in-memory from the packaged tar (subset read, no network)
    seed = deserialize_kinase_dict(list_ids=["EGFR", "ABL1"], bool_verbose=False)
    if "EGFR" not in seed or "ABL1" not in seed:
        pytest.skip("packaged KinaseInfo.tar.gz missing EGFR/ABL1")

    # build the seed archive at the location the pipeline will read/write
    path_objects = tmp_path / "KinaseInfo"
    path_tar = tmp_path / "KinaseInfo.tar.gz"
    path_seed = tmp_path / "seed"
    serialize_kinase_dict(seed, str_path=str(path_seed))
    create_tar_without_metadata(path_source=str(path_seed), filename_tar=str(path_tar))

    # stub the base build to return a tweaked EGFR (detectable via the header)
    sentinel = "SENTINEL_SPLICE_TEST"
    egfr = copy.deepcopy(seed["EGFR"])
    egfr.uniprot.header = sentinel

    def _fake_base_build(subset_uniprot=None):
        assert subset_uniprot == {egfr.uniprot_id}
        return {"EGFR": egfr}

    monkeypatch.setattr(pipeline, "run_base_build", _fake_base_build)
    # enrichment steps fetch structures/transcripts; keep the splice test network-free
    monkeypatch.setattr(build_steps, "_ENRICH_STEPS", {})

    pipeline.run(list_kinase=["EGFR"], path_objects=str(path_objects))

    after = deserialize_kinase_dict(str_path=str(path_tar), bool_verbose=False)
    # targeted entry updated, non-target untouched, count stable, objects dir cleaned
    assert len(after) == len(seed)
    assert after["EGFR"].uniprot.header == sentinel
    assert after["ABL1"].uniprot.header == seed["ABL1"].uniprot.header
    assert not path_objects.exists()

    # the rebuilt archive carries a manifest consistent with its contents
    manifest = load_manifest(str(path_tar))
    assert manifest is not None
    assert manifest.return_mismatches(after) == []
    assert set(manifest.packages) == set(pipeline.LIST_MANIFEST_PACKAGES)


def test_dated_reports_dir_uses_manifest(tmp_path):
    """The reports subdir is named by ``generated_at``, independent of the tar mtime."""
    import os

    seed = deserialize_kinase_dict(list_ids=["ABL1"], bool_verbose=False)
    pl = pipeline.Pipeline(
        str(tmp_path / "KinaseInfo"),
        str(tmp_path / "reports"),
        str(tmp_path / "KinaseInfo.tar.gz"),
    )
    pl._serialize_and_tar(seed)
    stamp = load_manifest(pl.path_tar).generated_at.strftime(
        pipeline.DATETIME_SUBDIR_FMT
    )

    path_before = pl._dated_reports_dir()
    os.utime(pl.path_tar, (0, 0))  # a fresh checkout changes the mtime
    assert pl._dated_reports_dir() == path_before
    assert os.path.basename(path_before) == stamp


def test_dated_reports_dir_falls_back_to_mtime(tmp_path, caplog):
    """A manifest-less archive names the subdir from the tar mtime, with a warning."""
    import os
    from datetime import datetime

    seed = deserialize_kinase_dict(list_ids=["ABL1"], bool_verbose=False)
    path_seed = tmp_path / "seed"
    path_tar = tmp_path / "KinaseInfo.tar.gz"
    serialize_kinase_dict(seed, str_path=str(path_seed))
    create_tar_without_metadata(path_source=str(path_seed), filename_tar=str(path_tar))

    pl = pipeline.Pipeline(str(path_seed), str(tmp_path / "reports"), str(path_tar))
    stamp = datetime.fromtimestamp(os.path.getmtime(path_tar)).strftime(
        pipeline.DATETIME_SUBDIR_FMT
    )
    assert os.path.basename(pl._dated_reports_dir()) == stamp
    assert "naming reports subdir from the tar mtime" in caplog.text


def test_reconstruct_dict_obj_groups_multidomain():
    """The raw dict_obj is keyed by Source, single-valued for hgnc/uniprot/pfam, and
    lists for kinhub/klifs/kincore with multi-domain entries grouped by base UniProt."""
    from mkt.databases.kinase_schema import Source

    seed = deserialize_kinase_dict(
        list_ids=["EGFR", "JAK1_1", "JAK1_2"], bool_verbose=False
    )
    if not {"EGFR", "JAK1_1", "JAK1_2"} <= set(seed):
        pytest.skip("packaged KinaseInfo.tar.gz missing EGFR/JAK1")

    dict_obj = pipeline._reconstruct_dict_obj(seed)
    assert set(dict_obj) == {source.value for source in Source}
    # single-valued sources
    assert dict_obj["hgnc"]["P00533"] == "EGFR"
    assert not isinstance(dict_obj["uniprot"]["P00533"], list)
    # list sources; single-domain -> length 1, multi-domain grouped to base UniProt
    assert isinstance(dict_obj["kinhub"]["P00533"], list)
    assert len(dict_obj["kinhub"]["P00533"]) == 1
    assert len(dict_obj["kincore"]["P23458"]) == 2  # JAK1 two kinase domains


def test_run_dispatches_source_only(monkeypatch, tmp_path):
    """--only <source> routes to the source-rebuild path, not full regen / per-entry."""
    calls = {}
    monkeypatch.setattr(
        pipeline, "_resolve_dir", lambda repo, rel, default: str(tmp_path)
    )
    monkeypatch.setattr(
        pipeline.Pipeline,
        "partial",
        lambda self, sources, names, bool_figs=True, force=False: calls.__setitem__(
            "partial", (sources, names)
        ),
    )
    monkeypatch.setattr(
        pipeline.Pipeline,
        "full",
        lambda self, names, bool_figs=True, force=False: calls.__setitem__(
            "full", names
        ),
    )
    monkeypatch.setattr(
        pipeline.Pipeline,
        "update",
        lambda self, names, list_kinase, bool_figs=True, force=False: calls.__setitem__(
            "update", (names, list_kinase)
        ),
    )

    pipeline.run(only=["kincore"])
    assert set(calls) == {"partial"}
    assert calls["partial"][0] == ["kincore"]


def test_run_source_with_skip_raises(monkeypatch, tmp_path):
    """--only <source> combined with --skip is rejected."""
    monkeypatch.setattr(pipeline, "_resolve_dir", lambda *a: str(tmp_path))
    with pytest.raises(ValueError, match="skip"):
        pipeline.run(only=["kincore"], skip=["alphafold"])


def test_fetch_source_unknown_raises():
    """fetch_source rejects a name outside the Source enum before any fetch."""
    from mkt.databases.kinase_schema import fetch_source

    with pytest.raises(ValueError):
        fetch_source("notasource", set())


def test_source_only_no_dict_falls_back_to_full(monkeypatch, tmp_path):
    """With no existing dict, --only <source> falls back to a full regen."""
    monkeypatch.setattr(pipeline, "deserialize_kinase_dict", lambda **k: {})
    calls = {}
    monkeypatch.setattr(
        pipeline.Pipeline,
        "full",
        lambda self, names, bool_figs=True, force=False: calls.__setitem__(
            "full", names
        ),
    )

    pl = pipeline.Pipeline(
        str(tmp_path / "objects"),
        str(tmp_path / "reports"),
        str(tmp_path / "absent.tar.gz"),
    )
    pl.partial(["kincore"], [])
    assert "full" in calls


def _data_pipeline(tmp_path, monkeypatch, str_yaml=None):
    """Pipeline with an optional study YAML and stubbed figures/full run modes."""
    config_path = None
    if str_yaml is not None:
        path_config = tmp_path / "study.yaml"
        path_config.write_text(str_yaml)
        config_path = str(path_config)
    calls = []
    monkeypatch.setattr(
        pipeline.Pipeline, "figures", lambda self: calls.append("figures")
    )
    monkeypatch.setattr(
        pipeline.Pipeline,
        "full",
        lambda self, names, bool_figs=True, force=False: calls.append(
            ("full", bool_figs)
        ),
    )
    pl = pipeline.Pipeline(
        str(tmp_path / "objects"),
        str(tmp_path / "reports"),
        str(tmp_path / "absent.tar.gz"),
        config_path=config_path,
    )
    return pl, calls


STR_YAML_NO_DATA = "kinaseinfo:\n  data: false\n"
"""str: Study YAML whose kinaseinfo task draws figures from the existing archive."""


@pytest.mark.parametrize(
    "str_yaml,kwargs,expected",
    [
        (None, {}, [("full", True)]),
        (None, {"bool_figs": False}, [("full", False)]),
        (None, {"bool_data": False}, ["figures"]),
        (STR_YAML_NO_DATA, {}, ["figures"]),
        (STR_YAML_NO_DATA, {"bool_data": True}, [("full", True)]),
        ("kinaseinfo:\n  data: true\n", {"bool_data": False}, ["figures"]),
    ],
)
def test_run_data_figs_resolution(tmp_path, monkeypatch, str_yaml, kwargs, expected):
    """An explicit ``bool_data`` wins over ``kinaseinfo.data``; figures follow ``bool_figs``."""
    pl, calls = _data_pipeline(tmp_path, monkeypatch, str_yaml)
    pl.run(**kwargs)
    assert calls == expected


def test_run_no_data_rejects_rebuild_selectors(tmp_path, monkeypatch):
    """``--only``/``--skip``/``--kinase`` with data off point the user at ``--data``."""
    pl, calls = _data_pipeline(tmp_path, monkeypatch, STR_YAML_NO_DATA)
    for kwargs in ({"only": ["exon"]}, {"skip": ["exon"]}, {"list_kinase": ["ABL1"]}):
        with pytest.raises(ValueError, match="pass --data"):
            pl.run(**kwargs)
    assert calls == []


def test_run_no_data_no_figs_raises(tmp_path, monkeypatch):
    """Turning off both data and figures is an error, not a silent no-op."""
    pl, calls = _data_pipeline(tmp_path, monkeypatch, STR_YAML_NO_DATA)
    with pytest.raises(ValueError, match="nothing to do"):
        pl.run(bool_figs=False)
    assert calls == []
