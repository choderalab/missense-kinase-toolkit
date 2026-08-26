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
from mkt.schema.io_utils import deserialize_kinase_dict, serialize_kinase_dict


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


def test_resolve_step_names_empty_registry():
    assert build_steps.resolve_step_names() == []
    assert build_steps.resolve_step_names(skip=[]) == []


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
    monkeypatch.setattr(build_steps, "_DEFAULT_STEPS", list(fake_registry))

    # --only preserves registry order, not the order supplied
    assert build_steps.resolve_step_names(only=["gamma", "alpha"]) == ["alpha", "gamma"]
    # --skip removes named steps, keeps the rest in registry order
    assert build_steps.resolve_step_names(skip=["beta"]) == ["alpha", "gamma"]
    # neither runs all default-on steps
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

    pipeline.run(list_kinase=["EGFR"], path_objects=str(path_objects))

    after = deserialize_kinase_dict(str_path=str(path_tar), bool_verbose=False)
    # targeted entry updated, non-target untouched, count stable, objects dir cleaned
    assert len(after) == len(seed)
    assert after["EGFR"].uniprot.header == sentinel
    assert after["ABL1"].uniprot.header == seed["ABL1"].uniprot.header
    assert not path_objects.exists()


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
        pipeline, "_run_source_only", lambda *a: calls.__setitem__("source", a)
    )
    monkeypatch.setattr(pipeline, "_run_full", lambda *a: calls.__setitem__("full", a))
    monkeypatch.setattr(
        pipeline, "_run_update", lambda *a: calls.__setitem__("update", a)
    )

    pipeline.run(only=["kincore"])
    assert set(calls) == {"source"}
    assert calls["source"][0] == ["kincore"]


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
    monkeypatch.setattr(pipeline, "_run_full", lambda *a: calls.__setitem__("full", a))

    pipeline._run_source_only(
        ["kincore"],
        [],
        str(tmp_path / "objects"),
        str(tmp_path / "reports"),
        str(tmp_path / "absent.tar.gz"),
    )
    assert "full" in calls
