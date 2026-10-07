"""Tests for the compositional KinaseInfo build pipeline (:mod:`mkt.databases.generator`).

Covers the deterministic, network-free logic: multi-kinase-domain suffix stripping,
enrichment-step selection/validation (``--only``/``--skip``), HGNC->UniProt target
resolution, and the ``--kinase`` splice round-trip (with the base build stubbed so no
API calls are made and the committed archive is never touched).
"""

import copy
import dataclasses
import logging

import pytest
from mkt.databases.generator import pipeline
from mkt.databases.generator import steps as build_steps
from mkt.databases.io_utils import create_tar_without_metadata
from mkt.databases.plot_config import ArgumentError
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
    list_all = build_steps.return_component_names("step")
    assert build_steps.resolve_step_names() == list_all
    assert build_steps.resolve_step_names(skip=[]) == list_all
    assert "alphafold" not in build_steps.resolve_step_names(skip=["alphafold"])


def test_resolve_step_names_only_and_skip_order(monkeypatch):
    """--only/--skip return steps in registry order regardless of arg order."""
    fake_registry = {
        name: build_steps.Component(name, "step", name, run=lambda ctx: None)
        for name in ("alpha", "beta", "gamma")
    }
    monkeypatch.setattr(build_steps, "COMPONENTS", fake_registry)

    # --only preserves registry order, not the order supplied
    assert build_steps.resolve_step_names(only=["gamma", "alpha"]) == ["alpha", "gamma"]
    # --skip removes named steps, keeps the rest in registry order
    assert build_steps.resolve_step_names(skip=["beta"]) == ["alpha", "gamma"]
    # neither runs all steps
    assert build_steps.resolve_step_names() == ["alpha", "beta", "gamma"]


def _fake_steps(dict_reads):
    """Fake step registry (insertion order kept) with the given ``reads`` per step."""
    return {
        name: build_steps.Component(
            name, "step", name, reads=frozenset(reads), run=lambda ctx: None
        )
        for name, reads in dict_reads.items()
    }


def test_run_order_follows_reads(monkeypatch):
    """A step listed before what it reads still runs after it; ties keep registry order,
    and a step that isn't running still orders the steps that read it."""
    monkeypatch.setattr(
        build_steps,
        "COMPONENTS",
        _fake_steps({"beta": {"alpha"}, "alpha": set(), "gamma": set()}),
    )
    assert build_steps.resolve_step_names() == ["alpha", "beta", "gamma"]
    assert build_steps.resolve_step_names(only=["alpha"]) == ["alpha", "beta"]
    assert build_steps.resolve_step_names(skip=["alpha"]) == ["beta", "gamma"]


def test_run_order_rejects_cycles_and_unknown_reads(monkeypatch):
    """A read cycle or a read of an unregistered name fails before anything runs."""
    from graphlib import CycleError

    monkeypatch.setattr(
        build_steps, "COMPONENTS", _fake_steps({"alpha": {"beta"}, "beta": {"alpha"}})
    )
    with pytest.raises(CycleError):
        build_steps.resolve_step_names()

    monkeypatch.setattr(build_steps, "COMPONENTS", _fake_steps({"alpha": {"typo"}}))
    with pytest.raises(ValueError, match=r"alpha reads \['typo'\]"):
        build_steps.resolve_step_names()


def test_registry_steps_run_after_their_reads():
    """In the real registry, every step runs after each step it reads."""
    list_order = build_steps.resolve_step_names()
    for idx, name in enumerate(list_order):
        set_steps_read = build_steps.COMPONENTS[name].reads & set(list_order)
        assert set_steps_read <= set(list_order[:idx]), name


@pytest.mark.parametrize(
    "only,expected",
    [
        (["kincore"], ["kincore_msa", "kincore_structure_props", "alphafold"]),
        (["klifs"], ["kincore_msa", "kincore_structure_props", "alphafold"]),
        (["pfam"], ["alphafold"]),
        (["hgnc"], ["kincore_msa", "kincore_structure_props", "alphafold", "exon"]),
        (["uniprot"], ["kincore_msa", "kincore_structure_props", "alphafold", "exon"]),
        (["kinhub"], []),
        (["kincore_msa"], ["kincore_msa", "kincore_structure_props", "alphafold"]),
        (["exon"], ["exon"]),
        (["kinhub", "exon"], ["exon"]),
    ],
)
def test_resolve_step_names_adds_downstream(only, expected):
    """--only runs the requested components plus every step downstream of them."""
    assert build_steps.resolve_step_names(only=only) == expected


def test_merge_rebuilt_entries_carries_clears_and_removes():
    """Unrun steps' fields carry over (creating a KinCoRe shell if needed), re-run steps'
    fields are cleared unless the step checks its own inputs, and rebuilt UniProts'
    vanished entries are removed."""
    seed = deserialize_kinase_dict(list_ids=["EGFR", "JAK1_1"], bool_verbose=False)
    if not {"EGFR", "JAK1_1"} <= set(seed):
        pytest.skip("packaged KinaseInfo.tar.gz missing EGFR/JAK1")
    assert seed["EGFR"].kincore.msa is not None and seed["EGFR"].exon is not None

    dict_existing = {k: copy.deepcopy(v) for k, v in seed.items()}
    dict_existing["JAK1_9"] = copy.deepcopy(
        seed["JAK1_1"]
    )  # a domain the rebuild drops
    dict_existing["JAK1_9"].hgnc_name = "JAK1_9"
    egfr = copy.deepcopy(seed["EGFR"])
    egfr.kincore, egfr.exon, egfr.alphafold = None, None, None
    jak1 = copy.deepcopy(seed["JAK1_1"])

    subset_hgnc = pipeline.merge_rebuilt_entries(
        dict_existing,
        {"EGFR": egfr, "JAK1_1": jak1},
        names=["alphafold"],
        set_base_uniprot={"P00533", "P23458"},
    )

    assert subset_hgnc == {"EGFR", "JAK1_1"}
    assert set(dict_existing) == {"EGFR", "JAK1_1"}
    # exon and kincore.msa carried over; the msa needed a KinCoRe shell to land on
    assert dict_existing["EGFR"].exon == seed["EGFR"].exon
    assert dict_existing["EGFR"].kincore.msa == seed["EGFR"].kincore.msa
    assert dict_existing["EGFR"].kincore.fasta is None
    # alphafold re-runs but checks its own inputs, so the stored value is carried over
    assert dict_existing["JAK1_1"].alphafold == seed["JAK1_1"].alphafold

    # kincore_msa has no input check, so a re-run starts it cleared
    dict_existing = {"JAK1_1": copy.deepcopy(seed["JAK1_1"])}
    jak1 = copy.deepcopy(seed["JAK1_1"])
    pipeline.merge_rebuilt_entries(
        dict_existing, {"JAK1_1": jak1}, ["kincore_msa"], {"P23458"}
    )
    assert dict_existing["JAK1_1"].kincore.msa is None


def test_registry_sources_match_source_enum():
    """The registry's sources are exactly the Source enum (the raw-data key type)."""
    from mkt.databases.kinase_schema import Source

    assert build_steps.return_component_names("source") == [s.value for s in Source]


def test_cli_help_lists_every_component():
    """--only/--skip help and the epilog come from the registry, not a hardcoded list."""
    from mkt.databases.cli import generate_kinaseinfo_objects as cli

    for name in build_steps.return_component_names():
        assert name in cli.STR_COMPONENTS
        assert f"{name}: {build_steps.COMPONENTS[name].help}" in cli.STR_EPILOG
    assert cli.STR_STEPS.split(", ") == build_steps.return_component_names("step")


def test_skip_warns_about_downstream_steps(caplog):
    """--skip X names the steps that still run on X's output, and the fields at risk."""
    caplog.set_level(logging.WARNING, logger=build_steps.__name__)
    names = build_steps.resolve_step_names(skip=["kincore_msa"])
    build_steps.warn_skipped_steps(["kincore_msa"], names, "all entries")

    assert (
        "--skip kincore_msa: kincore_structure_props, alphafold read its output and "
        "will run on all entries without a fresh kincore_msa" in caplog.text
    )
    assert "kincore.cif.sasa, kincore.cif.superposition, alphafold" in caplog.text

    caplog.clear()
    build_steps.warn_skipped_steps(["exon"], ["kincore_msa"], "all entries")
    assert caplog.text == ""  # nothing reads exon


def _stub_fetch_from_seed(monkeypatch, module, seed):
    """Make ``module.fetch_source`` return each source's data for the seed entries."""

    def _fetch(source, set_uniprot):
        dict_src = pipeline._reconstruct_dict_obj(seed)[source]
        if set_uniprot is None:
            return dict(dict_src)
        return {k: v for k, v in dict_src.items() if k in set_uniprot}

    monkeypatch.setattr(module, "fetch_source", _fetch)


def test_base_build_subset_restricts_every_source(monkeypatch):
    """--kinase used to crash with KeyError: the discovery sources (KinHub/KLIFS/KinCoRe)
    kept the whole kinome while HGNC/UniProt/Pfam held only the subset."""
    from mkt.databases import kinase_schema as databases_kinase_schema
    from mkt.databases.kinase_schema import (
        combine_kinaseinfo,
        combine_kinaseinfo_kd,
        combine_kinaseinfo_uniprot,
    )

    seed = deserialize_kinase_dict(
        list_ids=["ABL1", "EGFR", "BRAF"], bool_verbose=False
    )
    if set(seed) != {"ABL1", "EGFR", "BRAF"}:
        pytest.skip("packaged KinaseInfo.tar.gz missing ABL1/EGFR/BRAF")
    _stub_fetch_from_seed(monkeypatch, databases_kinase_schema, seed)

    dict_obj = databases_kinase_schema.generate_dict_obj_from_api_or_scraper(
        subset_uniprot={"P00533"}
    )
    assert all(set(dict_src) <= {"P00533"} for dict_src in dict_obj.values())
    dict_out = combine_kinaseinfo(
        combine_kinaseinfo_uniprot(dict_obj), combine_kinaseinfo_kd(dict_obj)
    )
    assert list(dict_out) == ["EGFR"]


def test_only_hgnc_applies_name_fallback(monkeypatch):
    """--only hgnc names an entry HGNC returns no symbol for, as the base build does."""
    seed = deserialize_kinase_dict(list_ids=["ABL1"], bool_verbose=False)
    if "ABL1" not in seed:
        pytest.skip("packaged KinaseInfo.tar.gz missing ABL1")
    _stub_fetch_from_seed(monkeypatch, pipeline, seed)
    # HGNC has no symbol: the client returns the UniProt ID as the name
    stub_fetch = pipeline.fetch_source
    monkeypatch.setattr(
        pipeline,
        "fetch_source",
        lambda source, ids: (
            {"P00519": "P00519"} if source == "hgnc" else stub_fetch(source, ids)
        ),
    )

    dict_new = pipeline.run_source_rebuild(["hgnc"], seed)
    assert list(dict_new) == ["ABL1"]


class _Named:
    """Minimal source record with name attributes for find_alternative_hgnc."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_find_alternative_hgnc_returns_one_name(caplog):
    """Duplicates collapse to one name; disagreeing sources warn and use the first; no
    name raises."""
    from mkt.databases.kinase_schema import find_alternative_hgnc

    kinhub = {"X": [_Named(hgnc_name="SIK1B", xname=None)]}
    kincore = {"X": [_Named(fasta=_Named(hgnc={"SIK1B"}))]}
    assert find_alternative_hgnc("X", kinhub, {}, kincore) == "SIK1B"

    caplog.set_level(logging.WARNING)
    klifs = {"X": [_Named(gene_name="SIK1")]}
    assert find_alternative_hgnc("X", kinhub, klifs, kincore) == "SIK1B"
    assert "sources give different names (SIK1B, SIK1); using SIK1B" in caplog.text

    with pytest.raises(ValueError, match="No HGNC name for Y"):
        find_alternative_hgnc("Y", {}, {}, {})


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


def _write_seed_archive(tmp_path, seed):
    """Write ``seed`` to ``tmp_path/KinaseInfo.tar.gz`` as a build would (manifest, entry
    hashes, sources table)."""
    path_tar = tmp_path / "KinaseInfo.tar.gz"
    pipeline.Pipeline(
        str(tmp_path / "seed"), str(tmp_path / "reports"), str(path_tar)
    )._serialize_and_tar(seed)
    return path_tar


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
    path_tar = _write_seed_archive(tmp_path, seed)

    # stub the base build to return a tweaked EGFR (detectable via the header)
    sentinel = "SENTINEL_SPLICE_TEST"
    egfr = copy.deepcopy(seed["EGFR"])
    egfr.uniprot.header = sentinel

    def _fake_base_build(subset_uniprot=None):
        assert subset_uniprot == {egfr.uniprot_id}
        return {"EGFR": egfr}

    monkeypatch.setattr(pipeline, "run_base_build", _fake_base_build)
    # enrichment steps fetch structures/transcripts; keep the splice test network-free
    monkeypatch.setattr(
        build_steps,
        "COMPONENTS",
        {
            name: component
            for name, component in build_steps.COMPONENTS.items()
            if component.kind == "source"
        },
    )

    # no figures: they would render into the repo's images/ reports dir
    pipeline.run(list_kinase=["EGFR"], path_objects=str(path_objects), bool_figs=False)

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
    # entry hashes are recorded (the reload above already verified them)
    assert sorted(manifest.entry_sha256) == ["ABL1.json", "EGFR.json"]


def test_archive_writes_sources_table(tmp_path, monkeypatch):
    """A SHA-256-only record source is written to the manifest's sources table and
    resolves after reload; an unregistered one fails before anything is written."""
    from mkt.schema import kinase_schema
    from mkt.schema.kinase_schema import Provenance

    monkeypatch.setattr(kinase_schema, "_DICT_SOURCES", {})
    seed = deserialize_kinase_dict(list_ids=["ABL1"], bool_verbose=False)
    abl1 = copy.deepcopy(seed["ABL1"])
    str_sha = "e" * 64
    full = abl1.kincore.cif.source.resolve().model_copy(update={"sha256": str_sha})
    abl1.kincore.cif.source = Provenance(sha256=str_sha)
    pl = pipeline.Pipeline(
        str(tmp_path / "KinaseInfo"),
        str(tmp_path / "reports"),
        str(tmp_path / "KinaseInfo.tar.gz"),
    )

    with pytest.raises(ValueError, match="no registered provenance"):
        pl._serialize_and_tar({"ABL1": abl1})
    assert not (tmp_path / "KinaseInfo.tar.gz").exists()

    kinase_schema.register_sources({str_sha: full})
    pl._serialize_and_tar({"ABL1": abl1})
    assert load_manifest(pl.path_tar).sources[str_sha] == full

    kinase_schema._DICT_SOURCES.clear()  # a fresh session
    after = deserialize_kinase_dict(str_path=pl.path_tar, bool_verbose=False)
    assert after["ABL1"].kincore.cif.source.resolve() == full


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


def test_interrupted_build_leaves_no_staging(tmp_path, monkeypatch):
    """A build that fails mid-archive leaves no staging files and keeps the old archive."""
    import tempfile

    seed = deserialize_kinase_dict(list_ids=["ABL1"], bool_verbose=False)
    path_tmp = tmp_path / "tmp"
    path_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(path_tmp))
    path_tar = tmp_path / "KinaseInfo.tar.gz"
    path_tar.write_bytes(b"previous archive")

    def _fail_tar(path_source, filename_tar):
        # fail after a partial archive has been started
        with open(filename_tar, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(pipeline, "create_tar_without_metadata", _fail_tar)
    pl = pipeline.Pipeline(
        str(tmp_path / "KinaseInfo"),
        str(tmp_path / "reports"),
        str(tmp_path / "KinaseInfo.tar.gz"),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        pl._serialize_and_tar(seed)

    assert not (tmp_path / "KinaseInfo").exists()
    assert list(path_tmp.iterdir()) == []
    assert path_tar.read_bytes() == b"previous archive"
    assert not (tmp_path / "KinaseInfo.tar.gz.partial").exists()


def test_dated_reports_dir_requires_manifest(tmp_path):
    """A manifest-less archive has no version to name a reports folder by, so it raises."""
    seed = deserialize_kinase_dict(list_ids=["ABL1"], bool_verbose=False)
    path_seed = tmp_path / "seed"
    path_tar = tmp_path / "KinaseInfo.tar.gz"
    serialize_kinase_dict(seed, str_path=str(path_seed))
    create_tar_without_metadata(path_source=str(path_seed), filename_tar=str(path_tar))

    pl = pipeline.Pipeline(str(path_seed), str(tmp_path / "reports"), str(path_tar))
    with pytest.raises(ArgumentError, match="rebuild the archive with --data"):
        pl._dated_reports_dir()
    assert not (tmp_path / "reports").exists()


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


def test_reconstruct_dict_obj_keeps_later_domain_pfam():
    """A Pfam hit dropped from ``_1`` but kept on ``_2`` still reaches the rebuild."""
    seed = deserialize_kinase_dict(list_ids=["JAK1_1", "JAK1_2"], bool_verbose=False)
    if not {"JAK1_1", "JAK1_2"} <= set(seed):
        pytest.skip("packaged KinaseInfo.tar.gz missing JAK1")

    seed = {k: copy.deepcopy(seed[k]) for k in ["JAK1_1", "JAK1_2"]}
    seed["JAK1_1"].pfam = None
    assert seed["JAK1_2"].pfam is not None
    dict_obj = pipeline._reconstruct_dict_obj(seed)
    assert dict_obj["pfam"]["P23458"] is seed["JAK1_2"].pfam


def test_drop_nonintersecting_pfam():
    """Pfam survives if it overlaps KinCoRe/KLIFS, or is the sole source of a single domain."""
    from mkt.databases.kinase_schema import drop_nonintersecting_pfam

    seed = deserialize_kinase_dict(list_ids=["EGFR", "JAK1_1"], bool_verbose=False)
    if not {"EGFR", "JAK1_1"} <= set(seed):
        pytest.skip("packaged KinaseInfo.tar.gz missing EGFR/JAK1")
    pfam = seed["EGFR"].pfam

    def _kept(hgnc_name, start, end, bool_strip=False):
        obj = copy.deepcopy(seed[hgnc_name])
        obj.pfam = pfam.model_copy(update={"start": start, "end": end})
        if bool_strip:
            obj.kincore, obj.KLIFS2UniProtIdx = None, None
        return drop_nonintersecting_pfam(obj).pfam is not None

    assert _kept("EGFR", pfam.start, pfam.end)  # overlaps KinCoRe/KLIFS
    assert not _kept("EGFR", 1, 10)  # single-domain, disjoint
    assert _kept("EGFR", 1, 10, bool_strip=True)  # single-domain, Pfam only
    assert not _kept("JAK1_1", 1, 10)  # multi-domain, disjoint
    assert not _kept("JAK1_1", 1, 10, bool_strip=True)  # multi-domain, Pfam only


def test_run_dispatches_source_only(monkeypatch, tmp_path):
    """--only <source> routes to the source-rebuild path, not full regen / per-entry."""
    calls = {}
    monkeypatch.setattr(
        pipeline, "_resolve_dir", lambda repo, rel, default: str(tmp_path)
    )
    monkeypatch.setattr(
        pipeline.Pipeline,
        "partial",
        lambda self, sources, names, **kwargs: calls.__setitem__(
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


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"only": ["exon"], "skip": ["alphafold"]}, "mutually exclusive"),
        ({"only": ["kincore"], "skip": ["alphafold"]}, "mutually exclusive"),
        ({"only": ["notacomponent"]}, r"unknown --only .*'hgnc'.*'exon'"),
        ({"skip": ["klifs"]}, "unknown --skip"),
    ],
)
def test_run_rejects_invalid_only_skip(tmp_path, monkeypatch, kwargs, match):
    """--only/--skip are validated once, before any work, against every valid component."""
    pl, calls = _data_pipeline(tmp_path, monkeypatch)
    with pytest.raises(ArgumentError, match=match):
        pl.run(**kwargs)
    assert calls == []


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


def _seeded_pipeline(tmp_path, monkeypatch, list_ids):
    """Pipeline over a seed archive of real packaged entries, with no network access.

    ``fetch_source`` returns each source's existing data for the seed entries, and every
    enrichment step is replaced by a recorder (registry order kept) that logs
    ``(step, targeted hgnc names)`` without touching the objects.

    Returns
    -------
    tuple[Pipeline, dict, list]
        The pipeline, the seed dict, and the step-call log.
    """
    seed = deserialize_kinase_dict(list_ids=list_ids, bool_verbose=False)
    if set(seed) != set(list_ids):
        pytest.skip(f"packaged KinaseInfo.tar.gz missing {set(list_ids) - set(seed)}")

    path_tar = _write_seed_archive(tmp_path, seed)

    def _fetch_existing(source, set_uniprot):
        dict_src = pipeline._reconstruct_dict_obj(seed)[source]
        return {k: v for k, v in dict_src.items() if k in set_uniprot}

    monkeypatch.setattr(pipeline, "fetch_source", _fetch_existing)

    calls = []

    def _recorder(name):
        def _step(ctx):
            set_hgnc = (
                set(ctx.dict_kinaseinfo)
                if ctx.subset_hgnc is None
                else set(ctx.subset_hgnc)
            )
            calls.append((name, set_hgnc))

        return _step

    monkeypatch.setattr(
        build_steps,
        "COMPONENTS",
        {
            name: (
                dataclasses.replace(component, run=_recorder(name))
                if component.kind == "step"
                else component
            )
            for name, component in build_steps.COMPONENTS.items()
        },
    )
    pl = pipeline.Pipeline(
        str(tmp_path / "KinaseInfo"), str(tmp_path / "reports"), str(path_tar)
    )
    return pl, seed, calls


def _assert_exon_preserved(path_tar, seed):
    """Every seed entry's exon map survives the rebuild unchanged."""
    after = deserialize_kinase_dict(str_path=str(path_tar), bool_verbose=False)
    assert set(after) == set(seed)
    for hgnc_name, obj in seed.items():
        assert after[hgnc_name].exon == obj.exon, hgnc_name


def test_source_rebuild_runs_downstream_and_keeps_unrelated_fields(
    tmp_path, monkeypatch
):
    """Bug 1: ``--only kincore --only kincore_msa`` dropped every entry's ``exon``.

    A source rebuild now runs the requested steps plus every step downstream of the
    source, and carries over the fields of steps that don't depend on it.
    """
    pl, seed, calls = _seeded_pipeline(tmp_path, monkeypatch, ["ABL1", "EGFR"])
    assert all(obj.exon is not None for obj in seed.values())

    pl.run(only=["kincore", "kincore_msa"], bool_figs=False)

    assert [name for name, _ in calls] == [
        "kincore_msa",
        "kincore_structure_props",
        "alphafold",
    ]
    _assert_exon_preserved(pl.path_tar, seed)


def test_non_kincore_source_rebuild_handles_missing_kincore(tmp_path, monkeypatch):
    """Bug 2: rebuilding a non-kincore source crashed on a domain without KinCoRe (TEX14_2)."""
    list_ids = ["ABL1", "EGFR", "TEX14_1", "TEX14_2"]
    pl, seed, calls = _seeded_pipeline(tmp_path, monkeypatch, list_ids)
    assert seed["TEX14_2"].kincore is None

    pl.run(only=["klifs"], bool_figs=False)

    # klifs feeds the KLIFS mapping, so everything but exon re-runs
    assert [name for name, _ in calls] == [
        "kincore_msa",
        "kincore_structure_props",
        "alphafold",
    ]
    _assert_exon_preserved(pl.path_tar, seed)


def test_source_rebuild_runs_every_step_on_new_entries(tmp_path, monkeypatch):
    """An entry the rebuild produces with no existing counterpart (e.g. a domain renamed by
    a KinHub refresh) gets every step; existing entries get only the downstream ones."""
    pl, seed, calls = _seeded_pipeline(tmp_path, monkeypatch, ["ABL1", "EGFR"])
    abl1_renamed = copy.deepcopy(seed["ABL1"])
    abl1_renamed.hgnc_name = "ABL1_RENAMED"
    abl1_renamed.exon = None

    def _fake_rebuild(sources, dict_existing, subset_uniprot=None):
        return {
            "EGFR": copy.deepcopy(seed["EGFR"]),
            "ABL1_RENAMED": abl1_renamed,
        }

    monkeypatch.setattr(pipeline, "run_source_rebuild", _fake_rebuild)
    pl.run(only=["kinhub"], bool_figs=False)

    # kinhub has no downstream steps, so only the new entry is enriched, by every step
    assert calls == [
        (name, {"ABL1_RENAMED"}) for name in build_steps.return_component_names("step")
    ]
    after = deserialize_kinase_dict(str_path=str(pl.path_tar), bool_verbose=False)
    assert set(after) == {"EGFR", "ABL1_RENAMED"}  # the old ABL1 key is gone
    assert after["EGFR"].exon == seed["EGFR"].exon


def test_only_and_kinase_compose(tmp_path, monkeypatch):
    """Bug 3: ``--only exon --kinase EGFR`` ran exon on the whole kinome."""
    pl, _, calls = _seeded_pipeline(tmp_path, monkeypatch, ["ABL1", "EGFR"])

    pl.run(only=["exon"], list_kinase=["EGFR"], bool_figs=False)

    assert calls == [("exon", {"EGFR"})]


def test_kinase_splice_keeps_skipped_step_fields(tmp_path, monkeypatch):
    """``--kinase EGFR --skip exon`` rebuilds EGFR without wiping its existing exon map."""
    pl, seed, calls = _seeded_pipeline(tmp_path, monkeypatch, ["ABL1", "EGFR"])
    fresh = copy.deepcopy(seed["EGFR"])
    fresh.exon = None  # a fresh base build has no enrichment fields

    monkeypatch.setattr(
        pipeline, "run_base_build", lambda subset_uniprot=None: {"EGFR": fresh}
    )
    pl.run(list_kinase=["EGFR"], skip=["exon"], bool_figs=False)

    assert "exon" not in [name for name, _ in calls]
    _assert_exon_preserved(pl.path_tar, seed)


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
        with pytest.raises(ArgumentError, match="pass --data"):
            pl.run(**kwargs)
    assert calls == []


def test_run_no_data_no_figs_raises(tmp_path, monkeypatch):
    """Turning off both data and figures is an error, not a silent no-op."""
    pl, calls = _data_pipeline(tmp_path, monkeypatch, STR_YAML_NO_DATA)
    with pytest.raises(ArgumentError, match="nothing to do"):
        pl.run(bool_figs=False)
    assert calls == []
