"""Derived values are recomputed when their inputs change, and only then.

Each test computes a value once, checks that an unchanged rerun skips it, then changes one
input and checks that the value is recomputed. The expensive computation (SASA, the
superposition, the AlphaFold fetch) is replaced by a recorder, so the tests are offline.
"""

import ast
import copy
import logging

import pytest
from mkt.databases import alphafold, input_check, sasa, superpose
from mkt.databases.utils import convert_mmcifdict2structure
from mkt.schema.io_utils import deserialize_kinase_dict

STR_SRC = '''
def f(x):
    """Docstring."""
    return x + 1  # comment
'''
"""str: Small source for code-hash tests."""


def _tree_sha256(str_src):
    """Code hash of a source string, as return_code_sha256 computes it for an object."""
    tree = input_check._strip_docstrings(ast.parse(str_src))
    return input_check.return_sha256(ast.dump(tree).encode("utf-8"))


def test_code_hash_ignores_comments_docstrings_and_formatting():
    """Only logic changes alter the code hash."""
    str_base = _tree_sha256(STR_SRC)
    assert _tree_sha256(STR_SRC.replace("# comment", "# other")) == str_base
    assert _tree_sha256(STR_SRC.replace("Docstring.", "Edited.")) == str_base
    assert _tree_sha256(STR_SRC.replace("x + 1", "(x\n        + 1)")) == str_base
    assert _tree_sha256(STR_SRC.replace("x + 1", "x + 2")) != str_base


def test_code_hash_names_modules_and_functions():
    """Module and cross-module helper hashes are keyed by their qualified names."""
    dict_module = input_check.return_code_sha256(sasa)
    dict_function = input_check.return_code_sha256(convert_mmcifdict2structure)
    assert list(dict_module) == ["code:mkt.databases.sasa"]
    assert list(dict_function) == [
        "code:mkt.databases.utils.convert_mmcifdict2structure"
    ]
    assert all(
        len(str_sha) == 64 for str_sha in {**dict_module, **dict_function}.values()
    )


def test_input_check_reports_what_changed(caplog):
    """Changed hashes and readable fields are named in the result and the log."""
    check = input_check.InputCheck(
        "ABL1 kincore SASA",
        {"structure": "a" * 64, "klifs_map": "b" * 64},
        {"probe_radius": 1.4},
    )
    assert check.return_changed(None) == ["no recorded inputs"]
    assert check.return_changed(check.dict_sha256, {"probe_radius": 1.4}) == []
    assert check.return_changed(
        {"structure": "a" * 64, "klifs_map": "c" * 64}, {"probe_radius": 1.2}
    ) == ["klifs_map", "probe_radius"]

    class _Stored:
        input_sha256 = {"structure": "a" * 64, "klifs_map": "c" * 64}
        probe_radius = 1.4

    caplog.set_level(logging.INFO, logger=input_check.__name__)
    assert check.is_stale(_Stored())
    assert "ABL1 kincore SASA stale: klifs_map changed; recomputing" in caplog.text
    assert not check.is_stale(
        type("_Same", (), {"input_sha256": check.dict_sha256, "probe_radius": 1.4})()
    )
    assert check.is_stale(None) and check.is_stale(_Stored(), force=True)


@pytest.fixture(scope="module")
def dict_seed():
    """ABL1 (KinCoRe CIF + AlphaFold) and INSR (the superposition reference kinase)."""
    seed = deserialize_kinase_dict(list_ids=["ABL1", "INSR"], bool_verbose=False)
    if not {"ABL1", "INSR"} <= set(seed):
        pytest.skip("packaged KinaseInfo.tar.gz missing ABL1/INSR")
    return seed


def _shift_one_klifs_index(obj):
    """Point one mapped KLIFS pocket position at the next UniProt residue."""
    label = next(k for k, v in obj.KLIFS2UniProtIdx.items() if v is not None)
    obj.KLIFS2UniProtIdx[label] += 1


def test_sasa_recomputed_when_klifs_mapping_changes(dict_seed, monkeypatch):
    """SASA is keyed on KLIFS2UniProtIdx, so a changed mapping must recompute it."""
    abl1 = copy.deepcopy(dict_seed["ABL1"])
    calls = []

    def _record(task):
        calls.append(task[0])
        return task[0], {}

    monkeypatch.setattr(sasa, "_sasa_pool_worker", _record)

    sasa.enrich_kinases_with_sasa({"ABL1": abl1}, only="kincore")
    calls.clear()
    sasa.enrich_kinases_with_sasa({"ABL1": abl1}, only="kincore")
    assert calls == [], "unchanged inputs should not recompute"

    _shift_one_klifs_index(abl1)
    sasa.enrich_kinases_with_sasa({"ABL1": abl1}, only="kincore")
    assert calls == ["ABL1::0"]


def test_superposition_recomputed_when_mapping_changes(dict_seed, monkeypatch):
    """The superposition matches residues via KLIFS2UniProtIdx, so a change must redo it."""
    abl1 = copy.deepcopy(dict_seed["ABL1"])
    frame = superpose.build_reference_frame(dict_seed)
    sentinel = copy.deepcopy(abl1.kincore.cif.superposition)
    calls = []

    def _record(dict_cif, obj_kinase, frame, structure_id):
        calls.append(structure_id)
        return copy.deepcopy(sentinel)

    monkeypatch.setattr(superpose, "_superpose_structure", _record)

    superpose.superpose_structure(abl1.kincore.cif, abl1, frame, "ABL1_kincore")
    calls.clear()
    superpose.superpose_structure(abl1.kincore.cif, abl1, frame, "ABL1_kincore")
    assert calls == [], "unchanged inputs should not recompute"

    _shift_one_klifs_index(abl1)
    superpose.superpose_structure(abl1.kincore.cif, abl1, frame, "ABL1_kincore")
    assert calls == ["ABL1_kincore"]


def test_alphafold_refetched_when_canonical_sequence_changes(dict_seed, monkeypatch):
    """The AF slice is checked against the canonical sequence, so a change must refetch it,
    even when the kinase-domain bounds are unchanged."""
    abl1 = copy.deepcopy(dict_seed["ABL1"])
    stored = copy.deepcopy(abl1.alphafold)
    calls = []

    def _record(uniprot_id, start, end, canonical_seq=None):
        calls.append((uniprot_id, start, end))
        return copy.deepcopy(stored)

    monkeypatch.setattr(alphafold, "fetch_alphafold_kd", _record)

    alphafold.enrich_with_alphafold(abl1)
    calls.clear()
    alphafold.enrich_with_alphafold(abl1)
    assert calls == [], "unchanged inputs should not refetch"

    str_seq = abl1.uniprot.canonical_seq
    abl1.uniprot.canonical_seq = ("A" if str_seq[0] != "A" else "C") + str_seq[1:]
    alphafold.enrich_with_alphafold(abl1)
    assert len(calls) == 1
