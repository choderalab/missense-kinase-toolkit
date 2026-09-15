"""Unit and integration tests for the reference-frame superposition enrichment."""

import numpy as np
import pytest
from mkt.databases import superpose


def test_superpose_recovers_known_rigid_transform():
    """_superpose recovers a known rigid transform (RMSD ~0) and maps moving onto reference."""
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ]
    )
    # rotate 90 deg about z, then translate
    rot90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moving = ref @ rot90 + np.array([5.0, -3.0, 2.0])

    rotation, translation, rmsd = superpose._superpose(ref, moving, origin=np.zeros(3))
    assert rmsd < 1e-6
    mapped = moving @ np.array(rotation) + np.array(translation)
    assert np.allclose(mapped, ref, atol=1e-6)


def test_superpose_origin_shift_subtracts_from_translation():
    """The origin shift is subtracted from the translation (recentering the shared frame)."""
    ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    moving = ref + np.array([10.0, 0.0, 0.0])
    origin = np.array([1.0, 2.0, 3.0])

    _, tran_no_shift, _ = superpose._superpose(ref, moving, origin=np.zeros(3))
    _, tran_shift, _ = superpose._superpose(ref, moving, origin=origin)
    assert np.allclose(np.array(tran_shift), np.array(tran_no_shift) - origin)


def test_pairs_by_region_pairs_on_shared_mapped_keys():
    """_pairs_by_region pairs only regions present in the reference and mapped in the structure."""
    dict_ref = {"I:1": np.array([0.0, 0.0, 0.0]), "II:2": np.array([1.0, 1.0, 1.0])}
    region2uniprot = {
        "I:1": 10,
        "II:2": None,
        "III:3": 30,
    }  # II unmapped, III not in ref
    dict_ca = {10: np.array([9.0, 9.0, 9.0]), 30: np.array([3.0, 3.0, 3.0])}

    ref_coords, mov_coords = superpose._pairs_by_region(
        dict_ref, region2uniprot, dict_ca
    )
    assert len(ref_coords) == 1 and len(mov_coords) == 1
    assert np.allclose(mov_coords[0], [9.0, 9.0, 9.0])


@pytest.fixture(scope="module")
def dict_subset():
    """Deserialize a small subset (reference kinase + one per alignment tier)."""
    from mkt.schema.io_utils import deserialize_kinase_dict

    return deserialize_kinase_dict(list_ids=["INSR", "ABL1", "PIP5K1A"])


@pytest.fixture(scope="module")
def frame(dict_subset):
    return superpose.build_reference_frame(dict_subset)


def test_build_reference_frame_pocket_and_origin(frame):
    """The reference frame carries a populated KLIFS pocket and a 3-vector origin."""
    assert len(frame.klifs) > 50
    assert frame.origin.shape == (3,)
    assert len(frame.full_seq) == len(frame.full_coords)


def test_superpose_klifs_tier_and_centering(dict_subset, frame):
    """ABL1 superposes via the KLIFS tier with low RMSD and centers its pocket at the origin."""
    from mkt.databases.utils import convert_mmcifdict2structure

    obj = dict_subset["ABL1"]
    obj.kincore.cif.superposition = None
    superpose.superpose_structure(obj.kincore.cif, obj, frame, "ABL1_kincore")

    sp = obj.kincore.cif.superposition
    assert sp.method == "klifs" and sp.reference == "1GAG"
    assert sp.rmsd < 2.5 and sp.n_atoms > 50

    structure = convert_mmcifdict2structure(obj.kincore.cif.cif, structure_id="ABL1")
    structure.transform(np.array(sp.rotation), np.array(sp.translation))
    dict_ca = {r.id[1]: r["CA"].coord for r in structure.get_residues() if "CA" in r}
    pocket = np.array(
        [dict_ca[u] for u in obj.KLIFS2UniProtIdx.values() if u in dict_ca]
    )
    assert np.allclose(pocket.mean(axis=0), 0.0, atol=1.0)


def test_superpose_sequence_tier_fallback(dict_subset, frame):
    """A kinase without a KLIFS pocket falls back to the full-sequence tier."""
    obj = dict_subset["PIP5K1A"]
    if obj.alphafold is not None:
        obj.alphafold.superposition = None
    superpose.superpose_structure(obj.alphafold, obj, frame, "PIP5K1A_alphafold")
    assert obj.alphafold.superposition.method == "sequence"


def test_superpose_is_idempotent_unless_forced(dict_subset, frame):
    """A structure keeps its superposition unless ``force=True`` recomputes it."""
    obj = dict_subset["ABL1"]
    superpose.superpose_structure(obj.kincore.cif, obj, frame, "ABL1_kincore")
    first = obj.kincore.cif.superposition
    superpose.superpose_structure(obj.kincore.cif, obj, frame, "ABL1_kincore")
    assert obj.kincore.cif.superposition is first  # idempotent skip

    superpose.superpose_structure(
        obj.kincore.cif, obj, frame, "ABL1_kincore", force=True
    )
    assert obj.kincore.cif.superposition is not first  # forced recompute
