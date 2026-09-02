"""Superpose kinase-domain structures onto a shared reference frame (PDB 1GAG).

Each structure (a KinCoRe active-state CIF or an AlphaFold DB model) is rigid-body
superposed onto the INSR kinase-domain template ``data/1gag_template.pdb`` on equivalent
C-alpha atoms, and the resulting transform is recentered so the reference KLIFS-pocket
centroid sits at the origin. The C-alpha correspondence is chosen per structure in three
tiers: the KLIFS pocket (primary, sequence-independent), the Modi & Dunbrack MSA columns
(for the few kinases lacking a KLIFS pocket but carrying an MSA row), and a full-sequence
pairwise alignment to the template (last resort, e.g. atypical non-ePK folds).

The computed rotation/translation is stored on the structure model
(:class:`~mkt.schema.kinase_schema.Superposition`) and applied via ``Bio.PDB`` before any
PyMOL output or app render so all structures share a common frame.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio.SVDSuperimposer import SVDSuperimposer
from mkt.databases.aligners import BioAligner
from mkt.databases.io_utils import DataSource
from mkt.databases.utils import convert_mmcifdict2structure
from mkt.schema.io_utils import get_repo_root
from mkt.schema.kinase_schema import KinaseInfo, Superposition

logger = logging.getLogger(__name__)

REFERENCE_HGNC = "INSR"
"""HGNC name of the kinase whose KLIFS/MSA maps index the reference template."""

REFERENCE_SOURCE = DataSource(
    name="1gag_template.pdb",
    path=os.path.join(get_repo_root(), "data", "1gag_template.pdb"),
    citation="PDB 1GAG (Hubbard, EMBO J. 1997) -- INSR kinase domain",
)
"""Reference template: the insulin-receptor (INSR) tyrosine-kinase domain."""

MIN_SUPERPOSE_ATOMS = 3
"""Minimum equivalent C-alpha pairs required to define a rigid-body transform."""

WARN_RMSD = 5.0
"""RMSD (Å) above which a superposition is logged as a poor fit (e.g. atypical non-ePK folds).

The transform is still stored -- the recorded RMSD flags structures that do not share the
canonical ePK reference frame, rather than silently dropping them.
"""


@dataclass
class ReferenceFrame:
    """Reference C-alpha coordinates the kinase-domain structures are superposed onto.

    Built once per run from the 1GAG template and the reference kinase's KLIFS/MSA maps.
    """

    name: str
    """Reference name recorded on each Superposition (e.g. "1GAG")."""
    klifs: dict[str, np.ndarray]
    """KLIFS region:idx -> template C-alpha coordinate."""
    msa: dict[str, np.ndarray]
    """MSA region:idx -> template C-alpha coordinate."""
    full_seq: str
    """Template one-letter sequence (C-alpha / residue-number order)."""
    full_coords: np.ndarray
    """Template C-alpha coordinates, parallel to full_seq (N x 3)."""
    origin: np.ndarray
    """Origin shift (KLIFS-pocket centroid) subtracted so the shared frame is centered."""


def _parse_ca(chain) -> tuple[dict[int, np.ndarray], str, list[int]]:
    """Extract ordered C-alpha coordinates and sequence from a Bio.PDB chain/structure.

    Parameters
    ----------
    chain : Iterable[Residue]
        Any iterable of Bio.PDB residues (a chain or ``structure.get_residues()``).

    Returns
    -------
    tuple[dict[int, np.ndarray], str, list[int]]
        Residue number -> C-alpha coordinate, the one-letter sequence (residue-number
        order), and the ordered residue numbers.
    """
    dict_ca: dict[int, np.ndarray] = {}
    dict_name: dict[int, str] = {}
    for residue in chain:
        if residue.id[0] == " " and "CA" in residue:
            dict_ca[residue.id[1]] = residue["CA"].coord
            dict_name[residue.id[1]] = residue.resname
    list_resnum = sorted(dict_ca)
    str_seq = "".join(protein_letters_3to1.get(dict_name[i], "X") for i in list_resnum)
    return dict_ca, str_seq, list_resnum


def _structure_ca(
    dict_cif: dict[str, str | list[str]], structure_id: str
) -> tuple[dict[int, np.ndarray], str, list[int]]:
    """Parse a structure's mmCIF dict into ordered C-alpha coordinates and sequence.

    Parameters
    ----------
    dict_cif : dict[str, str | list[str]]
        The mmCIF dict (residues numbered by UniProt position).
    structure_id : str
        Identifier for the parsed structure.

    Returns
    -------
    tuple[dict[int, np.ndarray], str, list[int]]
        See :func:`_parse_ca`.
    """
    structure = convert_mmcifdict2structure(dict_cif, structure_id=structure_id)
    return _parse_ca(structure.get_residues())


def build_reference_frame(dict_kinase: dict[str, KinaseInfo]) -> ReferenceFrame:
    """Build the shared reference frame from the 1GAG template and the reference kinase.

    The template PDB numbering differs from UniProt, so the reference kinase's canonical
    sequence is aligned to the template sequence to map its KLIFS/MSA UniProt indices onto
    template residue numbers.

    Parameters
    ----------
    dict_kinase : dict[str, KinaseInfo]
        The kinase dictionary (must contain the reference kinase, :data:`REFERENCE_HGNC`).

    Returns
    -------
    ReferenceFrame
        The reference C-alpha coordinates and origin shift.
    """
    structure = PDBParser(QUIET=True).get_structure(
        "template", REFERENCE_SOURCE.resolve()
    )
    dict_ca, str_pdb_seq, list_resnum = _parse_ca(next(iter(next(iter(structure)))))
    full_coords = np.array([dict_ca[i] for i in list_resnum])

    obj_ref = dict_kinase[REFERENCE_HGNC]
    seq_ref = obj_ref.uniprot.canonical_seq
    # map reference UniProt index (1-based) -> template residue number via global alignment
    alignment = BioAligner(mode="global", substitution_matrix="BLOSUM62").align(
        seq_ref, str_pdb_seq
    )[0]
    indices = alignment.indices  # 2 x L; -1 marks a gap
    uniprot2resnum: dict[int, int] = {}
    for col in range(indices.shape[1]):
        i_seq, i_pdb = indices[0, col], indices[1, col]
        if i_seq >= 0 and i_pdb >= 0:
            uniprot2resnum[i_seq + 1] = list_resnum[i_pdb]

    def _coords(dict_region2uniprot: dict[str, int | None]) -> dict[str, np.ndarray]:
        return {
            region: dict_ca[uniprot2resnum[uniprot_idx]]
            for region, uniprot_idx in dict_region2uniprot.items()
            if uniprot_idx and uniprot_idx in uniprot2resnum
        }

    klifs = _coords(obj_ref.KLIFS2UniProtIdx or {})
    msa = (
        _coords(obj_ref.kincore.msa.region2uniprot)
        if obj_ref.kincore is not None and obj_ref.kincore.msa is not None
        else {}
    )
    origin = np.mean(np.array(list(klifs.values())), axis=0)

    logger.info(
        f"Built reference frame from {REFERENCE_SOURCE.name} via {REFERENCE_HGNC}: "
        f"{len(klifs)} KLIFS, {len(msa)} MSA, {len(str_pdb_seq)} full C-alpha atoms."
    )
    return ReferenceFrame(
        name="1GAG",
        klifs=klifs,
        msa=msa,
        full_seq=str_pdb_seq,
        full_coords=full_coords,
        origin=origin,
    )


def _pairs_by_region(
    dict_ref: dict[str, np.ndarray],
    dict_region2uniprot: dict[str, int | None],
    dict_ca: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pair reference and structure C-alpha coordinates on shared region keys.

    Parameters
    ----------
    dict_ref : dict[str, np.ndarray]
        Reference region:idx -> coordinate (KLIFS or MSA).
    dict_region2uniprot : dict[str, int | None]
        Structure region:idx -> UniProt index.
    dict_ca : dict[int, np.ndarray]
        Structure residue number -> C-alpha coordinate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Parallel (reference, structure) coordinate arrays over the shared, mapped regions.
    """
    ref_coords, mov_coords = [], []
    for region, coord in dict_ref.items():
        uniprot_idx = dict_region2uniprot.get(region)
        if uniprot_idx is not None and uniprot_idx in dict_ca:
            ref_coords.append(coord)
            mov_coords.append(dict_ca[uniprot_idx])
    return np.array(ref_coords), np.array(mov_coords)


def _pairs_by_sequence(
    frame: ReferenceFrame,
    str_seq: str,
    dict_ca: dict[int, np.ndarray],
    list_resnum: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pair reference and structure C-alpha coordinates via full-sequence alignment.

    Parameters
    ----------
    frame : ReferenceFrame
        The reference frame (full template sequence + coordinates).
    str_seq : str
        The structure one-letter sequence (residue-number order).
    dict_ca : dict[int, np.ndarray]
        Structure residue number -> C-alpha coordinate.
    list_resnum : list[int]
        Structure residue numbers, parallel to ``str_seq``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Parallel (reference, structure) coordinate arrays over aligned columns.
    """
    struct_coords = np.array([dict_ca[i] for i in list_resnum])
    alignment = BioAligner(mode="global", substitution_matrix="BLOSUM62").align(
        frame.full_seq, str_seq
    )[0]
    indices = alignment.indices  # 2 x L; -1 marks a gap
    ref_coords, mov_coords = [], []
    for col in range(indices.shape[1]):
        i_ref, i_mov = indices[0, col], indices[1, col]
        if i_ref >= 0 and i_mov >= 0:
            ref_coords.append(frame.full_coords[i_ref])
            mov_coords.append(struct_coords[i_mov])
    return np.array(ref_coords), np.array(mov_coords)


def _superpose(
    ref_coords: np.ndarray,
    mov_coords: np.ndarray,
    origin: np.ndarray,
) -> tuple[list[list[float]], list[float], float]:
    """Superpose structure coordinates onto reference coordinates, recentered to the origin.

    Parameters
    ----------
    ref_coords : np.ndarray
        Reference C-alpha coordinates (N x 3).
    mov_coords : np.ndarray
        Structure C-alpha coordinates (N x 3), paired with ``ref_coords``.
    origin : np.ndarray
        Origin shift subtracted from the translation (reference pocket centroid).

    Returns
    -------
    tuple[list[list[float]], list[float], float]
        The 3x3 rotation, length-3 translation (origin-recentered), and RMSD; applied as
        ``coord @ rotation + translation``.
    """
    sup = SVDSuperimposer()
    sup.set(ref_coords, mov_coords)
    sup.run()
    rot, tran = sup.get_rotran()
    return rot.tolist(), (tran - origin).tolist(), float(sup.get_rms())


def _superpose_structure(
    dict_cif: dict[str, str | list[str]],
    obj_kinase: KinaseInfo,
    frame: ReferenceFrame,
    structure_id: str,
) -> Superposition | None:
    """Compute the reference-frame transform for one structure via the tiered correspondence.

    Parameters
    ----------
    dict_cif : dict[str, str | list[str]]
        The structure's mmCIF dict.
    obj_kinase : KinaseInfo
        The kinase carrying the structure (supplies KLIFS/MSA maps).
    frame : ReferenceFrame
        The shared reference frame.
    structure_id : str
        Identifier for the parsed structure.

    Returns
    -------
    Superposition | None
        The transform, or None if too few equivalent C-alpha atoms were found.
    """
    dict_ca, str_seq, list_resnum = _structure_ca(dict_cif, structure_id)

    method = "klifs"
    ref_coords, mov_coords = _pairs_by_region(
        frame.klifs, obj_kinase.KLIFS2UniProtIdx or {}, dict_ca
    )
    if len(ref_coords) < MIN_SUPERPOSE_ATOMS:
        method = "msa"
        msa = obj_kinase.kincore.msa if obj_kinase.kincore is not None else None
        ref_coords, mov_coords = _pairs_by_region(
            frame.msa, msa.region2uniprot if msa is not None else {}, dict_ca
        )
    if len(ref_coords) < MIN_SUPERPOSE_ATOMS:
        method = "sequence"
        ref_coords, mov_coords = _pairs_by_sequence(
            frame, str_seq, dict_ca, list_resnum
        )
    if len(ref_coords) < MIN_SUPERPOSE_ATOMS:
        logger.warning(
            f"Too few equivalent C-alpha atoms to superpose {structure_id}; skipping."
        )
        return None

    rotation, translation, rmsd = _superpose(ref_coords, mov_coords, frame.origin)
    if rmsd > WARN_RMSD:
        logger.warning(
            f"Poor reference-frame fit for {structure_id} "
            f"(method={method}, n={len(ref_coords)}, rmsd={rmsd:.1f} Å); "
            "structure likely does not share the canonical ePK frame."
        )
    return Superposition(
        rotation=rotation,
        translation=translation,
        reference=frame.name,
        method=method,
        rmsd=rmsd,
        n_atoms=len(ref_coords),
        source=REFERENCE_SOURCE.provenance(),
    )


def superpose_structure(
    structure_model,
    obj_kinase: KinaseInfo,
    frame: ReferenceFrame,
    structure_id: str,
    force: bool = False,
) -> None:
    """Populate the reference-frame :class:`Superposition` on one structure model in place.

    Called by the structure-owning enrichment steps (``kincore`` for the KinCoRe CIF,
    ``alphafold`` for the AF model) so a structure's superposition is (re)generated alongside
    the structure itself.

    Parameters
    ----------
    structure_model : KinCoReCIF | AlphaFold | None
        The structure model to superpose (carries a ``.cif`` mmCIF dict and a
        ``.superposition`` field); a no-op when None.
    obj_kinase : KinaseInfo
        The kinase carrying the structure (supplies KLIFS/MSA maps).
    frame : ReferenceFrame
        The shared reference frame (see :func:`build_reference_frame`).
    structure_id : str
        Identifier for the parsed structure.
    force : bool, optional
        Recompute even when the structure already carries a superposition, by default False.

    Returns
    -------
    None
    """
    if structure_model is None:
        return
    if structure_model.superposition is not None and not force:
        return  # idempotent: keep an already-computed superposition (unchanged structure)
    structure_model.superposition = _superpose_structure(
        structure_model.cif, obj_kinase, frame, structure_id
    )
