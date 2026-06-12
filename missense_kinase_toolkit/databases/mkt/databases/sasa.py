import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
from typing import ClassVar

import pandas as pd
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Structure import Structure
from mkt.databases.colors import map_aa_to_single_letter_code
from mkt.databases.io_utils import return_kinase_dict
from mkt.databases.utils import (
    convert_mmcifdict2structure,
    convert_structure2string,
    load_kinase_object,
)
from mkt.schema.kinase_schema import KinaseInfo
from mkt.schema.utils import rgetattr
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from tqdm import tqdm

logger = logging.getLogger(__name__)


MAX_ASA_TIEN_2013 = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}
"""Theoretical maximum accessible surface area (Å^2) per residue.

From Tien et al. (2013), PLoS ONE 8(11): e80635; used to normalize SASA to
relative solvent accessibility (rSASA), keyed by 3-letter residue name.
"""

DEFAULT_PROBE_RADIUS = 1.40
"""Default solvent/probe radius (Å), shared by both SASA backends."""
DEFAULT_N_POINTS = 100
"""Default Shrake-Rupley sphere points per atom (Bio.PDB); presets raise this to
~960 for converged per-residue SASA."""
DEFAULT_DOT_DENSITY = 3
"""Default PyMOL surface point density (1-4)."""


@dataclass(kw_only=True)
class BaseSASAConfig:
    """Backend-compatible options for :class:`ResidueSASA`.

    Field names mirror the ``ResidueSASA`` constructor so a config can be
    expanded via :meth:`ResidueSASA.from_dataclass`. Presets (subclasses)
    override these defaults into vetted, internally consistent recipes; see
    :class:`StandardSASAConfigs`.
    """

    bool_biopython: bool = True
    """Compute SASA with the Bio.PDB Shrake-Rupley backend."""
    bool_pymol: bool = False
    """Compute SASA with the PyMOL ``dot_solvent`` backend."""
    bool_include_hydrogens: bool = False
    """Keep explicit hydrogens; False gives conventional heavy-atom SASA."""
    bool_relative: bool = True
    """Add relative SASA (rSASA); requires heavy-atom SASA (no explicit H)."""
    probe_radius: float = DEFAULT_PROBE_RADIUS
    """Solvent/probe radius (Å), shared by both backends."""
    n_points: int = DEFAULT_N_POINTS
    """Shrake-Rupley sphere points per atom (Bio.PDB only); default 100 is
    unlikely to be converged for per-residue SASA (presets use ~960)."""
    dot_density: int = DEFAULT_DOT_DENSITY
    """PyMOL surface point density 1-4 (PyMOL only)."""


@dataclass(kw_only=True)
class BioPythonHeavyConfig(BaseSASAConfig):
    """Heavy-atom Bio.PDB Shrake-Rupley with converged sampling (recommended)."""

    bool_biopython: bool = True
    bool_pymol: bool = False
    n_points: int = 960


@dataclass(kw_only=True)
class PyMOLHeavyConfig(BaseSASAConfig):
    """Heavy-atom PyMOL ``dot_solvent`` with high dot density."""

    bool_biopython: bool = False
    bool_pymol: bool = True
    dot_density: int = 4


@dataclass(kw_only=True)
class CrossValidationConfig(BaseSASAConfig):
    """Heavy-atom SASA from both backends with matched probe/sampling."""

    bool_biopython: bool = True
    bool_pymol: bool = True
    n_points: int = 960
    dot_density: int = 4


@dataclass(kw_only=True)
class BioPythonHydrogenConfig(BaseSASAConfig):
    """All-atom (explicit-H) Bio.PDB Shrake-Rupley; rSASA disabled.

    Bondi radii + explicit hydrogens is a standard, self-consistent all-atom
    surface (not a double count). rSASA is disabled because the Tien et al.
    (2013) maxima are a heavy-atom reference; the structures must actually
    carry hydrogens (the run warns per structure if they do not).
    """

    bool_biopython: bool = True
    bool_pymol: bool = False
    bool_include_hydrogens: bool = True
    bool_relative: bool = False
    n_points: int = 960


@dataclass(kw_only=True)
class PyMOLHydrogenConfig(BaseSASAConfig):
    """All-atom (explicit-H) PyMOL ``dot_solvent``; rSASA disabled."""

    bool_biopython: bool = False
    bool_pymol: bool = True
    bool_include_hydrogens: bool = True
    bool_relative: bool = False
    dot_density: int = 4


@dataclass(kw_only=True)
class CrossValidationHydrogenConfig(BaseSASAConfig):
    """All-atom (explicit-H) SASA from both backends; rSASA disabled."""

    bool_biopython: bool = True
    bool_pymol: bool = True
    bool_include_hydrogens: bool = True
    bool_relative: bool = False
    n_points: int = 960
    dot_density: int = 4


class StandardSASAConfigs(Enum):
    """Named, internally consistent SASA configurations.

    Use via ``ResidueSASA.from_dataclass(StandardSASAConfigs.BIOPYTHON_HEAVY)``.
    Each value omits any backend-inert field, so nothing silently no-ops.
    ``*_HEAVY`` give heavy-atom SASA with rSASA; ``*_HYDROGEN`` give all-atom
    (explicit-H) absolute SASA with rSASA disabled.
    """

    BIOPYTHON_HEAVY = BioPythonHeavyConfig()
    PYMOL_HEAVY = PyMOLHeavyConfig()
    CROSS_VALIDATION = CrossValidationConfig()
    BIOPYTHON_HYDROGEN = BioPythonHydrogenConfig()
    PYMOL_HYDROGEN = PyMOLHydrogenConfig()
    CROSS_VALIDATION_HYDROGEN = CrossValidationHydrogenConfig()


class StandardSASAConfigChoice(str, Enum):
    """String aliases for :class:`StandardSASAConfigs` (e.g. for CLI choices)."""

    BIOPYTHON_HEAVY = "BIOPYTHON_HEAVY"
    PYMOL_HEAVY = "PYMOL_HEAVY"
    CROSS_VALIDATION = "CROSS_VALIDATION"
    BIOPYTHON_HYDROGEN = "BIOPYTHON_HYDROGEN"
    PYMOL_HYDROGEN = "PYMOL_HYDROGEN"
    CROSS_VALIDATION_HYDROGEN = "CROSS_VALIDATION_HYDROGEN"


# module-level compute functions (picklable, so they can run in worker
# processes); ResidueSASA.run dispatches to them in serial or via a pool


def _strip_hydrogens(structure: Structure) -> None:
    """Remove all hydrogen atoms from a Bio.PDB Structure in place."""
    for residue in structure.get_residues():
        for atom_id in [a.id for a in residue if a.element == "H"]:
            residue.detach_child(atom_id)


def _residue_sasa_biopython(
    structure: Structure,
    *,
    bool_include_hydrogens: bool,
    probe_radius: float,
    n_points: int,
) -> list[dict[str, object]]:
    """Per-residue SASA rows via Bio.PDB Shrake-Rupley."""
    # AF2 CIFs carry explicit hydrogens; remove them for heavy-atom SASA
    if not bool_include_hydrogens:
        _strip_hydrogens(structure)

    sr = ShrakeRupley(probe_radius=probe_radius, n_points=n_points)
    sr.compute(structure, level="R")

    list_rows = []
    for residue in structure.get_residues():
        # skip heteroatoms/waters; standard residues have a blank hetero flag
        if residue.id[0] != " ":
            continue
        list_rows.append(
            {
                "uniprot_idx": residue.id[1],
                "resname": residue.resname,
                "sasa": residue.sasa,
            }
        )
    return list_rows


def _residue_sasa_pymol(
    structure: Structure,
    *,
    bool_include_hydrogens: bool,
    probe_radius: float,
    dot_density: int,
) -> list[dict[str, object]]:
    """Per-residue SASA rows via PyMOL ``dot_solvent`` (lazy ``pymol2`` import)."""
    import pymol2

    pdb_string = convert_structure2string(structure)

    dict_sasa: dict[tuple[int, str], float] = {}
    with pymol2.PyMOL() as session:
        cmd = session.cmd
        cmd.read_pdbstr(pdb_string, "kinase")

        # AF2 CIFs carry explicit hydrogens; remove them for heavy-atom SASA
        if not bool_include_hydrogens:
            cmd.remove("hydro")

        # dot_solvent=1 -> solvent accessible (vs molecular) surface area
        cmd.set("dot_solvent", 1)
        cmd.set("dot_density", dot_density)
        cmd.set("solvent_radius", probe_radius)

        # load_b=1 stores per-atom SASA in the b-factor for iteration
        cmd.get_area("kinase", load_b=1)
        cmd.iterate(
            "kinase and polymer",
            "dict_sasa[(int(resi), resn)] = "
            "dict_sasa.get((int(resi), resn), 0.0) + b",
            space={"dict_sasa": dict_sasa},
        )

    return [
        {"uniprot_idx": resi, "resname": resn, "sasa": sasa}
        for (resi, resn), sasa in dict_sasa.items()
    ]


def _assemble_sasa_df(
    list_rows: list[dict[str, object]],
    *,
    bool_relative: bool,
) -> pd.DataFrame:
    """Build the per-residue DataFrame and optional relative-SASA column."""
    df = pd.DataFrame(list_rows)
    df["residue"] = df["resname"].apply(map_aa_to_single_letter_code)
    df = df[["uniprot_idx", "residue", "resname", "sasa"]]
    df = df.sort_values("uniprot_idx").reset_index(drop=True)

    if bool_relative:
        df["rsa"] = df.apply(
            lambda row: (
                row["sasa"] / MAX_ASA_TIEN_2013[row["resname"]]
                if row["resname"] in MAX_ASA_TIEN_2013
                else float("nan")
            ),
            axis=1,
        )

    return df


def _compute_kinase_sasa(
    obj_kinase: KinaseInfo,
    str_method: str,
    *,
    bool_include_hydrogens: bool,
    bool_relative: bool,
    probe_radius: float,
    n_points: int,
    dot_density: int,
) -> pd.DataFrame | None:
    """Compute per-residue SASA for one kinase with one backend.

    Returns the per-residue DataFrame (without ``hgnc_name``/``method`` tags),
    or None if the kinase has no KinCore CIF structure.
    """
    if rgetattr(obj_kinase, "kincore.cif") is None:
        logger.warning(f"No KinCore CIF structure for {obj_kinase.hgnc_name}")
        return None

    structure = convert_mmcifdict2structure(
        obj_kinase.kincore.cif.cif,
        structure_id=obj_kinase.hgnc_name,
    )

    # guard the silent no-op: explicit-H analysis is meaningless if the
    # structure was never protonated (some KinCore models are heavy-atom only)
    if bool_include_hydrogens and not any(
        atom.element == "H" for atom in structure.get_atoms()
    ):
        logger.warning(
            "bool_include_hydrogens=True but %s has no explicit hydrogens; "
            "SASA equals the heavy-atom result (structure not protonated).",
            obj_kinase.hgnc_name,
        )

    if str_method == "biopython":
        list_rows = _residue_sasa_biopython(
            structure,
            bool_include_hydrogens=bool_include_hydrogens,
            probe_radius=probe_radius,
            n_points=n_points,
        )
    else:
        list_rows = _residue_sasa_pymol(
            structure,
            bool_include_hydrogens=bool_include_hydrogens,
            probe_radius=probe_radius,
            dot_density=dot_density,
        )

    return _assemble_sasa_df(list_rows, bool_relative=bool_relative)


def _sasa_task(task: tuple) -> pd.DataFrame | None:
    """Worker entry point: compute one (kinase, backend) task and tag the rows.

    ``task`` is ``(hgnc_name, obj_kinase, str_method, params)`` where ``params``
    is the scalar config dict from :meth:`ResidueSASA._compute_params`.
    """
    hgnc_name, obj_kinase, str_method, params = task
    df = _compute_kinase_sasa(obj_kinase, str_method, **params)
    if df is None:
        return None
    df.insert(0, "method", str_method)
    df.insert(0, "hgnc_name", hgnc_name)
    return df


class ResidueSASA(BaseModel):
    """Calculate per-residue solvent accessible surface area from KinCore CIFs.

    Configure the run via the fields below, then call :meth:`run` to compute
    per-residue SASA for each kinase with each selected backend; results are
    cached on and accessed via the :attr:`df` property. Residues are numbered
    by UniProt sequence position (the CIF ``auth_seq_id``), and hydrogens are
    removed by default so SASA reflects conventional heavy atoms.

    Prefer :meth:`from_dataclass` with a :class:`StandardSASAConfigs` preset for
    vetted, backend-compatible options; direct construction is allowed but warns
    that option compatibility is the caller's responsibility.

    Parameters:
    -----------
    dict_kinase : dict[str, KinaseInfo] | None
        Mapping of HGNC name to ``KinaseInfo`` object. If None, kinases are
        deserialized on demand: one at a time via ``load_kinase_object`` when
        ``list_ids`` is given, else the whole proteome via ``return_kinase_dict``.
    list_ids : list[str] | None
        HGNC names to deserialize (one at a time) when ``dict_kinase`` is None;
        None loads all kinases.
    bool_biopython : bool
        If True (default), compute SASA with Bio.PDB Shrake-Rupley.
    bool_pymol : bool
        If True, also compute SASA with PyMOL ``dot_solvent`` (requires
        ``pymol-open-source``, installed via conda); False by default. If PyMOL
        cannot be imported it is skipped with a warning, unless it is the only
        selected backend, in which case an ``ImportError`` is raised.
    bool_include_hydrogens : bool
        If False (default), strip hydrogens for heavy-atom SASA; AlphaFold2
        CIFs include explicit hydrogens.
    bool_relative : bool
        If True (default), add an ``rsa`` column normalizing SASA by the
        Tien et al. (2013) theoretical maximum ASA for each residue type.
    probe_radius : float
        Solvent/probe radius in Å, shared by both backends, by default 1.40.
    n_points : int
        Shrake-Rupley sphere points per atom (Bio.PDB), by default 100. The
        default is fast but **unlikely to be converged** for per-residue SASA;
        use ~960 (e.g. ``StandardSASAConfigs.BIOPYTHON_HEAVY``) for analysis.
    dot_density : int
        PyMOL surface point density (1-4), by default 3.
    n_jobs : int
        Worker processes for :meth:`run`. 1 (default) runs serially; >1 uses a
        process pool; -1 uses all CPU cores. Parallelism is process-based
        (Shrake-Rupley is CPU-bound); each task carries its ``KinaseInfo``, so
        workers do no deserialization. Worthwhile for many kinases or high
        ``n_points``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # toggled True by from_dataclass so model_post_init skips the "verify config
    # compatibility" nudge for vetted presets; ClassVar -> not a model field
    _building_from_dataclass: ClassVar[bool] = False

    dict_kinase: dict[str, KinaseInfo] | None = None
    """Mapping of HGNC name to ``KinaseInfo``; deserialized on demand if None."""
    list_ids: list[str] | None = None
    """HGNC names to deserialize (one at a time) when ``dict_kinase`` is None."""
    bool_biopython: bool = True
    """Compute SASA with the Bio.PDB Shrake-Rupley backend."""
    bool_pymol: bool = False
    """Compute SASA with the PyMOL ``dot_solvent`` backend."""
    bool_include_hydrogens: bool = False
    """Keep explicit hydrogens; False gives conventional heavy-atom SASA."""
    bool_relative: bool = True
    """Add relative SASA (rSASA); requires heavy-atom SASA (no explicit H)."""
    probe_radius: float = Field(default=DEFAULT_PROBE_RADIUS, gt=0)
    """Solvent/probe radius (Å), shared by both backends."""
    n_points: int = Field(default=DEFAULT_N_POINTS, ge=1)
    """Shrake-Rupley sphere points per atom (Bio.PDB only); default 100 is
    unlikely to be converged for per-residue SASA (use ~960)."""
    dot_density: int = Field(default=DEFAULT_DOT_DENSITY, ge=1, le=4)
    """PyMOL surface point density 1-4 (PyMOL only)."""
    n_jobs: int = 1
    """Worker processes for :meth:`run`: 1 serial, >1 pool, -1 all cores."""

    _df: pd.DataFrame | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_config(self) -> "ResidueSASA":
        """Reject incompatible option combinations and warn on inert options.

        Numeric ranges are enforced by ``Field`` constraints; this covers the
        cross-field rules that ``Field`` cannot express.

        Raises:
        -------
        ValueError
            If no backend is selected, or if relative SASA is requested together
            with explicit hydrogens.
        """
        if not (self.bool_biopython or self.bool_pymol):
            raise ValueError(
                "at least one of bool_biopython or bool_pymol must be True"
            )

        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError(
                f"n_jobs must be -1 (all cores) or a positive integer, "
                f"got {self.n_jobs}."
            )

        if self.bool_include_hydrogens and self.bool_relative:
            raise ValueError(
                "bool_relative is incompatible with bool_include_hydrogens: the "
                "Tien et al. (2013) maxima are a heavy-atom reference, so relative "
                "SASA is invalid with explicit hydrogens. Set bool_relative=False "
                "(e.g. a *_HYDROGEN preset in StandardSASAConfigs)."
            )

        # warn on options that silently no-op for the selected backend(s)
        if not self.bool_pymol and self.dot_density != DEFAULT_DOT_DENSITY:
            logger.warning(
                "dot_density=%s set but bool_pymol=False; it only affects the "
                "PyMOL backend and will be ignored.",
                self.dot_density,
            )
        if not self.bool_biopython and self.n_points != DEFAULT_N_POINTS:
            logger.warning(
                "n_points=%s set but bool_biopython=False; it only affects the "
                "Bio.PDB backend and will be ignored.",
                self.n_points,
            )

        return self

    def model_post_init(self, __context) -> None:
        """Warn on direct construction and resolve ``dict_kinase``."""
        if not type(self)._building_from_dataclass:
            logger.warning(
                "ResidueSASA was constructed directly; backend-compatible options "
                "are your responsibility. Prefer ResidueSASA.from_dataclass(...) "
                "with a StandardSASAConfigs preset to avoid incompatible settings "
                "(e.g. hydrogen/relative, probe radius, n_points/dot_density)."
            )

        if self.dict_kinase is None:
            if self.list_ids is None:
                # whole proteome: one bulk deserialization
                self.dict_kinase = return_kinase_dict()
            else:
                # deserialize one KinaseInfo at a time so only the requested
                # kinases are loaded (and the module stays cheap to import)
                self.dict_kinase = {}
                for hgnc in self.list_ids:
                    try:
                        self.dict_kinase[hgnc] = load_kinase_object(hgnc)
                    except ValueError:
                        logger.warning("Kinase %s not found; skipping.", hgnc)

    @classmethod
    def from_dataclass(
        cls,
        config: "BaseSASAConfig | StandardSASAConfigs",
        **kwargs,
    ) -> "ResidueSASA":
        """Build a ResidueSASA from a vetted config dataclass or standard preset.

        This is the recommended constructor: presets in
        :class:`StandardSASAConfigs` carry backend-compatible options, so the
        direct-construction compatibility warning is suppressed.

        Parameters:
        -----------
        config : BaseSASAConfig | StandardSASAConfigs
            A SASA config dataclass (e.g. ``BioPythonHeavyConfig``) or a
            ``StandardSASAConfigs`` enum member.
        **kwargs
            Extra constructor arguments, e.g. ``dict_kinase`` or ``list_ids``.

        Returns:
        --------
        ResidueSASA
            Configured (but not yet run) instance.
        """
        if isinstance(config, StandardSASAConfigs):
            config = config.value

        cls._building_from_dataclass = True
        try:
            return cls(**asdict(config), **kwargs)
        finally:
            cls._building_from_dataclass = False

    @property
    def df(self) -> pd.DataFrame | None:
        """Per-residue SASA results; None until :meth:`run` is called."""
        return self._df

    @property
    def list_methods(self) -> list[str]:
        """Backends to run, selected via ``bool_biopython`` / ``bool_pymol``."""
        list_out = []
        if self.bool_biopython:
            list_out.append("biopython")
        if self.bool_pymol:
            list_out.append("pymol")
        return list_out

    @staticmethod
    def _pymol_available() -> bool:
        """Check whether the optional PyMOL backend can be imported.

        The PyMOL PyPI wheel is unreliable, so any import failure (not just
        ``ModuleNotFoundError``) is treated as unavailable.

        Returns:
        --------
        bool
            True if ``pymol2`` imports successfully, otherwise False.
        """
        try:
            import pymol2  # noqa: F401

            return True
        except Exception as e:
            logger.debug(f"pymol2 import failed: {e}")
            return False

    def _resolve_methods(self) -> list[str]:
        """Resolve the backends to run, dropping PyMOL if it cannot be imported.

        Returns:
        --------
        list[str]
            Selected backends with "pymol" removed (with a warning) when
            ``pymol2`` is unavailable but the Bio.PDB backend can stand in.

        Raises:
        -------
        ImportError
            If PyMOL is the only requested backend but is unavailable.
        """
        list_methods = self.list_methods
        if "pymol" in list_methods and not self._pymol_available():
            if self.bool_biopython:
                logger.warning(
                    "PyMOL backend unavailable (pymol-open-source not installed); "
                    "falling back to the Bio.PDB Shrake-Rupley backend only."
                )
                list_methods = [method for method in list_methods if method != "pymol"]
            else:
                raise ImportError(
                    "PyMOL backend requested but pymol-open-source is not available; "
                    "install it via conda "
                    "(`conda install -c conda-forge pymol-open-source`) "
                    "or set bool_biopython=True."
                )
        return list_methods

    def run(self) -> pd.DataFrame:
        """Compute per-residue SASA for all kinases and backends.

        Loops over each selected backend and kinase, tagging rows with
        ``method`` and ``hgnc_name``; kinases without a KinCore CIF are skipped.

        Returns:
        --------
        pd.DataFrame
            Long-format DataFrame (also cached on :attr:`df`) with columns
            ``hgnc_name``, ``method``, ``uniprot_idx`` (UniProt position),
            ``residue`` (single-letter code), ``resname`` (3-letter code),
            ``sasa`` (Å^2), and, if ``bool_relative``, ``rsa``. Empty if no
            kinase had a CIF structure.
        """
        list_methods = self._resolve_methods()

        # drop kinases without CIF structures up front to avoid redundant checks
        dict_temp = {
            k: v
            for k, v in self.dict_kinase.items()
            if rgetattr(v, "kincore.cif") is not None
        }

        list_no_cif = set(self.dict_kinase.keys()) - set(dict_temp.keys())
        if list_no_cif:
            logger.warning(
                "The following %d kinases have no KinCore CIF structure and will be skipped: %s",
                len(list_no_cif),
                ", ".join(sorted(list_no_cif)),
            )

        params = self._compute_params()
        list_tasks = [
            (hgnc_name, obj_kinase, str_method, params)
            for str_method in list_methods
            for hgnc_name, obj_kinase in dict_temp.items()
        ]
        list_df = self._run_tasks(list_tasks)

        if list_df:
            self._df = pd.concat(list_df, ignore_index=True)
        else:
            logger.warning("No kinases with CIF structures; returning empty DataFrame.")
            self._df = pd.DataFrame()

        return self._df

    def _compute_params(self) -> dict[str, object]:
        """Scalar config forwarded to the (picklable) per-kinase compute worker."""
        return {
            "bool_include_hydrogens": self.bool_include_hydrogens,
            "bool_relative": self.bool_relative,
            "probe_radius": self.probe_radius,
            "n_points": self.n_points,
            "dot_density": self.dot_density,
        }

    def _resolve_n_workers(self) -> int:
        """Resolve ``n_jobs`` to a concrete worker count (>=1)."""
        if self.n_jobs == -1:
            return os.cpu_count() or 1
        return self.n_jobs

    def _run_tasks(self, list_tasks: list[tuple]) -> list[pd.DataFrame]:
        """Run ``(kinase, backend)`` tasks serially or across a process pool."""
        if not list_tasks:
            return []

        n_workers = min(self._resolve_n_workers(), len(list_tasks))
        desc = "Calculating per-residue SASA"

        if n_workers == 1:
            return [
                df
                for df in (
                    _sasa_task(task) for task in tqdm(list_tasks, desc=f"{desc}...")
                )
                if df is not None
            ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            return [
                df
                for df in tqdm(
                    executor.map(_sasa_task, list_tasks),
                    total=len(list_tasks),
                    desc=f"{desc} ({n_workers} workers)...",
                )
                if df is not None
            ]
