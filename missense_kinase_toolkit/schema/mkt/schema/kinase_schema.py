"""Core Pydantic models representing a kinase at the kinase-domain level.

Defines :class:`KinaseInfo` and its component models (:class:`UniProt`,
:class:`KinHub`, :class:`KLIFS`, :class:`Pfam`, :class:`KinCoRe`) along with the
:class:`Group`/:class:`Family` enumerations. :class:`KinaseInfo` exposes adjudication
helpers for kinase-domain sequence and group assignment.
"""

import logging
from enum import Enum

from mkt.schema.constants import (
    LIST_FULL_KLIFS_REGION,
    LIST_KLIFS_REGION,
    LIST_MSA_APE,
    LIST_MSA_REGION,
    LIST_PFAM_KD,
)
from mkt.schema.utils import fill_missing_none, rgetattr
from pydantic import BaseModel, ConfigDict, constr, field_validator
from strenum import StrEnum

logger = logging.getLogger(__name__)


class Group(StrEnum):
    """Enum class for kinase groups."""

    AGC = "AGC"  # Protein Kinase A, G, and C families
    Atypical = "Atypical"  # Atypical protein kinases
    CAMK = "CAMK"  # Calcium/calmodulin-dependent protein kinase family
    CK1 = "CK1"  # Casein kinase 1 family
    CMGC = "CMGC"  # Cyclin-dependent kinase, Mitogen-activated protein kinase, Glycogen synthase kinase, and CDK-like kinase families
    NEK = "NEK"  # NIMA (Never in Mitosis Gene A)-related kinase family - KinCoRe treats as group
    RGC = "RGC"  # Receptor guanylate cyclase family
    STE = "STE"  # Homologs of yeast Sterile 7, Sterile 11, Sterile 20 kinases
    TK = "TK"  # Tyrosine kinase family
    TKL = "TKL"  # Tyrosine kinase-like family
    Other = "Other"  # Other protein kinases


class Family(Enum):
    """Enum class for kinase families (>=5 in KinHub)."""

    STE20 = "STE20"
    CAMKL = "CAMKL"
    CDK = "CDK"
    Eph = "Eph"
    PIK = "PIK"
    MAPK = "MAPK"
    STKR = "STKR"
    NEK = "NEK"
    Src = "Src"
    DYRK = "DYRK"
    PKC = "PKC"
    STE11 = "STE11"
    RSK = "RSK"
    MLK = "MLK"
    GRK = "GRK"
    CK1 = "CK1"
    DMPK = "DMPK"
    STE7 = "STE7"
    PIKK = "PIKK"
    RSKb = "RSKb"
    Alpha = "Alpha"
    Tec = "Tec"
    CAMK1 = "CAMK1"
    PDGFR = "PDGFR"
    ULK = "ULK"
    DAPK = "DAPK"
    RAF = "RAF"
    RIPK = "RIPK"
    MLCK = "MLCK"
    PKA = "PKA"
    MAPKAPK = "MAPKAPK"
    RGC = "RGC"
    CDKL = "CDKL"
    MAST = "MAST"
    TSSK = "TSSK"
    ABC1 = "ABC1"
    PDHK = "PDHK"
    JakA = ("Jak", "JakA")
    JakB = ("Jakb", "JakB")
    PIPK = "PIPK"
    PLK = "PLK"
    Other = "Other"
    Null = None


KinaseDomainName = StrEnum(
    "KinaseDomainName", {"KD" + str(idx + 1): kd for idx, kd in enumerate(LIST_PFAM_KD)}
)

SeqUniProt = constr(pattern=r"^[ACDEFGHIKLMNPQRSTVWXY]+$")
"""Pydantic model for UniProt sequence constraints."""
SeqKLIFS = constr(pattern=r"^[ACDEFGHIKLMNPQRSTVWY\-]{85}$")
"""Pydantic model for KLIFS pocket sequence constraints."""

SwissProtPattern = r"^[A-Z][0-9][A-Z0-9]{3}[0-9](_[12])?$"
"""Regex pattern for SwissProt ID."""
SwissProtID = constr(pattern=SwissProtPattern)
"""Pydantic model for SwissProt ID constraints."""
TrEMBLPattern = r"^[A-Z][0-9][A-Z][A-Z0-9]{2}[0-9][A-Z][A-Z0-9]{2}[0-9](_[12])?$"
"""Regex pattern for TrEBML ID."""
TrEMBLID = constr(pattern=TrEMBLPattern)
"""Pydantic model for TrEMBL ID constraints."""

SwissProtPatternSuffix = SwissProtPattern.replace("$", "") + r"(_[12])?$"
"""Regex pattern for SwissProt ID with optional '_1' or '_2' for multi-KD."""
SwissProtIDSuffix = constr(pattern=SwissProtPatternSuffix)
"""Pydantic model for SwissProt ID constraints with suffix."""
TrEMBLPatternSuffix = TrEMBLPattern.replace("$", "") + r"(_[12])?$"
"""Regex pattern for TrEBML ID with optional '_1' or '_2' for multi-KD."""
TrEMBLIDSuffix = constr(pattern=TrEMBLPatternSuffix)
"""Pydantic model for TrEMBL ID constraints with suffix."""


class TemplateSource(StrEnum):
    """Enum class for template source."""

    PDB70 = "PDB70"
    activeAF2 = "activeAF2"
    activePDB = "activePDB"
    none = "notemp"


class MSASource(StrEnum):
    """Enum class for MSA source."""

    family = "family"
    ortholog = "ortholog"
    uniref90 = "uniref90"


class DFGConf(StrEnum):
    """Enum class for the DFG-motif spatial conformation (Modi & Dunbrack, PNAS 2019).

    The AF2 active-model set is all ``DFGin``; the inactive spatial groups are included
    for forward compatibility.
    """

    DFGin = "DFGin"
    DFGinter = "DFGinter"
    DFGout = "DFGout"


class DihedralCluster(StrEnum):
    """Enum class for the backbone-dihedral cluster (Modi & Dunbrack, PNAS 2019).

    The AF2 active-model set uses ``BLAminus``/``ABAminus``; the remaining clusters are
    included for forward compatibility.
    """

    BLAminus = "BLAminus"
    BLAplus = "BLAplus"
    ABAminus = "ABAminus"
    BLBminus = "BLBminus"
    BLBplus = "BLBplus"
    BLBtrans = "BLBtrans"
    BABtrans = "BABtrans"
    BBAminus = "BBAminus"


class SNC(StrEnum):
    """Enum class for the KinCoRe SNC (spine/salt-bridge) state label.

    Values observed in the AF2 active-model set; extend if a future release adds more.
    """

    SNCiii = "SNCiii"
    SNCiin = "SNCiin"
    SNCnii = "SNCnii"
    SNCoii = "SNCoii"


class KinHub(BaseModel):
    """Pydantic model for KinHub information."""

    model_config = ConfigDict(use_enum_values=True)

    hgnc_name: str | None = None
    kinase_name: str | None = None
    manning_name: str
    xname: str
    group: Group
    family: Family


class UniProt(BaseModel):
    """Pydantic model for UniProt information."""

    header: str
    canonical_seq: SeqUniProt
    phospho_sites: list[int] | None = None
    phospho_evidence: list[set[str]] | None = None
    phospho_description: list[str] | None = None


class KLIFS(BaseModel):
    """Pydantic model for KLIFS information."""

    model_config = ConfigDict(use_enum_values=True)

    gene_name: str
    name: str
    full_name: str
    group: Group
    family: Family
    iuphar: int
    kinase_id: int
    pocket_seq: SeqKLIFS | None = None


class Pfam(BaseModel):
    """Pydantic model for Pfam information."""

    model_config = ConfigDict(use_enum_values=True)

    domain_name: KinaseDomainName
    start: int
    end: int
    protein_length: int
    pfam_accession: str
    in_alphafold: bool


class KinCoReSeqSource(str, Enum):
    """KinCoRe kinase-domain sequence source, in fallback priority order (latest first)."""

    GIZZIO_2026 = (
        "Gizzio-Dunbrack_2026"  # kinasedomainfasta.tar.gz (current AF2 active models)
    )
    FAEZOV_2023 = "Faezov-Dunbrack_2023"  # AF2-active.fasta (earlier AF2 active models)
    MODI_2019 = "Modi-Dunbrack_2019"  # Human-PK.fasta (no active-state structure)


class KinCoReStructureSource(str, Enum):
    """KinCoRe active-state structure source, in fallback priority order (latest first)."""

    GIZZIO_2026 = "Gizzio-Dunbrack_2026"  # AF2_Active_Models_v2.zip (current)
    FAEZOV_2023 = (
        "Faezov-Dunbrack_2023"  # Kincore_AlphaFold2_ActiveHumanCatalyticKinases (v1)
    )


class Provenance(BaseModel):
    """Source provenance for a derived record (dataset name, version, citation, query date)."""

    name: str  # dataset/archive/file name
    version: str | None = None  # e.g. "v1"
    citation: str | None = None  # publication or DOI
    query_date: str | None = (
        None  # ISO date: download date (re-fetched) or file mtime (local)
    )


class SASA(BaseModel):
    """KLIFS-pocket solvent accessibility over one kinase-domain structure.

    Nested on the structure it was computed over (a KinCoRe active-state CIF or an AlphaFold DB
    model), so a kinase can carry both a KinCoRe and an AF2 SASA for comparison; the parent
    structure determines the provenance. Absolute SASA (Å^2) and relative solvent accessibility
    (RSA) are computed **heavy-atom** (``include_hydrogens=False``) so the two structure sources
    are directly comparable: KinCoRe v2 CIFs carry explicit hydrogens (which are stripped before
    the calculation), whereas AlphaFold DB CIFs are heavy-atom only. RSA normalizes SASA by the
    ``max_asa_reference`` maxima, which are themselves a heavy-atom reference (so relative
    accessibility is only defined heavy-atom). The ``method``/``probe_radius``/``n_points``/
    ``include_hydrogens``/``max_asa_reference`` fields record the methodology used.
    """

    sasa: dict[str, float | None]  # KLIFS region:idx -> absolute SASA (Å^2)
    rsa: dict[str, float | None]  # KLIFS region:idx -> relative solvent accessibility
    method: str  # SASA methodology (backend/algorithm)
    probe_radius: float
    n_points: int
    include_hydrogens: bool
    max_asa_reference: str  # reference maxima used to normalize rsa

    @field_validator("sasa", "rsa", mode="before")
    @classmethod
    def validate_klifs_sasa(
        cls,
        value: dict[str, float | None],
    ) -> dict[str, float | None]:
        """Fill all KLIFS positions on the SASA/RSA maps (see :func:`fill_missing_none`)."""
        return fill_missing_none(value, LIST_KLIFS_REGION)


class KinCoReFASTA(BaseModel):
    """Pydantic model for KinCoRe FASTA information."""

    model_config = ConfigDict(use_enum_values=True)

    seq: SeqUniProt
    group: Group
    hgnc: set[str]
    swissprot: str
    uniprot: SwissProtID | TrEMBLID
    start_md: int  # Modi-Dunbrack, 2019
    end_md: int  # Modi-Dunbrack, 2019
    length_md: int | None = None  # Modi-Dunbrack, 2019
    start_af2: int | None = None  # AF2 active state
    end_af2: int | None = None  # AF2 active state
    length_af2: int | None = None  # AF2 active state
    length_uniprot: int | None = None  # AF2 active state
    source: Provenance | None = (
        None  # sequence source provenance (see KinCoReSeqSource)
    )
    start: int | None = None  # fasta2uniprot
    end: int | None = None  # fasta2uniprot
    mismatch: list[int] | None = None  # fasta2uniprot


class KinCoReCIF(BaseModel):
    """Pydantic model for KinCoRe CIF information."""

    model_config = ConfigDict(use_enum_values=True)

    cif: dict[str, str | list[str]]
    group: Group
    hgnc: str
    # v1 (Kincore_AlphaFold2_ActiveHumanCatalyticKinases_v2) fields
    min_aloop_pLDDT: float | None = None
    template_source: TemplateSource | None = None
    msa_size: int | None = None
    msa_source: MSASource | None = None
    model_no: int | None = None
    # v2 (AF2_Active_Models_v2) fields
    model_confidence: float | None = None  # _ma_qa_metric_global mean pLDDT (0-1)
    dfg_conf: DFGConf | None = None
    dihedral: DihedralCluster | None = None
    snc: SNC | None = None
    af_id: str | None = (
        None  # KinCoRe active-model id: AF-<uniprot>-K{3,4}A (K4 = 2nd KD, A = active)
    )
    # calculated fields
    source: Provenance | None = (
        None  # structure source provenance (see KinCoReStructureSource)
    )
    sasa: SASA | None = (
        None  # KLIFS-pocket SASA over this KinCoRe active-state structure
    )
    start: int | None = None  # cif2uniprot
    end: int | None = None  # cif2uniprot
    mismatch: list[int] | None = None  # cif2uniprot


class MSA(BaseModel):
    """Per-domain slice of the Modi & Dunbrack (2019) structure-based kinase MSA.

    Stores this domain's row of the Human-PK alignment (497 canonical protein-kinase
    domains) as ordered aligned/unaligned regions, plus a per-aligned-column map to UniProt
    canonical indices -- mirroring the KLIFS2UniProt maps -- so the activation loop (DFG in
    the ALN block through APE in the ALC block) can be read in UniProt coordinates.
    """

    regions: dict[
        str, str
    ]  # ordered region label -> gapped MSA substring (17 aligned + 16 unaligned)
    region2uniprot: dict[
        str, int | None
    ]  # KLIFS-style "REGION:idx" ("B1N:001".."HI:229") -> UniProt index (None = gap)
    start: int | None = (
        None  # KD start in UniProt canonical coords (msa2uniprot; reconciled)
    )
    end: int | None = (
        None  # KD end in UniProt canonical coords (msa2uniprot; reconciled)
    )
    reconciled: bool = (
        False  # True if a local alignment was needed (isoform/numbering shift)
    )
    source: Provenance | None = None  # Human-PK-alignment provenance

    @field_validator("region2uniprot", mode="before")
    @classmethod
    def validate_region2uniprot(
        cls,
        value: dict[str, int | None],
    ) -> dict[str, int | None]:
        """Fill all aligned MSA positions on region2uniprot (see :func:`fill_missing_none`)."""
        return fill_missing_none(value, LIST_MSA_REGION)


class KinCoRe(BaseModel):
    """Pydantic model for KinCoRe information."""

    fasta: KinCoReFASTA | None = None
    cif: KinCoReCIF | None = None
    msa: MSA | None = None
    start: int | None = None  # fasta2cif
    end: int | None = None  # fasta2cif
    mismatch: list[int] | None = None  # fasta2cif


class AlphaFold(BaseModel):
    """Pydantic model for an AlphaFold DB structure sliced to the kinase domain."""

    cif: dict[str, str | list[str]]
    start: int | None = None  # kinase-domain slice bounds (UniProt, from adjudication)
    end: int | None = None
    entry_id: str
    uniprot_accession: str
    global_metric_value: float | None = None  # global mean pLDDT (0-100)
    model_created_date: str | None = None
    latest_version: int | None = None
    tool_used: str | None = None
    mismatch: list[int] | None = (
        None  # KD-slice positions differing from canonical UniProt
    )
    source: Provenance | None = None  # EBI AlphaFold DB provenance
    sasa: SASA | None = None  # KLIFS-pocket SASA over this AlphaFold structure


class KinaseInfoUniProt(BaseModel):
    """Pydantic model for kinase information at the level of the UniProt ID."""

    hgnc_name: str
    uniprot_id: SwissProtID | TrEMBLID
    uniprot: UniProt
    pfam: Pfam | None = None


class KinaseInfoKinaseDomain(BaseModel):
    """Pydantic model for kinase information at the level of the kinase domain."""

    uniprot_id: SwissProtIDSuffix | TrEMBLIDSuffix
    kinhub: KinHub | None = None
    klifs: KLIFS | None = None
    kincore: KinCoRe | None = None


class KinaseInfo(BaseModel):
    """Pydantic model for kinase information at the level of the kinase domain."""

    hgnc_name: str
    uniprot_id: SwissProtIDSuffix | TrEMBLIDSuffix
    uniprot: UniProt
    kinhub: KinHub | None = None
    klifs: KLIFS | None = None
    pfam: Pfam | None = None
    kincore: KinCoRe | None = None
    alphafold: AlphaFold | None = None
    KLIFS2UniProtIdx: dict[str, int | None] | None = None
    KLIFS2UniProtSeq: dict[str, str | None] | None = None

    @field_validator("KLIFS2UniProtIdx", mode="before")
    @classmethod
    def validate_klifs2uniprotidx(
        cls,
        value: dict[str, int | None] | None,
    ) -> dict[str, int | None] | None:
        """Fill all KLIFS pocket positions on KLIFS2UniProtIdx (see :func:`fill_missing_none`)."""
        return fill_missing_none(value, LIST_KLIFS_REGION)

    @field_validator("KLIFS2UniProtSeq", mode="before")
    @classmethod
    def validate_klifs2uniprotseq(
        cls,
        value: dict[str, str | None] | None,
    ) -> dict[str, str | None] | None:
        """Fill all KLIFS regions on KLIFS2UniProtSeq (see :func:`fill_missing_none`)."""
        return fill_missing_none(value, LIST_FULL_KLIFS_REGION)

    def extract_sequence_from_cif(self, bool_verbose: bool = False) -> str | None:
        """Extract sequence from CIF if available.

        Parameters
        ----------
        bool_verbose : bool, optional
            Whether to log verbose messages, by default False.

        Returns
        -------
        str | None
            The sequence from the CIF if available, otherwise None.
        """
        from mkt.schema.utils import extract_sequence_from_cif

        seq = extract_sequence_from_cif(self.kincore)
        if seq is None and bool_verbose:
            logger.info(f"No CIF sequence for {self.hgnc_name}")
        return seq

    def adjudicate_kd_sequence(self, bool_verbose: bool = False) -> str | None:
        """Adjudicate the kinase domain sequence as the canonical UniProt KD slice.

        Returns ``canonical_seq[start - 1 : end]`` using the adjudicated kinase-domain
        bounds (:meth:`adjudicate_kd_start` / :meth:`adjudicate_kd_end`) so the sequence is
        1-to-1 with those bounds. This also makes the sequence match the KD-sliced AlphaFold
        structure by construction (both are canonical UniProt over ``[start, end]``).

        Parameters
        ----------
        bool_verbose : bool, optional
            Whether to log verbose messages, by default False.

        Returns
        -------
        str | None
            The canonical kinase-domain sequence if bounds are available, otherwise None.
        """
        start = self.adjudicate_kd_start(bool_verbose=bool_verbose)
        end = self.adjudicate_kd_end(bool_verbose=bool_verbose)
        if start is None or end is None:
            if bool_verbose:
                logger.info(f"No kinase domain sequence found for {self.hgnc_name}")
            return None
        return self.uniprot.canonical_seq[start - 1 : end]

    def _klifs_uniprot_idx_bounds(self) -> tuple[int, int] | None:
        """Return the min and max non-None UniProt indices in KLIFS2UniProtIdx.

        Returns
        -------
        tuple[int, int] | None
            The (min, max) UniProt indices spanned by the KLIFS pocket if any
            are available, otherwise None.
        """
        if self.KLIFS2UniProtIdx is None:
            return None
        list_idx = [idx for idx in self.KLIFS2UniProtIdx.values() if idx is not None]
        if len(list_idx) == 0:
            return None
        return min(list_idx), max(list_idx)

    def _reconcile_kd_bound_with_klifs(
        self,
        bound: int,
        klifs_bound: int,
        is_start: bool,
        int_max_gap: float,
        bool_verbose: bool,
    ) -> int | None:
        """Reconcile an adjudicated kinase domain bound with the KLIFS pocket.

        The KLIFS pocket should fall within the kinase domain, i.e. the minimum
        KLIFS index should be >= the kinase domain start and the maximum KLIFS
        index should be <= the kinase domain end. When this is violated, the gap
        is compared against ``int_max_gap``: gaps within the cutoff expand the
        bound to the KLIFS index, larger gaps return None since the mapping is
        treated as unreliable.

        A finite cutoff was originally required because the MTOR kinase domain
        was incorrectly annotated by Pfam (the "Serine/threonine-protein kinase
        mTOR domain" region rather than the catalytic PI3/4-kinase domain), which
        produced a spurious multi-hundred-residue gap. Once the atypical families
        were re-annotated (PIKKs, PI3/4-kinases, etc.), the remaining large gaps
        were found to be genuine kinase-domain inserts missed by Pfam but present
        in KLIFS, so ``int_max_gap`` now defaults to ``float("inf")`` (no cutoff)
        and the KLIFS index is trusted as the better-annotated bound.

        Parameters
        ----------
        bound : int
            The adjudicated kinase domain bound (start or end).
        klifs_bound : int
            The corresponding KLIFS pocket bound (min for start, max for end).
        is_start : bool
            Whether ``bound`` is the kinase domain start (True) or end (False).
        int_max_gap : float
            Maximum allowed gap between the kinase domain bound and the KLIFS
            bound before the bound is treated as unreliable and None is returned.
        bool_verbose : bool
            Whether to log verbose messages.

        Returns
        -------
        int | None
            The reconciled bound, expanded to the KLIFS index when the gap is
            within ``int_max_gap``, otherwise None.
        """
        str_bound = "start" if is_start else "end"
        # violation is min < start (start) or max > end (end)
        gap = bound - klifs_bound if is_start else klifs_bound - bound
        if gap <= 0:
            return bound
        if gap <= int_max_gap:
            if bool_verbose:
                logger.info(
                    f"KLIFS pocket {str_bound} ({klifs_bound}) extends past kinase "
                    f"domain {str_bound} ({bound}) for {self.hgnc_name} by {gap}; "
                    f"expanding {str_bound} to {klifs_bound}."
                )
            return klifs_bound
        logger.warning(
            f"Kinase domain {str_bound} found for {self.hgnc_name} but KLIFS pocket "
            f"{str_bound} ({klifs_bound}) gap is {gap} (larger than cut-off "
            f"{int_max_gap}); returning None."
        )
        return None

    def adjudicate_kd_start(
        self, int_max_gap: float = float("inf"), bool_verbose: bool = False
    ) -> int | None:
        """Adjudicate kinase domain start based on available data.

        Parameters
        ----------
        int_max_gap : float, optional
            Maximum allowed gap between the kinase domain start and the minimum
            KLIFS pocket index before the start is treated as unreliable and None
            is returned, by default ``float("inf")`` (no cutoff; the KLIFS index
            is always trusted). See :meth:`_reconcile_kd_bound_with_klifs` for why
            the historical finite cutoff was relaxed.
        bool_verbose : bool, optional
            Whether to log verbose messages, by default False.

        Returns
        -------
        int | None
            The start of the kinase domain if available, otherwise None.
        """
        # priority: KinCoRe CIF > KinCoRe FASTA > Dunbrack MSA > Pfam. Gate on the bound
        # itself (not on ``kincore``) so an MSA-only KinCoRe (cif/fasta None) still falls
        # through to Pfam rather than short-circuiting to None.
        start = (
            rgetattr(self, "kincore.cif.start")
            or rgetattr(self, "kincore.fasta.start")
            or rgetattr(self, "kincore.msa.start")
        )
        if start is None and self.pfam is not None:
            start = self.pfam.start
        if start is None:
            if bool_verbose:
                logger.info(
                    f"No kinase domain sequence start found for {self.hgnc_name}"
                )
            return None

        bounds = self._klifs_uniprot_idx_bounds()
        if start is not None and bounds is not None:
            start = self._reconcile_kd_bound_with_klifs(
                bound=start,
                klifs_bound=bounds[0],
                is_start=True,
                int_max_gap=int_max_gap,
                bool_verbose=bool_verbose,
            )
        return start

    def adjudicate_kd_end(
        self, int_max_gap: float = float("inf"), bool_verbose: bool = False
    ) -> int | None:
        """Adjudicate kinase domain end based on available data.

        Parameters
        ----------
        int_max_gap : float, optional
            Maximum allowed gap between the kinase domain end and the maximum
            KLIFS pocket index before the end is treated as unreliable and None
            is returned, by default ``float("inf")`` (no cutoff; the KLIFS index
            is always trusted). See :meth:`_reconcile_kd_bound_with_klifs` for why
            the historical finite cutoff was relaxed.
        bool_verbose : bool, optional
            Whether to log verbose messages, by default False.

        Returns
        -------
        int | None
            The end of the kinase domain if available, otherwise None.
        """
        # priority: KinCoRe CIF > KinCoRe FASTA > Dunbrack MSA > Pfam (see adjudicate_kd_start)
        end = (
            rgetattr(self, "kincore.cif.end")
            or rgetattr(self, "kincore.fasta.end")
            or rgetattr(self, "kincore.msa.end")
        )
        if end is None and self.pfam is not None:
            end = self.pfam.end
        if end is None:
            if bool_verbose:
                logger.info(f"No kinase domain sequence end found for {self.hgnc_name}")
            return None

        bounds = self._klifs_uniprot_idx_bounds()
        if end is not None and bounds is not None:
            end = self._reconcile_kd_bound_with_klifs(
                bound=end,
                klifs_bound=bounds[1],
                is_start=False,
                int_max_gap=int_max_gap,
                bool_verbose=bool_verbose,
            )
        return end

    def adjudicate_APE(self) -> list[int | None] | None:
        """Return the APE-motif UniProt indices (Ala, Pro, Glu) from the Dunbrack MSA.

        Reads ``kincore.msa.region2uniprot`` at the APE-motif positions (:data:`LIST_MSA_APE`,
        the ALC-block Ala/Pro/Glu; the Glu is the end-of-activation-loop anchor). Returns None
        when the motif is entirely unmapped -- either no MSA is stored, or the kinase lacks the
        APE motif with all three positions gapped (e.g. HASPIN, PAN3, PEAK3, RNASEL, PLK5) --
        mirroring the all-None molecular-brake triad check in
        :meth:`return_molecular_brake_residues`.

        Returns
        -------
        list[int | None] | None
            UniProt indices of the APE motif ``[Ala, Pro, Glu]`` (an element may be None if
            only part of the motif is gapped), or None if no APE motif is present.
        """
        region2uniprot = rgetattr(self, "kincore.msa.region2uniprot")
        if region2uniprot is None:
            return None
        list_idx = [region2uniprot.get(key) for key in LIST_MSA_APE]
        if all(idx is None for idx in list_idx):
            return None
        return list_idx

    def adjudicate_group(self, bool_verbose: bool = False) -> str | None:
        """Adjudicate group based on available data.

        Parameters
        ----------
        bool_verbose : bool, optional
            Whether to log verbose messages, by default False.

        Returns
        -------
        str | None
            The group of the kinase if available, otherwise None.
        """
        list_attr = ["kincore.fasta.group", "kinhub.group", "klifs.group"]

        for attr in list_attr:
            group = rgetattr(self, attr)
            if group is not None:
                return group

        if bool_verbose:
            logger.info(f"No group found for {self.hgnc_name}")
        return None

    def is_lipid_kinase(self) -> bool:
        """Return boolean if a lipid kinase.

        Returns
        -------
        bool
            Whether or not is a lipid kinase
        """
        str_hgnc = self.hgnc_name.split("_")[0]

        bool1 = str_hgnc.startswith("PI")
        bool2 = not (str_hgnc.startswith("PIM") or str_hgnc.startswith("PIN"))
        # protein kinase
        # previously included "PI4KAP1" and "PI4KAP2" since pseudogenes but they're lipid
        bool3 = not (str_hgnc in ["PIK3R4"])

        if bool1 and bool2 and bool3:
            return True
        else:
            return False

    def is_pseudogene(self) -> bool:
        """Return boolean if a pseudogene.

        Returns
        -------
        bool
            Whether or not is a pseudogene
        """
        from mkt.schema.utils import rgetattr

        for attr, str_attr in [
            ("uniprot.header", "putative"),
            ("klifs.full_name", "pseudogene"),
        ]:
            val = rgetattr(self, attr)
            if val is not None and str_attr in val.lower():
                return True

        return False

    def is_pseudokinase(self) -> bool:
        """Return boolean if a (predicted) pseudokinase.

        Predicts catalytic deficiency from the KLIFS pocket by testing the three
        canonical catalytic residues -- the VAIK beta3 lysine (III:17), the HRD
        catalytic aspartate (c.l:70) and the DFG aspartate (xDFG:81); a kinase missing
        any one is called a pseudokinase. The catalytic lysine may instead sit in beta2
        (II:13) in the WNK family, which is accepted as present. A KinCoRe active-state
        CIF takes precedence over everything else: it marks an experimentally/AF2-
        validated catalytically active kinase, so such a kinase is never a pseudokinase.
        Two hand-curated overrides then correct the known failure modes of the heuristic
        (see ``LIST_PSEUDOKINASE_TRIAD_INTACT`` and
        ``LIST_PSEUDOKINASE_HEURISTIC_FALSE_POSITIVE`` in ``mkt.schema.constants`` for
        membership and citations). Returns False when no KLIFS pocket is available.

        Returns
        -------
        bool
            Whether or not is a predicted pseudokinase
        """
        from mkt.schema.constants import (
            LIST_KLIFS_REGION,
            LIST_PSEUDOKINASE_HEURISTIC_FALSE_POSITIVE,
            LIST_PSEUDOKINASE_TRIAD_INTACT,
            STR_KLIFS_BETA2_LYSINE,
            STR_KLIFS_BETA3_LYSINE,
            STR_KLIFS_CATALYTIC_ASP,
            STR_KLIFS_DFG_ASP,
        )

        # a KinCoRe active-state CIF marks a catalytically active kinase, so it is never
        # a pseudokinase -- this overrides both the curated lists and the heuristic
        if self.kincore is not None and self.kincore.cif is not None:
            return False

        # curated literature overrides take precedence over the sequence heuristic
        if self.hgnc_name in LIST_PSEUDOKINASE_TRIAD_INTACT:
            return True
        if self.hgnc_name in LIST_PSEUDOKINASE_HEURISTIC_FALSE_POSITIVE:
            return False

        # cannot assess catalytic residues without a pocket
        if self.klifs is None or self.klifs.pocket_seq is None:
            return False

        pocket = self.klifs.pocket_seq

        def _residue(label):
            return pocket[LIST_KLIFS_REGION.index(label)]

        # the catalytic lysine is normally in beta3 (VAIK); the WNK family relocates it
        # to beta2 ("With No K [in beta3]"), so accept a lysine at either position
        has_lysine = (
            _residue(STR_KLIFS_BETA3_LYSINE) == "K"
            or _residue(STR_KLIFS_BETA2_LYSINE) == "K"
        )
        has_catalytic_asp = _residue(STR_KLIFS_CATALYTIC_ASP) == "D"
        has_dfg_asp = _residue(STR_KLIFS_DFG_ASP) == "D"

        # a pseudokinase is missing at least one of the three catalytic residues
        return not (has_lysine and has_catalytic_asp and has_dfg_asp)

    def return_molecular_brake_residues(self) -> dict[str, str | None] | None:
        """Return this kinase's residues at the molecular brake KLIFS positions.

        The molecular brake is a network of conserved residues in the KLIFS pocket
        (see ``DICT_MOLECULAR_BRAKE`` in ``mkt.schema.constants`` for the region:idx
        labels and their canonical identities). This reads the residue at each of
        those positions from the KLIFS-to-UniProt index mapping, i.e.
        ``canonical_seq[KLIFS2UniProtIdx[region:idx] - 1 + offset]``, where a per-label
        offset is encoded as a trailing signed integer on the ``DICT_MOLECULAR_BRAKE``
        key (e.g. ``"VIII:79-1"`` -> region:idx ``"VIII:79"`` with offset -1, since the
        brake lysine sits one residue N-terminal to its VIII:79 KLIFS-aligned position).

        Returns
        -------
        dict[str, str | None] | None
            Dictionary mapping each molecular brake label (region:idx with its optional
            offset) to the residue found at that position in this kinase, or None where
            the position is unmapped. Returns None entirely when no KLIFS pocket mapping
            is available (``KLIFS2UniProtIdx`` is None).
        """
        from mkt.schema.constants import DICT_MOLECULAR_BRAKE

        if self.KLIFS2UniProtIdx is None:
            return None

        seq = self.uniprot.canonical_seq
        dict_residues: dict[str, str | None] = {}
        for label in DICT_MOLECULAR_BRAKE:
            # split off a trailing signed offset ("-n"/"+n") to recover the KLIFS
            # region:idx key, then subtract/add it back to the mapped UniProt index
            klifs_key, offset = label, 0
            for sign, mult in (("-", -1), ("+", 1)):
                base, sep, num = label.rpartition(sign)
                if sep and num.isdigit():
                    klifs_key, offset = base, mult * int(num)
                    break
            idx = self.KLIFS2UniProtIdx.get(klifs_key)
            if idx is None:
                dict_residues[label] = None
                continue
            dict_residues[label] = seq[idx - 1 + offset]
        return dict_residues

    def check_molecular_brake_against_canonical(
        self,
    ) -> tuple[bool, bool, bool] | None:
        """Check this kinase's molecular brake residues against the canonical identities.

        Compares the residue at each molecular brake position (see
        ``return_molecular_brake_residues``) against the conserved canonical residue in
        ``DICT_MOLECULAR_BRAKE``. An unmapped position (None) is treated as not matching.

        Returns
        -------
        tuple[bool, bool, bool] | None
            One boolean per molecular brake position -- in ``DICT_MOLECULAR_BRAKE`` order
            -- indicating whether this kinase's residue matches the canonical identity.
            Returns None when no KLIFS pocket mapping is available.
        """
        from mkt.schema.constants import DICT_MOLECULAR_BRAKE

        dict_residues = self.return_molecular_brake_residues()
        if dict_residues is None:
            return None

        return tuple(
            dict_residues[label] == canonical
            for label, canonical in DICT_MOLECULAR_BRAKE.items()
        )
