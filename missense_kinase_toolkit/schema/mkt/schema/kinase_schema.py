import logging
from enum import Enum

from mkt.schema.constants import LIST_FULL_KLIFS_REGION, LIST_KLIFS_REGION, LIST_PFAM_KD
from mkt.schema.utils import rgetattr
from pydantic import BaseModel, ConfigDict, Field, constr, field_validator
from strenum import StrEnum

logger = logging.getLogger(__name__)


class Group(StrEnum):
    """Enum class for kinase groups."""

    AGC = "AGC"  # Protein Kinase A, G, and C families
    Atypical = "Atypical"  # Atypical protein kinases
    CAMK = "CAMK"  # Calcium/calmodulin-dependent protein kinase family
    CK1 = "CK1"  # Casein kinase 1 family
    CMGC = "CMGC"  # Cyclin-dependent kinase, Mitogen-activated protein kinase, Glycogen synthase kinase, and CDK-like kinase families
    NEK = "NEK"  # NIMA (Never in Mitosis Gene A)-related kinase family - KinCore treats as group
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


class KinCoreFASTA(BaseModel):
    """Pydantic model for KinCore FASTA information."""

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
    source_file: str
    start: int | None = None  # fasta2uniprot
    end: int | None = None  # fasta2uniprot
    mismatch: list[int] | None = None  # fasta2uniprot


class KinCoreCIF(BaseModel):
    """Pydantic model for KinCore CIF information."""

    model_config = ConfigDict(use_enum_values=True)

    cif: dict[str, str | list[str]]
    group: Group
    hgnc: str
    min_aloop_pLDDT: float
    template_source: TemplateSource
    msa_size: int
    msa_source: MSASource
    model_no: int = Field(..., ge=1, lt=6)
    start: int | None = None  # cif2uniprot
    end: int | None = None  # cif2uniprot
    mismatch: list[int] | None = None  # cif2uniprot


class KinCore(BaseModel):
    """Pydantic model for KinCore information."""

    fasta: KinCoreFASTA
    cif: KinCoreCIF | None = None
    start: int | None = None  # fasta2cif
    end: int | None = None  # fasta2cif
    mismatch: list[int] | None = None  # fasta2cif


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
    kincore: KinCore | None = None


class KinaseInfo(BaseModel):
    """Pydantic model for kinase information at the level of the kinase domain."""

    hgnc_name: str
    uniprot_id: SwissProtIDSuffix | TrEMBLIDSuffix
    uniprot: UniProt
    kinhub: KinHub | None = None
    klifs: KLIFS | None = None
    pfam: Pfam | None = None
    kincore: KinCore | None = None
    KLIFS2UniProtIdx: dict[str, int | None] | None = None
    KLIFS2UniProtSeq: dict[str, str | None] | None = None

    @field_validator("KLIFS2UniProtIdx", mode="before")
    @classmethod
    def validate_klifs2uniprotidx(
        cls,
        value: dict[str, int | None] | None,
    ) -> dict[str, int | None] | None:
        """Validate KLIFS2UniProtIdx dictionary to include all regions since TOML doesn't save None.

        Parameters
        ----------
        value : dict[str, int | None]
            Dictionary mapping KLIFS residue to UniProt indices.

        Returns
        -------
        dict[str, int | None]
            Dictionary mapping KLIFS residue to UniProt indices.
        """
        dict_temp = dict.fromkeys(LIST_KLIFS_REGION, None)

        if value is not None:
            for key, val in value.items():
                dict_temp[key] = val
            return dict_temp
        else:
            return None

    @field_validator("KLIFS2UniProtSeq", mode="before")
    @classmethod
    def validate_klifs2uniprotseq(
        cls,
        value: dict[str, str | None] | None,
    ) -> dict[str, str | None] | None:
        """Validate KLIFS2UniProtSeq dictionary to include all regions since TOML doesn't save None.

        Parameters
        ----------
        value : dict[str, str | None]
            Dictionary mapping KLIFS residue to UniProt residue.

        Returns
        -------
        dict[str, str | None]
            Dictionary mapping KLIFS residue to UniProt residue.
        """
        dict_temp = dict.fromkeys(LIST_FULL_KLIFS_REGION, None)

        if value is not None:
            for key, val in value.items():
                dict_temp[key] = val
            return dict_temp
        else:
            return None

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
        key_seq = "_entity_poly.pdbx_seq_one_letter_code"

        try:
            if self.kincore is not None and self.kincore.cif is not None:
                return self.kincore.cif.cif[key_seq][0].replace("\n", "")
            else:
                if bool_verbose:
                    logger.info(f"No CIF sequence for {self.hgnc_name}")
                return None
        except Exception as e:
            if bool_verbose:
                logger.info(f"No Kincore entry for {self.hgnc_name}: {e}")
            return None

    def adjudicate_kd_sequence(self, bool_verbose: bool = False) -> str | None:
        """Adjudicate kinase domain sequence based on available data.

        Parameters
        ----------
        bool_verbose : bool, optional
            Whether to log verbose messages, by default False.

        Returns
        -------
        str | None
            The kinase domain sequence if available, otherwise None.
        """
        if self.kincore is not None:
            seq = self.extract_sequence_from_cif(bool_verbose=bool_verbose)
            if seq is not None:
                return seq
            else:
                # all non-None entries will have fastas
                return self.kincore.fasta.seq
        elif self.pfam is not None:
            return self.uniprot.canonical_seq[self.pfam.start - 1 : self.pfam.end]
        else:
            if bool_verbose:
                logger.info(f"No kinase domain sequence found for {self.hgnc_name}")
            return None

    def adjudicate_kd_start(self, bool_verbose: bool = False) -> int | None:
        """Adjudicate kinase domain start based on available data.

        Parameters
        ----------
        bool_verbose : bool, optional
            Whether to log verbose messages, by default False.

        Returns
        -------
        int | None
            The start of the kinase domain if available, otherwise None.
        """
        if self.kincore is not None:
            start = rgetattr(self, "kincore.cif.start") or rgetattr(
                self, "kincore.fasta.start"
            )
            return start
        elif self.pfam is not None:
            return self.pfam.start
        else:
            if bool_verbose:
                logger.info(
                    f"No kinase domain sequence start found for {self.hgnc_name}"
                )
            return None

    def adjudicate_kd_end(self, bool_verbose: bool = False) -> int | None:
        """Adjudicate kinase domain end based on available data.

        Parameters
        ----------
        bool_verbose : bool, optional
            Whether to log verbose messages, by default False.

        Returns
        -------
        int | None
            The end of the kinase domain if available, otherwise None.
        """
        if self.kincore is not None:
            end = rgetattr(self, "kincore.cif.end") or rgetattr(
                self, "kincore.fasta.end"
            )
            return end
        elif self.pfam is not None:
            return self.pfam.end
        else:
            if bool_verbose:
                logger.info(f"No kinase domain sequence end found for {self.hgnc_name}")
            return None

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

        if str_hgnc.startswith("PI") and not (
            str_hgnc.startswith("PIM") or str_hgnc.startswith("PIN")
        ):
            return True
        else:
            return False
