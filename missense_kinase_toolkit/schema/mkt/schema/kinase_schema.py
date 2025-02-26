import logging
from enum import Enum
from itertools import chain

from pydantic import BaseModel, ConfigDict, constr, field_validator
from strenum import StrEnum

logger = logging.getLogger(__name__)


LIST_PFAM_KD = [
    "Protein kinase domain",
    "Protein tyrosine and serine/threonine kinase",
]
"""list[str]: List of Pfam kinase domain names."""

LIST_FULL_KLIFS_REGION = [
    "I",
    "g.l",
    "II",
    "II:III",
    "III",
    "III:αC",
    "αC",
    "b.l_1",
    "b.l_intra",
    "b.l_2",
    "IV",
    "IV:V",
    "V",
    "GK",
    "hinge",
    "hinge:linker",
    "linker_1",
    "linker_intra",
    "linker_2",
    "αD",
    "αD:αE",
    "αE",
    "αE:VI",
    "VI",
    "c.l",
    "VII",
    "VII:VIII",
    "VIII",
    "xDFG",
    "a.l",
]
"""list[str]: List of KLIFS region, including intra and inter regions in order."""

# start/end and colors courtesy of OpenCADD
DICT_POCKET_KLIFS_REGIONS = {
    "I": {
        "start": 1,
        "end": 3,
        "contiguous": True,
        "color": "khaki",
    },
    "g.l": {
        "start": 4,
        "end": 9,
        "contiguous": True,
        "color": "green",
    },
    "II": {
        "start": 10,
        "end": 13,
        "contiguous": True,
        "color": "khaki",
    },
    "III": {
        "start": 14,
        "end": 19,
        "contiguous": False,
        "color": "khaki",
    },
    "αC": {
        "start": 20,
        "end": 30,
        "contiguous": True,
        "color": "red",
    },
    "b.l": {
        "start": 31,
        "end": 37,
        "contiguous": True,
        "color": "green",
    },
    "IV": {
        "start": 38,
        "end": 41,
        "contiguous": False,
        "color": "khaki",
    },
    "V": {
        "start": 42,
        "end": 44,
        "contiguous": True,
        "color": "khaki",
    },
    "GK": {
        "start": 45,
        "end": 45,
        "contiguous": True,
        "color": "orange",
    },
    "hinge": {
        "start": 46,
        "end": 48,
        "contiguous": True,
        "color": "magenta",
    },
    "linker": {
        "start": 49,
        "end": 52,
        "contiguous": True,
        "color": "cyan",
    },
    "αD": {
        "start": 53,
        "end": 59,
        "contiguous": False,
        "color": "red",
    },
    "αE": {
        "start": 60,
        "end": 64,
        "contiguous": True,
        "color": "red",
    },
    "VI": {
        "start": 65,
        "end": 67,
        "contiguous": True,
        "color": "khaki",
    },
    "c.l": {
        "start": 68,
        "end": 75,
        "contiguous": True,
        "color": "darkorange",
    },
    "VII": {
        "start": 76,
        "end": 78,
        "contiguous": False,
        "color": "khaki",
    },
    "VIII": {
        "start": 79,
        "end": 79,
        "contiguous": True,
        "color": "khaki",
    },
    "xDFG": {
        "start": 80,
        "end": 83,
        "contiguous": True,
        "color": "cornflowerblue",
    },
    "a.l": {
        "start": 84,
        "end": 85,
        "contiguous": False,
        "color": "cornflowerblue",
    },
}
"""dict[str, dict[str, int | bool | str]]: Mapping KLIFS pocket region to start and end indices, \
    boolean denoting if subsequent regions are contiguous, and colors."""

LIST_KLIFS_REGION = list(
    chain(
        *[
            [f"{key}:{i}" for i in range(val["start"], val["end"] + 1)]
            for key, val in DICT_POCKET_KLIFS_REGIONS.items()
        ]
    )
)
"""list[str]: List of string of all KLIFS pocket regions in format region:idx."""


class Group(StrEnum):
    """Enum class for kinase groups."""

    AGC = "AGC"  # Protein Kinase A, G, and C families
    Atypical = "Atypical"  # Atypical protein kinases
    CAMK = "CAMK"  # Calcium/calmodulin-dependent protein kinase family
    CK1 = "CK1"  # Casein kinase 1 family
    CMGC = "CMGC"  # Cyclin-dependent kinase, Mitogen-activated protein kinase, Glycogen synthase kinase, and CDK-like kinase families
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
    Jak = "Jak"
    Jakb = "Jakb"
    Other = "Other"
    Null = None


KinaseDomainName = StrEnum(
    "KinaseDomainName", {"KD" + str(idx + 1): kd for idx, kd in enumerate(LIST_PFAM_KD)}
)
SeqUniProt = constr(pattern=r"^[ACDEFGHIKLMNPQRSTVWXY]+$")
"""Pydantic model for UniProt sequence constraints."""
SeqKLIFS = constr(pattern=r"^[ACDEFGHIKLMNPQRSTVWY\-]{85}$")
"""Pydantic model for KLIFS pocket sequence constraints."""
UniProtID = constr(pattern=r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$")
"""Pydantic model for UniProt ID constraints."""


class KinHub(BaseModel):
    """Pydantic model for KinHub information."""

    model_config = ConfigDict(use_enum_values=True)

    kinase_name: str
    manning_name: list[str]
    xname: list[str]
    group: list[Group]
    family: list[Family]


class UniProt(BaseModel):
    """Pydantic model for UniProt information."""

    canonical_seq: SeqUniProt


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


class KinCore(BaseModel):
    """Pydantic model for KinCore information."""

    seq: SeqUniProt
    start: int
    end: int
    mismatch: list[int] | None = None


class KinaseInfo(BaseModel):
    """Pydantic model for kinase information."""

    hgnc_name: str
    uniprot_id: UniProtID
    kinhub: KinHub
    uniprot: UniProt
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
