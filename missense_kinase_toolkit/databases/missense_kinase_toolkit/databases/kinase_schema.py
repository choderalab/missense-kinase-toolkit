from dataclasses import dataclass
from enum import Enum
from typing import List

from pydantic import BaseModel, constr


class Group(str, Enum):
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


class Family(str, Enum):
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
    Other = "Other"


UniProtAlphabet = constr(pattern="^[ACDEFGHIKLMNPQRSTVWXY]+$")
KLIFSAlphabet = constr(pattern=r"^[ACDEFGHIKLMNPQRSTVWY\-]+$")


class Kinase(BaseModel):
    """Pydantic model for kinase information."""

    hgnc_name: str
    uniprot_id: str
    kinase_name: str
    manning_name: list[str]
    xname: list[str]
    group: list[Group]
    family: list[Family]
    uniprot_seq: UniProtAlphabet
    klifs_pocket: KLIFSAlphabet
    pfam_id: str
    pfam_start: int
    pfam_end: int
    klifs_pocket_seq: str
    klifs_pocket_start: int
    klifs_pocket_end: int
