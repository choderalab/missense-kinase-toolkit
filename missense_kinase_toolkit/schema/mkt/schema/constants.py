"""Constants for KLIFS regions, Pfam kinase domains, and kinase groups/families.

Defines the canonical KLIFS region orderings, Pfam kinase-domain accessions, and the
controlled vocabularies for kinase groups and families referenced throughout the
schema and databases packages.
"""

import logging
from itertools import chain

logger = logging.getLogger(__name__)


LIST_PFAM_KD = [
    "Protein kinase domain",
    "Protein tyrosine and serine/threonine kinase",
    # ADCK1/2/5, COQ8A/B
    "ABC1 atypical kinase-like domain",
    # ALPK1/2/3, EEF2K, TRPM6/7
    "Alpha-kinase family",
    # PIKK (ATM, ATR, MTOR, PRKDC, SMG1, TRRAP), PI3/4K
    "Phosphatidylinositol 3- and 4-kinase",
    # PIP5K/PI4P5K
    "Phosphatidylinositol-4-phosphate 5-Kinase",
    # RIOK1/2/3
    "RIO domain",
    # GHKL kinases: BCKDK, PDK1-4
    "Histidine kinase-, DNA gyrase B-, and HSP90-like ATPase",
]
"""list[str]: List of Pfam kinase domain names, including lipid and atypical kinase
domains. Note MTOR's catalytic kinase domain is the C-terminal
"Phosphatidylinositol 3- and 4-kinase" region, not the regulatory
"Serine/threonine-protein kinase mTOR domain"."""

LIST_LEGACY_KINASES = [
    # GCK: hexokinase (sugar kinase, EC 2.7.1.2), out of scope
    "GCK",
    # BET family - historically contested Ser/Thr activity
    "BRD2",
    "BRD3",
    "BRD4",
    "BRDT",
    # E3 ligases/transcription co-factors
    "TRIM24",
    "TRIM28",
    "TRIM33",
    "TRIM66",
    # other bromodomain-containing proteins
    "BAZ1A",
    "BAZ1B",
    "TAF1",
    "TAF1L",
    # RhoGEF/GAPs
    "ABR",
    "BCR",
    # conflated with ATR in KLIFS as also sometimes abbreviated ATR
    "ANTXR1",
    # Ephrin ligand - not Ephrin
    "EFNA2",
    # reclassified as a winged-helix domain protein in 2024
    "WHR1",
    # not kinases - holdovers from Manning et al. 2002
    "BLVRA",
    "CERT1",
    "GTF2F1",
    "FASTK",
    "HSPB8",
]
"""list[str]: List of legacy (mostly Manning) kinases that should not be included in canonical kinases."""

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

# --- Modi & Dunbrack (2019) structure-based kinase-domain MSA ---
# Human-PK-alignment.fasta aligns 497 canonical protein-kinase domains in 17 conserved blocks
# (Hanks nomenclature) at fixed alignment-column ranges (Modi & Dunbrack, Sci Rep 2019, Table
# 2), separated by 16 unaligned insertion regions. Columns where the reference kinase Aurora A
# carries a residue are numbered continuously and KLIFS-style, giving 229 aligned positions
# "REGION:idx" ("B1N:001".."HI:229"); the rare insertion columns (Aurora A gapped) are dropped
# so positions stay consistent across kinases.
DICT_MSA_ALIGNED_REGION = {
    "B1N": (1, 4),
    "B1C": (21, 27),
    "B2": (44, 52),
    "B3": (93, 103),
    "HC": (140, 153),
    "B4": (180, 195),
    "B5": (420, 430),
    "HD": (445, 453),
    "HE": (939, 962),
    "CL": (1008, 1028),
    "ALN": (1331, 1351),
    "ALC": (1904, 1920),
    "HF": (1953, 1975),
    "FL": (1993, 1998),
    "HG": (2049, 2061),
    "HH": (2175, 2194),
    "HI": (2209, 2218),
}
"""dict[str, tuple[int, int]]: Aligned-block name -> (start, end) 1-based MSA column range \
(Modi & Dunbrack 2019, Table 2), in kinase-domain N->C order."""

SET_MSA_INSERTION_COL = frozenset({1015, 1016, 1017, 1018, 1019, 1351, 1957})
"""frozenset[int]: Aligned-block columns where Aurora A is gapped -- rare insertions present \
in only a few kinases (OTHER_STK16 in CL; one in ALN; a CAMK insertion in HF). Dropped from \
the reference numbering."""

_LIST_MSA_REF_COL = [
    (name, col)
    for name, (start, end) in DICT_MSA_ALIGNED_REGION.items()
    for col in range(start, end + 1)
    if col not in SET_MSA_INSERTION_COL
]

DICT_MSA_COL2LABEL = {
    col: f"{name}:{i:03d}" for i, (name, col) in enumerate(_LIST_MSA_REF_COL, start=1)
}
"""dict[int, str]: 1-based MSA column -> "REGION:idx" label over the 229 Aurora-reference \
columns (KLIFS-style continuous index)."""

LIST_MSA_REGION = list(DICT_MSA_COL2LABEL.values())
"""list[str]: The 229 aligned MSA positions in N->C order ("B1N:001".."HI:229")."""

LIST_MSA_APE = ["ALC:152", "ALC:153", "ALC:154"]
"""list[str]: APE-motif positions (Ala, Pro, Glu) as region2uniprot keys; the Glu \
(STR_MSA_APE) is the end-of-activation-loop anchor. All three unmapped (None) => no APE motif \
(e.g. HASPIN, PAN3, PEAK3, RNASEL, PLK5)."""

STR_MSA_APE = "ALC:154"
"""str: region2uniprot key of the APE-motif glutamate (end of the activation loop)."""

# --- pseudokinase catalytic-residue heuristic ---
# A (predicted) pseudokinase lacks at least one of the three canonical catalytic
# residues of the protein-kinase fold. We read these from the gapless 85-residue
# KLIFS pocket by their region:idx label (indexed via LIST_KLIFS_REGION):
#   - VAIK beta3 lysine       (III:17) -- orients the ATP alpha/beta phosphates
#   - HRD catalytic aspartate (c.l:70) -- the catalytic base
#   - DFG aspartate           (xDFG:81) -- chelates the Mg2+ ion
# Catalytic-residue definitions: Hanks & Hunter, FASEB J 1995; Taylor & Kornev,
# Trends Biochem Sci 2011. Pseudokinase concept/threshold (~10% of the kinome):
# Manning et al., Science 2002; Boudeau et al., Trends Cell Biol 2006; Murphy et
# al., Biochem J 2014; Kwon/Eyers et al., Sci Signal 2019.
STR_KLIFS_BETA3_LYSINE = "III:17"
"""str: KLIFS region:idx of the canonical VAIK beta3 catalytic lysine."""
STR_KLIFS_BETA2_LYSINE = "II:13"
"""str: KLIFS region:idx of the beta2 lysine used as the catalytic lysine by the WNK
("With No K [lysine]") family in place of the absent beta3 lysine -- Xu et al., J Biol
Chem 2000 (WNK1 lacks the subdomain-II lysine); Min et al., Structure 2004 (WNK1 Lys233
sits in beta2). Verified to rescue WNK1/2/3 (K at II:13) without rescuing the genuine
pseudokinases KSR1/2 or STRADA, which carry no beta2 lysine."""
STR_KLIFS_CATALYTIC_ASP = "c.l:70"
"""str: KLIFS region:idx of the HRD catalytic aspartate. Note the catalytic loop is
reverse-ordered in lipid/PIKK-like kinases (DRH rather than HRD), but the aspartate stays
at c.l:70 -- so this column is robust to that reversal."""
STR_KLIFS_DFG_ASP = "xDFG:81"
"""str: KLIFS region:idx of the DFG-motif aspartate."""
LIST_KLIFS_HRD_MOTIF = ["c.l:68", "c.l:69", "c.l:70"]
"""list[str]: KLIFS region:idx labels of the catalytic-loop HRD motif (His-Arg-Asp; the
aspartate at c.l:70 is the catalytic base). PIK/PIKK kinases read the loop in reverse (see
LIST_KLIFS_HRD_MOTIF_REVERSED)."""
LIST_KLIFS_HRD_MOTIF_REVERSED = ["c.l:72", "c.l:71", "c.l:70"]
"""list[str]: KLIFS region:idx labels of the HRD motif, in His-Arg-Asp order, for kinases
whose catalytic loop reads D-R-H (SET_FAMILY_HRD_REVERSED)."""
SET_FAMILY_HRD_REVERSED = {"PIK", "PIKK"}
"""set[str]: KinHub/KLIFS families (PI3K/PI4K and PIKK) whose catalytic loop reads D-R-H."""
STR_KLIFS_HRD_MOTIF = "HRD"
"""str: Canonical catalytic-loop motif (His-Arg-Asp)."""
LIST_KLIFS_DFG_MOTIF = ["xDFG:81", "xDFG:82", "xDFG:83"]
"""list[str]: KLIFS region:idx labels of the DFG motif (Asp-Phe-Gly; the aspartate at
xDFG:81 chelates the catalytic Mg2+)."""
STR_KLIFS_DFG_MOTIF = "DFG"
"""str: Canonical activation-segment start motif (Asp-Phe-Gly)."""

LIST_KLIFS_CATALYTIC = [
    STR_KLIFS_BETA3_LYSINE,
    STR_KLIFS_BETA2_LYSINE,
    *LIST_KLIFS_HRD_MOTIF,
    "c.l:71",
    "c.l:72",
    *LIST_KLIFS_DFG_MOTIF,
]
"""list[str]: KLIFS region:idx labels read by \
:meth:`mkt.schema.kinase_schema.KinaseInfo.return_catalytic_residues` -- the two candidate \
catalytic lysines, the HRD motif in either orientation (c.l:68-72) and the DFG motif, in \
N->C order."""

DICT_KLIFS2MSA_CATALYTIC = {
    "III:17": "B3:028",
    "II:13": "B2:017",
    "c.l:68": "CL:109",
    "c.l:69": "CL:110",
    "c.l:70": "CL:111",
    "c.l:71": "CL:112",
    "c.l:72": "CL:113",
    "xDFG:81": "ALN:129",
    "xDFG:82": "ALN:130",
    "xDFG:83": "ALN:131",
}
"""dict[str, str]: KLIFS region:idx -> Dunbrack MSA region2uniprot key at the \
LIST_KLIFS_CATALYTIC positions, read when a kinase has no KLIFS pocket. Precomputed so a \
single KinaseInfo never loads the corpus; kept in sync with \
:func:`mkt.schema.utils.return_catalytic_klifs2msa_dict` by the test suite."""

LIST_KLIFS_CATALYTIC_TRIAD = [
    STR_KLIFS_BETA3_LYSINE,
    STR_KLIFS_CATALYTIC_ASP,
    STR_KLIFS_DFG_ASP,
]
"""list[str]: The three canonical catalytic positions whose identities decide \
:meth:`mkt.schema.kinase_schema.KinaseInfo.is_pseudokinase` (the beta2 lysine at \
STR_KLIFS_BETA2_LYSINE substitutes for the beta3 lysine in the WNK family)."""

LIST_PSEUDOKINASE_TRIAD_INTACT = [
    "BUB1B",
    "ROR1",
    "ROR2",
    "RYK",
]
"""list[str]: Curated pseudokinases that retain an intact VAIK-K / HRD-D / DFG-D triad and
are therefore NOT caught by the catalytic-residue heuristic (false negatives); they are
catalytically dead for other reasons (degraded regulatory spine, glycine-rich loop, or
nucleotide binding). is_pseudokinase() force-returns True for these, unless the kinase has
a KinCoRe active-state CIF (which takes precedence and marks it catalytically active).

Citations:
  - BUB1B (BUBR1) -- a bona fide pseudokinase despite an intact catalytic triad:
    Suijkerbuijk et al., Dev Cell 2012; Murphy et al., Biochem J 2014.
  - ROR1, ROR2, RYK -- Wnt-receptor pseudokinases that retain catalytic residues but
    lack activity: Boudeau et al., Trends Cell Biol 2006; Reiterer et al., Trends Cell
    Biol 2014; Mendrola et al., Biochem Soc Trans 2013.

NOTE: PDIK1L and SBK3 were previously listed here (annotated pseudo on nucleotide-binding
grounds, Murphy et al., Biochem J 2014, but lower confidence) -- both now carry KinCoRe
active-state CIFs and are treated as catalytically active, so they were removed."""

LIST_PSEUDOKINASE_HEURISTIC_FALSE_POSITIVE = [
    "CAMKK1",
    "STYK1",
]
"""list[str]: Kinases the catalytic-residue heuristic flags as pseudokinases but that are
(debatably) catalytically active -- false positives held out for review. is_pseudokinase()
force-returns False for these. Status is genuinely contested in the literature.

Citations / rationale:
  - STYK1 (NOK, "Novel Oncogene with Kinase domain") -- fails only DFG-D (xDFG:81 = G);
    reported as an active oncogenic kinase by some and as catalytically deficient by
    others, i.e. unresolved: Reiterer et al., Trends Cell Biol 2014; Kung & Jura,
    Structure 2016.
  - CAMKK1 -- a well-established active Ca2+/calmodulin-dependent kinase kinase (Haribabu
    et al., EMBO J 1995) whose KLIFS pocket is anomalously degraded here (III:17=M,
    c.l:70=R, xDFG:81=A), most consistent with a pocket alignment/annotation artifact
    rather than true loss of catalysis.

NOTE (WNK4): WNK4 also trips the heuristic (no beta3 or beta2 lysine; III:17=C, II:13=R)
and is NOT rescued by the beta2-lysine alternative, unlike WNK1/2/3. It is not listed here
because it carries a KinCoRe active-state CIF, which is_pseudokinase() treats as
catalytically active (taking precedence over the heuristic)."""

DICT_KINASE_GROUP_COLORS = {
    "AGC": "#5B8DBE",  # Muted steel blue
    "Atypical": "#7A7A7A",  # Medium grey (kept similar)
    "CAMK": "#D4A574",  # Muted tan/sand
    "CK1": "#8B7355",  # Muted brown (replaces green)
    "CMGC": "#C17B7B",  # Muted rose/mauve (replaces red)
    "NEK": "#E5A672",  # Muted peach
    "Other": "#9B8AB8",  # Muted lavender
    "RGC": "#A67C52",  # Muted terracotta
    "STE": "#D39EB7",  # Muted dusty pink
    "TK": "#6BAFB8",  # Muted teal
    "TKL": "#B8AE6E",  # Muted gold/khaki
    "Lipid": "#8B6F84",  # Muted plum
    "Multiple": "#7FA08A",  # Muted sage (multi-domain gene, mixed groups)
}
"""dict[str, str]: Dictionary mapping kinase groups to colors.
Keys are kinase group names, and values are hex color codes.
This dictionary can be used to look up colors for kinase groups in visualizations.
"""

DICT_MOLECULAR_BRAKE = {
    "b.l:37": "N",
    "hinge:46": "E",
    "VIII:79-1": "K",
}
"""dict[str, str]: Dictionary mapping KLIFS pocket region:idx (with an optional index
offset) to the corresponding canonical molecular brake residue (N, E, K). The molecular brake
is a conserved triad that latches the active site into an autoinhibited state, characterized in
FGFR (Asn-Glu-Lys; Chen et al., Mol Cell 2007). This dictionary can be used to identify the
molecular brake residues in kinases based on their KLIFS pocket region and index. A key may
carry a trailing signed offset (``+n``/``-n``) applied to the KLIFS2UniProt-mapped index before
lookup: the brake lysine sits one residue N-terminal to its VIII:79 KLIFS-aligned position, so
``"VIII:79-1"`` applies a -1 offset (e.g. FGFR2 K641 maps to VIII:79 idx 642). A key with no
trailing sign (e.g. ``"b.l:37"``) uses no offset.
"""

DICT_CONSURF_GRADE_BANDS = {
    "Variable": (1, 3),
    "Intermediate": (4, 6),
    "Conserved": (7, 9),
}
"""dict[str, tuple[int, int]]: ConSurf conservation-grade bands as inclusive
(low, high) grade ranges. The nine ConSurf nonile grades collapse into three
qualitative bands -- variable (1-3), intermediate (4-6), conserved (7-9) -- used to
bracket-label the grade legend on the conservation dot heatmap.
"""

DICT_KINASE_GROUP = {
    "AAK1": "Other",
    "AATK": "TK",
    "ABL1": "TK",
    "ABL2": "TK",
    "ACVR1": "TKL",
    "ACVR1B": "TKL",
    "ACVR1C": "TKL",
    "ACVR2A": "TKL",
    "ACVR2B": "TKL",
    "ACVRL1": "TKL",
    "ADCK1": "Atypical",
    "ADCK2": "Atypical",
    "ADCK5": "Atypical",
    "AKT1": "AGC",
    "AKT2": "AGC",
    "AKT3": "AGC",
    "ALK": "TK",
    "ALPK1": "Atypical",
    "ALPK2": "Atypical",
    "ALPK3": "Atypical",
    "AMHR2": "TKL",
    "ANKK1": "TKL",
    "ARAF": "TKL",
    "ATM": "Atypical",
    "ATR": "Atypical",
    "AURKA": "CAMK",
    "AURKB": "CAMK",
    "AURKC": "CAMK",
    "AXL": "TK",
    "BCKDK": "Atypical",
    "BLK": "TK",
    "BMP2K": "Other",
    "BMPR1A": "TKL",
    "BMPR1B": "TKL",
    "BMPR2": "TKL",
    "BMX": "TK",
    "BRAF": "TKL",
    "BRSK1": "CAMK",
    "BRSK2": "CAMK",
    "BTK": "TK",
    "BUB1": "Other",
    "BUB1B": "Other",
    "CAMK1": "CAMK",
    "CAMK1D": "CAMK",
    "CAMK1G": "CAMK",
    "CAMK2A": "CAMK",
    "CAMK2B": "CAMK",
    "CAMK2D": "CAMK",
    "CAMK2G": "CAMK",
    "CAMK4": "CAMK",
    "CAMKK1": "CAMK",
    "CAMKK2": "CAMK",
    "CAMKV": "CAMK",
    "CASK": "CAMK",
    "CDC42BPA": "AGC",
    "CDC42BPB": "AGC",
    "CDC42BPG": "AGC",
    "CDC7": "Other",
    "CDK1": "CMGC",
    "CDK10": "CMGC",
    "CDK11A": "CMGC",
    "CDK11B": "CMGC",
    "CDK12": "CMGC",
    "CDK13": "CMGC",
    "CDK14": "CMGC",
    "CDK15": "CMGC",
    "CDK16": "CMGC",
    "CDK17": "CMGC",
    "CDK18": "CMGC",
    "CDK19": "CMGC",
    "CDK2": "CMGC",
    "CDK20": "CMGC",
    "CDK3": "CMGC",
    "CDK4": "CMGC",
    "CDK5": "CMGC",
    "CDK6": "CMGC",
    "CDK7": "CMGC",
    "CDK8": "CMGC",
    "CDK9": "CMGC",
    "CDKL1": "CMGC",
    "CDKL2": "CMGC",
    "CDKL3": "CMGC",
    "CDKL4": "CMGC",
    "CDKL5": "CMGC",
    "CHEK1": "CAMK",
    "CHEK2": "CAMK",
    "CHUK": "Other",
    "CILK1": "CMGC",
    "CIT": "AGC",
    "CLK1": "CMGC",
    "CLK2": "CMGC",
    "CLK3": "CMGC",
    "CLK4": "CMGC",
    "COQ8A": "Atypical",
    "COQ8B": "Atypical",
    "CSF1R": "TK",
    "CSK": "TK",
    "CSNK1A1": "CK1",
    "CSNK1A1L": "CK1",
    "CSNK1D": "CK1",
    "CSNK1E": "CK1",
    "CSNK1G1": "CK1",
    "CSNK1G2": "CK1",
    "CSNK1G3": "CK1",
    "CSNK2A1": "CMGC",
    "CSNK2A2": "CMGC",
    "CSNK2A3": "CMGC",
    "DAPK1": "CAMK",
    "DAPK2": "CAMK",
    "DAPK3": "CAMK",
    "DCLK1": "CAMK",
    "DCLK2": "CAMK",
    "DCLK3": "CAMK",
    "DDR1": "TK",
    "DDR2": "TK",
    "DMPK": "AGC",
    "DSTYK": "Other",
    "DYRK1A": "CMGC",
    "DYRK1B": "CMGC",
    "DYRK2": "CMGC",
    "DYRK3": "CMGC",
    "DYRK4": "CMGC",
    "EEF2K": "Atypical",
    "EGFR": "TK",
    "EIF2AK1": "Other",
    "EIF2AK2": "Other",
    "EIF2AK3": "Other",
    "EIF2AK4": "Other",
    "EIF2AK4_1": "Other",
    "EIF2AK4_2": "Other",
    "EPHA1": "TK",
    "EPHA10": "TK",
    "EPHA2": "TK",
    "EPHA3": "TK",
    "EPHA4": "TK",
    "EPHA5": "TK",
    "EPHA6": "TK",
    "EPHA7": "TK",
    "EPHA8": "TK",
    "EPHB1": "TK",
    "EPHB2": "TK",
    "EPHB3": "TK",
    "EPHB4": "TK",
    "EPHB6": "TK",
    "ERBB2": "TK",
    "ERBB3": "TK",
    "ERBB4": "TK",
    "ERN1": "Other",
    "ERN2": "Other",
    "FER": "TK",
    "FES": "TK",
    "FGFR1": "TK",
    "FGFR2": "TK",
    "FGFR3": "TK",
    "FGFR4": "TK",
    "FGR": "TK",
    "FLT1": "TK",
    "FLT3": "TK",
    "FLT4": "TK",
    "FRK": "TK",
    "FYN": "TK",
    "GAK": "Other",
    "GRK1": "AGC",
    "GRK2": "AGC",
    "GRK3": "AGC",
    "GRK4": "AGC",
    "GRK5": "AGC",
    "GRK6": "AGC",
    "GRK7": "AGC",
    "GSK3A": "CMGC",
    "GSK3B": "CMGC",
    "GUCY2C": "RGC",
    "GUCY2D": "RGC",
    "GUCY2F": "RGC",
    "HASPIN": "Other",
    "HCK": "TK",
    "HIPK1": "CMGC",
    "HIPK2": "CMGC",
    "HIPK3": "CMGC",
    "HIPK4": "CMGC",
    "HUNK": "CAMK",
    "IGF1R": "TK",
    "IKBKB": "Other",
    "IKBKE": "Other",
    "ILK": "TKL",
    "INSR": "TK",
    "INSRR": "TK",
    "IRAK1": "TKL",
    "IRAK2": "TKL",
    "IRAK3": "TKL",
    "IRAK4": "TKL",
    "ITK": "TK",
    "JAK1": "TK",
    "JAK1_1": "TK",
    "JAK1_2": "TK",
    "JAK2": "TK",
    "JAK2_1": "TK",
    "JAK2_2": "TK",
    "JAK3": "TK",
    "JAK3_1": "TK",
    "JAK3_2": "TK",
    "KALRN": "CAMK",
    "KDR": "TK",
    "KIT": "TK",
    "KSR1": "TKL",
    "KSR2": "TKL",
    "LATS1": "AGC",
    "LATS2": "AGC",
    "LCK": "TK",
    "LIMK1": "TKL",
    "LIMK2": "TKL",
    "LMTK2": "TK",
    "LMTK3": "TK",
    "LRRK1": "TKL",
    "LRRK2": "TKL",
    "LTK": "TK",
    "LYN": "TK",
    "MAK": "CMGC",
    "MAP2K1": "STE",
    "MAP2K2": "STE",
    "MAP2K3": "STE",
    "MAP2K4": "STE",
    "MAP2K5": "STE",
    "MAP2K6": "STE",
    "MAP2K7": "STE",
    "MAP3K1": "STE",
    "MAP3K10": "TKL",
    "MAP3K11": "TKL",
    "MAP3K12": "TKL",
    "MAP3K13": "TKL",
    "MAP3K14": "STE",
    "MAP3K15": "STE",
    "MAP3K19": "STE",
    "MAP3K2": "STE",
    "MAP3K20": "TKL",
    "MAP3K21": "TKL",
    "MAP3K3": "STE",
    "MAP3K4": "STE",
    "MAP3K5": "STE",
    "MAP3K6": "STE",
    "MAP3K7": "TKL",
    "MAP3K8": "STE",
    "MAP3K9": "TKL",
    "MAP4K1": "STE",
    "MAP4K2": "STE",
    "MAP4K3": "STE",
    "MAP4K4": "STE",
    "MAP4K5": "STE",
    "MAPK1": "CMGC",
    "MAPK10": "CMGC",
    "MAPK11": "CMGC",
    "MAPK12": "CMGC",
    "MAPK13": "CMGC",
    "MAPK14": "CMGC",
    "MAPK15": "CMGC",
    "MAPK3": "CMGC",
    "MAPK4": "CMGC",
    "MAPK6": "CMGC",
    "MAPK7": "CMGC",
    "MAPK8": "CMGC",
    "MAPK9": "CMGC",
    "MAPKAPK2": "CAMK",
    "MAPKAPK3": "CAMK",
    "MAPKAPK5": "CAMK",
    "MARK1": "CAMK",
    "MARK2": "CAMK",
    "MARK3": "CAMK",
    "MARK4": "CAMK",
    "MAST1": "AGC",
    "MAST2": "AGC",
    "MAST3": "AGC",
    "MAST4": "AGC",
    "MASTL": "AGC",
    "MATK": "TK",
    "MELK": "CAMK",
    "MERTK": "TK",
    "MET": "TK",
    "MINK1": "STE",
    "MKNK1": "CAMK",
    "MKNK2": "CAMK",
    "MLKL": "Other",
    "MOK": "CMGC",
    "MOS": "Other",
    "MST1R": "TK",
    "MTOR": "Atypical",
    "MUSK": "TK",
    "MYLK": "CAMK",
    "MYLK2": "CAMK",
    "MYLK3": "CAMK",
    "MYLK4": "CAMK",
    "MYO3A": "STE",
    "MYO3B": "STE",
    "NEK1": "NEK",
    "NEK10": "NEK",
    "NEK11": "NEK",
    "NEK2": "NEK",
    "NEK3": "NEK",
    "NEK4": "NEK",
    "NEK5": "NEK",
    "NEK6": "NEK",
    "NEK7": "NEK",
    "NEK8": "NEK",
    "NEK9": "NEK",
    "NIM1K": "CAMK",
    "NLK": "CMGC",
    "NPR1": "RGC",
    "NPR2": "RGC",
    "NRBP1": "Other",
    "NRBP2": "Other",
    "NRK": "STE",
    "NTRK1": "TK",
    "NTRK2": "TK",
    "NTRK3": "TK",
    "NUAK1": "CAMK",
    "NUAK2": "CAMK",
    "OBSCN": "CAMK",
    "OBSCN_1": "CAMK",
    "OBSCN_2": "CAMK",
    "OXSR1": "STE",
    "PAK1": "STE",
    "PAK2": "STE",
    "PAK3": "STE",
    "PAK4": "STE",
    "PAK5": "STE",
    "PAK6": "STE",
    "PAN3": "Other",
    "PASK": "CAMK",
    "PBK": "Other",
    "PDGFRA": "TK",
    "PDGFRB": "TK",
    "PDIK1L": "Other",
    "PDK1": "Atypical",
    "PDK2": "Atypical",
    "PDK3": "Atypical",
    "PDK4": "Atypical",
    "PDPK1": "AGC",
    "PDPK2P": "AGC",
    "PEAK1": "Other",
    "PEAK3": "Other",
    "PHKG1": "CAMK",
    "PHKG2": "CAMK",
    "PI4K2A": "Atypical",
    "PI4K2B": "Atypical",
    "PI4KA": "Atypical",
    "PI4KAP1": "Atypical",
    "PI4KAP2": "Atypical",
    "PI4KB": "Atypical",
    "PIK3C2A": "Atypical",
    "PIK3C2B": "Atypical",
    "PIK3C2G": "Atypical",
    "PIK3C3": "Atypical",
    "PIK3CA": "Atypical",
    "PIK3CB": "Atypical",
    "PIK3CD": "Atypical",
    "PIK3CG": "Atypical",
    "PIK3R4": "Other",
    "PIM1": "CAMK",
    "PIM2": "CAMK",
    "PIM3": "CAMK",
    "PINK1": "Other",
    "PIP4K2A": "Atypical",
    "PIP4K2B": "Atypical",
    "PIP4K2C": "Atypical",
    "PIP5K1A": "Atypical",
    "PIP5K1B": "Atypical",
    "PIP5K1C": "Atypical",
    "PKDCC": "Other",
    "PKMYT1": "Other",
    "PKN1": "AGC",
    "PKN2": "AGC",
    "PKN3": "AGC",
    "PLK1": "CAMK",
    "PLK2": "CAMK",
    "PLK3": "CAMK",
    "PLK4": "CAMK",
    "PLK5": "CAMK",
    "PNCK": "CAMK",
    "POMK": "Other",
    "PRAG1": "Other",
    "PRKAA1": "CAMK",
    "PRKAA2": "CAMK",
    "PRKACA": "AGC",
    "PRKACB": "AGC",
    "PRKACG": "AGC",
    "PRKCA": "AGC",
    "PRKCB": "AGC",
    "PRKCD": "AGC",
    "PRKCE": "AGC",
    "PRKCG": "AGC",
    "PRKCH": "AGC",
    "PRKCI": "AGC",
    "PRKCQ": "AGC",
    "PRKCZ": "AGC",
    "PRKD1": "CAMK",
    "PRKD2": "CAMK",
    "PRKD3": "CAMK",
    "PRKDC": "Atypical",
    "PRKG1": "AGC",
    "PRKG2": "AGC",
    "PRKX": "AGC",
    "PRKY": "AGC",
    "PRP4K": "CMGC",
    "PSKH1": "CAMK",
    "PSKH2": "CAMK",
    "PTK2": "TK",
    "PTK2B": "TK",
    "PTK6": "TK",
    "PTK7": "TK",
    "PXK": "Other",
    "RAF1": "TKL",
    "RET": "TK",
    "RIOK1": "Atypical",
    "RIOK2": "Atypical",
    "RIOK3": "Atypical",
    "RIPK1": "TKL",
    "RIPK2": "TKL",
    "RIPK3": "TKL",
    "RIPK4": "TKL",
    "RNASEL": "Other",
    "ROCK1": "AGC",
    "ROCK2": "AGC",
    "ROR1": "TK",
    "ROR2": "TK",
    "ROS1": "TK",
    "RPS6KA1": "Multiple",
    "RPS6KA1_1": "AGC",
    "RPS6KA1_2": "CAMK",
    "RPS6KA2": "Multiple",
    "RPS6KA2_1": "AGC",
    "RPS6KA2_2": "CAMK",
    "RPS6KA3": "Multiple",
    "RPS6KA3_1": "AGC",
    "RPS6KA3_2": "CAMK",
    "RPS6KA4": "Multiple",
    "RPS6KA4_1": "AGC",
    "RPS6KA4_2": "CAMK",
    "RPS6KA5": "Multiple",
    "RPS6KA5_1": "AGC",
    "RPS6KA5_2": "CAMK",
    "RPS6KA6": "Multiple",
    "RPS6KA6_1": "AGC",
    "RPS6KA6_2": "CAMK",
    "RPS6KB1": "AGC",
    "RPS6KB2": "AGC",
    "RPS6KC1": "Other",
    "RPS6KL1": "Other",
    "RSKR": "AGC",
    "RYK": "TK",
    "SBK1": "Other",
    "SBK2": "Other",
    "SBK3": "Other",
    "SCYL1": "Other",
    "SCYL2": "Other",
    "SCYL3": "Other",
    "SGK1": "AGC",
    "SGK2": "AGC",
    "SGK3": "AGC",
    "SIK1": "CAMK",
    "SIK1B": "CAMK",
    "SIK2": "CAMK",
    "SIK3": "CAMK",
    "SLK": "STE",
    "SMG1": "Atypical",
    "SNRK": "CAMK",
    "SPEG": "CAMK",
    "SPEG_1": "CAMK",
    "SPEG_2": "CAMK",
    "SRC": "TK",
    "SRMS": "TK",
    "SRPK1": "CMGC",
    "SRPK2": "CMGC",
    "SRPK3": "CMGC",
    "STK10": "STE",
    "STK11": "CAMK",
    "STK16": "Other",
    "STK17A": "CAMK",
    "STK17B": "CAMK",
    "STK24": "STE",
    "STK25": "STE",
    "STK26": "STE",
    "STK3": "STE",
    "STK31": "Other",
    "STK32A": "AGC",
    "STK32B": "AGC",
    "STK32C": "AGC",
    "STK33": "CAMK",
    "STK35": "Other",
    "STK36": "Other",
    "STK38": "AGC",
    "STK38L": "AGC",
    "STK39": "STE",
    "STK4": "STE",
    "STK40": "CAMK",
    "STKLD1": "Other",
    "STRADA": "STE",
    "STRADB": "STE",
    "STYK1": "TK",
    "SYK": "TK",
    "TAOK1": "STE",
    "TAOK2": "STE",
    "TAOK3": "STE",
    "TBCK": "Other",
    "TBK1": "Other",
    "TEC": "TK",
    "TEK": "TK",
    "TESK1": "TKL",
    "TESK2": "TKL",
    "TEX14": "Other",
    "TEX14_1": "Other",
    "TEX14_2": "Other",
    "TGFBR1": "TKL",
    "TGFBR2": "TKL",
    "TIE1": "TK",
    "TLK1": "Other",
    "TLK2": "Other",
    "TNIK": "STE",
    "TNK1": "TK",
    "TNK2": "TK",
    "TNNI3K": "TKL",
    "TP53RK": "Other",
    "TRIB1": "CAMK",
    "TRIB2": "CAMK",
    "TRIB3": "CAMK",
    "TRIO": "CAMK",
    "TRPM6": "Atypical",
    "TRPM7": "Atypical",
    "TRRAP": "Atypical",
    "TSSK1B": "CAMK",
    "TSSK2": "CAMK",
    "TSSK3": "CAMK",
    "TSSK4": "CAMK",
    "TSSK6": "CAMK",
    "TTBK1": "CK1",
    "TTBK2": "CK1",
    "TTK": "Other",
    "TTN": "CAMK",
    "TXK": "TK",
    "TYK2": "TK",
    "TYK2_1": "TK",
    "TYK2_2": "TK",
    "TYRO3": "TK",
    "UHMK1": "Other",
    "ULK1": "Other",
    "ULK2": "Other",
    "ULK3": "Other",
    "ULK4": "Other",
    "VRK1": "CK1",
    "VRK2": "CK1",
    "VRK3": "CK1",
    "WEE1": "Other",
    "WEE2": "Other",
    "WNK1": "Other",
    "WNK2": "Other",
    "WNK3": "Other",
    "WNK4": "Other",
    "YES1": "TK",
    "ZAP70": "TK",
}
"""dict[str, str]: Adjudicated kinase group for every ``DICT_KINASE`` key plus each bare
multi-domain gene symbol (``"Multiple"`` when its domains' groups differ); precomputed by
:func:`mkt.schema.utils.return_kinase_group_dict` so group lookups never load the corpus."""

SET_LIPID_KINASE = frozenset(
    {
        "PI4K2A",
        "PI4K2B",
        "PI4KA",
        "PI4KAP1",
        "PI4KAP2",
        "PI4KB",
        "PIK3C2A",
        "PIK3C2B",
        "PIK3C2G",
        "PIK3C3",
        "PIK3CA",
        "PIK3CB",
        "PIK3CD",
        "PIK3CG",
        "PIP4K2A",
        "PIP4K2B",
        "PIP4K2C",
        "PIP5K1A",
        "PIP5K1B",
        "PIP5K1C",
    }
)
"""frozenset[str]: Lipid kinases among ``DICT_KINASE`` keys (plus bare multi-domain symbols
whose domains all qualify); precomputed by :func:`mkt.schema.utils.return_lipid_kinase_set`."""
