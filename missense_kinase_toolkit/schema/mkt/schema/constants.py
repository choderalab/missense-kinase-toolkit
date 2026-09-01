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
