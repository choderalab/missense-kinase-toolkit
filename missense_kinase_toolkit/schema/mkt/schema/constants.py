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
    "Serine/threonine-protein kinase mTOR domain",
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
    "PDIK1L",
    "ROR1",
    "ROR2",
    "RYK",
    "SBK3",
]
"""list[str]: Curated pseudokinases that retain an intact VAIK-K / HRD-D / DFG-D triad and
are therefore NOT caught by the catalytic-residue heuristic (false negatives); they are
catalytically dead for other reasons (degraded regulatory spine, glycine-rich loop, or
nucleotide binding). is_pseudokinase() force-returns True for these.

Citations:
  - BUB1B (BUBR1) -- a bona fide pseudokinase despite an intact catalytic triad:
    Suijkerbuijk et al., Dev Cell 2012; Murphy et al., Biochem J 2014.
  - ROR1, ROR2, RYK -- Wnt-receptor pseudokinases that retain catalytic residues but
    lack activity: Boudeau et al., Trends Cell Biol 2006; Reiterer et al., Trends Cell
    Biol 2014; Mendrola et al., Biochem Soc Trans 2013.
  - PDIK1L, SBK3 -- annotated pseudokinases with intact triads; classified pseudo on
    nucleotide-binding / catalytic grounds (Murphy et al., Biochem J 2014). Lower
    confidence than the above; revisit if a more authoritative list is adopted."""

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
and is NOT rescued by the beta2-lysine alternative, unlike WNK1/2/3. WNK4 is the most
divergent, weakly/debatably active WNK -- left flagged pending review rather than added
here."""

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
