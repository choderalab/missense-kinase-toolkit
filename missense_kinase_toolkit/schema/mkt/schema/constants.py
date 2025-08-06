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
