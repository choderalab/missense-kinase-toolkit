import logging

logger = logging.getLogger(__name__)


# https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/abbreviation.html#refs
DICT_AA_PROPERTIES = {
    "A": {"charge": "nonpolar", "volume": 88.6},
    "C": {"charge": "polar", "volume": 108.5},
    "D": {"charge": "negative", "volume": 111.1},
    "E": {"charge": "negative", "volume": 138.4},
    "F": {"charge": "nonpolar", "volume": 189.9},
    "G": {"charge": "nonpolar", "volume": 60.1},
    "H": {"charge": "positive", "volume": 153.2},
    "I": {"charge": "nonpolar", "volume": 166.7},
    "K": {"charge": "positive", "volume": 168.6},
    "L": {"charge": "nonpolar", "volume": 166.7},
    "M": {"charge": "nonpolar", "volume": 162.9},
    "N": {"charge": "polar", "volume": 114.1},
    "P": {"charge": "nonpolar", "volume": 112.7},
    "Q": {"charge": "polar", "volume": 143.8},
    "R": {"charge": "positive", "volume": 173.4},
    "S": {"charge": "polar", "volume": 89.0},
    "T": {"charge": "polar", "volume": 116.1},
    "V": {"charge": "nonpolar", "volume": 140.0},
    "W": {"charge": "nonpolar", "volume": 227.8},
    "Y": {"charge": "polar", "volume": 193.6},
}
"""dict[str, dict[str, str | int]]: Dictionary of amino acid properties.
Keys are single-letter amino acid codes, and values are dictionaries with properties:
- `charge`: Charge of the amino acid (e.g., "nonpolar", "polar", "positive", "negative").
- `volume`: Volume of the amino acid in cubic angstroms (int).
This dictionary can be used to look up properties of amino acids by their single-letter code.
"""

DICT_AA_CHANGES = {
    "charge gain, positive": {
        "property": "charge",
        "from": "nonpolar" or "polar",
        "to": "positive",
    },
    "charge gain, negative": {
        "property": "charge",
        "from": "nonpolar" or "polar",
        "to": "negtive",
    },
    "charge loss, positive": {
        "property": "charge",
        "from": "positive",
        "to": "nonpolar" or "polar",
    },
    "charge loss, negative": {
        "property": "charge",
        "from": "negative",
        "to": "nonpolar" or "polar",
    },
    "charge change, positive to negative": {
        "property": "charge",
        "from": "positive",
        "to": "negative",
    },
    "charge change, negative to positive": {
        "property": "charge",
        "from": "negative",
        "to": "positive",
    },
    "polarity gain": {
        "property": "charge",
        "from": "nonpolar",
        "to": "polar",
    },
    "polarity loss": {
        "property": "charge",
        "from": "polar",
        "to": "nonpolar",
    },
    # >75 cubic angstrom gain
    "volume gain, large": {
        "property": "volume",
        "from": 75,
        "to": float("inf"),
    },
    # 25-75 cubic angstrom gain
    "volume gain, modest": {
        "property": "volume",
        "from": 25,
        "to": 75,
    },
    # >75 cubic angstrom loss
    "volume loss, large": {
        "property": "volume",
        "from": float("-inf"),
        "to": -75,
    },
    # 25-75 cubic angstrom loss
    "volume loss, modest": {
        "property": "volume",
        "from": -75,
        "to": -25,
    },
}
"""dict[str, dict[str, str | int]]: Dictionary of amino acid change properties.
Keys are descriptions of the change (e.g., "charge gain, positive"), and values are dictionaries with properties:
- `property`: The property being changed (e.g., "charge", "volume").
- `from`: The original value of the property before the change.
- `to`: The new value of the property after the change.
This dictionary can be used to look up the nature of changes in amino acid properties.
"""


def get_aa_property(aa: str, property_name: str) -> str | int | None:
    """
    Get a specific property of an amino acid.

    Parameters
    ----------
    aa : str
        Single-letter code of the amino acid (e.g., "A" for Alanine).
    property_name : str
        Name of the property to retrieve (e.g., "charge", "volume").

    Returns
    -------
    str | int | None
        The value of the specified property, or None if the amino acid or property does not exist.
    """
    aa = aa.upper()
    if aa in DICT_AA_PROPERTIES and property_name in DICT_AA_PROPERTIES[aa]:
        return DICT_AA_PROPERTIES[aa][property_name]
    return None


def classify_aa_change(
    aa_from: str,
    aa_to: str,
) -> str | None:
    """
    Classify the change between two amino acids.

    Parameters
    ----------
    aa_from : str
        Single-letter code of the original amino acid.
    aa_to : str
        Single-letter code of the new amino acid.

    Returns
    -------
    str | None
        Description of the change (e.g., "charge gain, positive"), or None if no change is found.
    """
    aa_from, aa_to = aa_from.upper(), aa_to.upper()

    if aa_from not in DICT_AA_PROPERTIES or aa_to not in DICT_AA_PROPERTIES:
        logger.error(
            f"Invalid amino acids: {aa_from} or {aa_to} not found in properties dictionary."
        )
        return None

    properties_from = DICT_AA_PROPERTIES[aa_from]
    properties_to = DICT_AA_PROPERTIES[aa_to]

    dict_out = dict.fromkeys(["charge", "volume"])
    for change, dict_inner in DICT_AA_CHANGES.items():
        # check charge changes, including polarity
        if dict_inner["property"] == "charge":
            charge_from = properties_from[dict_inner["property"]]
            charge_to = properties_to[dict_inner["property"]]
            if charge_from == dict_inner["from"] and charge_to == dict_inner["to"]:
                dict_out[dict_inner["property"]] = change
        # check volume changes
        elif dict_inner["property"] == "volume":
            volume_change = properties_to["volume"] - properties_from["volume"]
            if volume_change < dict_inner["to"] and volume_change >= dict_inner["from"]:
                dict_out[dict_inner["property"]] = change
    return dict_out
