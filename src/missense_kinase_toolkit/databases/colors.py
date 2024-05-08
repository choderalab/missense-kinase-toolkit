def map_aa_to_single_letter_code(
    aa: str,
) -> str | None:
    """Map any amino acid input from name or 3-letter or single-letter code to validated single-letter code.
    
    Parameters
    ----------
    aa : str
        Amino acid name or 3-letter or single-letter code

    Returns
    -------
    str | None
        Single-letter amino acid code if valid; otherwise None

    Notes
    -----
        3-letter and single-letter AA converted to uppercase; AA name converted to lowercase

    """
    if len(aa) == 3:
        aa_clean = aa.upper()
        dict_aa = {entry[2]: entry[1] for entry in AA_MAPPING}
        if aa_clean in dict_aa.keys():
            return dict_aa[aa_clean]
        else:
            print(f"Invalid 3-letter amino acid: {aa.upper()}")
    elif len(aa) > 3:
        aa_clean = aa.lower()
        dict_aa = {entry[0]: entry[1] for entry in AA_MAPPING}
        if aa_clean in dict_aa.keys():
            return dict_aa[aa_clean]
        else:
            print(f"Invalid amino acid name: {aa.lower()}")
            return None
    elif len(aa) == 1:
        aa_clean = aa.upper()
        if aa_clean in [entry[1] for entry in AA_MAPPING]:
            return aa.upper()
        else:
            print(f"Invalid single-letter amino acid: {aa_clean}")
            return None
    else:
        print(f"Length error and invalid amino acid: {aa}")
        return None


def map_single_letter_aa_to_color(aa, dict_color):
    """Map amino acid to color using specified dictionary.
    
    Parameters
    ----------
    aa : str
        Amino acid name or 3-letter or single-letter code
    dict_color : dict[str, str]
        Dictionary mapping single-letter amino acid to color

    Returns
    -------
    str
        Color that corresponds to AA single-letter code in selected palette

    """
    aa_clean = map_aa_to_single_letter_code(aa)
    if aa_clean is None:
        print(f"{aa} is an invalid amino acid; using '-' as default symbol.")
        aa_clean = "-" 
    if aa_clean  in dict_color.keys():
        return dict_color[aa_clean]
    else:
        print(f"Invalid amino acid: {aa}")


AA_MAPPING = [
    ("alanine", "A", "ALA"),
    ("arginine", "R", "ARG"),
    ("asparagine", "N", "ASN"),
    ("aspartic acid", "D", "ASP"),
    ("cysteine", "C", "CYS"),
    ("glutamic acid", "E", "GLU"),
    ("glutamine", "Q", "GLN"),
    ("glycine", "G", "GLY"),
    ("histidine", "H", "HIS"),
    ("isoleucine", "I", "ILE"),
    ("leucine", "L", "LEU"),
    ("lysine", "K", "LYS"),
    ("methionine", "M", "MET"),
    ("phenylalanine", "F", "PHE"),
    ("proline", "P", "PRO"),
    ("serine", "S", "SER"),
    ("threonine", "T", "THR"),
    ("tryptophan", "W", "TRP"),
    ("tyrosine", "Y", "TYR"),
    ("valine", "V", "VAL"),
]
"""list[tuple[str, str, str]]: Set of amino acid mappings (lower) to single-letter code and 3-letter code (upper)"""


# https://yulab-smu.top/ggmsa/articles/guides
# https://en.wikipedia.org/wiki/Help:Distinguishable_colors
COLORS_ALPHABET_PROJECT = {
    "-" : "#FFFFFF",
    "A" : "#F0A3FF",
    "B" : "#0075DC",
    "C" : "#993F00",
    "D" : "#4C005C",
    "E" : "#191919",
    "F" : "#005C31",
    "G" : "#2BCE48",
    "H" : "#FFCC99",
    "I" : "#808080",
    "J" : "#94FFB5",
    "K" : "#8F7C00",
    "L" : "#9DCC00",
    "M" : "#C20088",
    "N" : "#003380",
    "O" : "#FFA405",
    "P" : "#FFA8BB",
    "Q" : "#426600",
    "R" : "#FF0010",
    "S" : "#5EF1F2",
    "T" : "#00998F",
    "U" : "#E0FF66",
    "V" : "#740AFF",
    "W" : "#990000",
    "X" : "#FFFF80",
    "Y" : "#FFE100",
    "Z" : "#FF5005"
}
"""dict[str, str]: Mapping amino acid to color using 2010 Colour Alphabet Project"""


# asapdiscovery-genetics/asapdiscovery/genetics/seq_alignment.py
COLORS_ASAP = {
    "A": "red",         # Alanine (ALA)
    "R": "blue",        # Arginine (ARG)
    "N": "green",       # Asparagine (ASN)
    "D": "yellow",      # Aspartic acid (ASP)
    "C": "orange",      # Cysteine (CYS)
    "Q": "purple",      # Glutamine (GLN)
    "E": "cyan",        # Glutamic acid (GLU)
    "G": "magenta",     # Glycine (GLY)
    "H": "pink",        # Histidine (HIS)
    "I": "brown",       # Isoleucine (ILE)
    "L": "gray",        # Leucine (LEU)
    "K": "lime",        # Lysine (LYS)
    "M": "teal",        # Methionine (MET)
    "F": "navy",        # Phenylalanine (PHE)
    "P": "olive",       # Proline (PRO)
    "S": "maroon",      # Serine (SER)
    "T": "silver",      # Threonine (THR)
    "W": "gold",        # Tryptophan (TRP)
    "Y": "skyblue",     # Tyrosine (TYR)
    "V": "violet",      # Valine (VAL)
    "-": "white",
}
"""dict[str, str]: Mapping amino acid to color using ASAP Discovery palette"""


# http://openrasmol.org/doc/rasmol.html#aminocolours
COLORS_RASMOL = {
    "A": "#C8C8C8",  # Alanine (ALA)
    "R": "#145AFF",  # Arginine (ARG)
    "N": "#00DCDC",  # Asparagine (ASN)
    "D": "#E60A0A",  # Aspartic acid (ASP)
    "C": "#E6E600",  # Cysteine (CYS)
    "Q": "#00DCDC",  # Glutamine (GLN)
    "E": "#E60A0A",  # Glutamic acid (GLU)
    "G": "#EBEBEB",  # Glycine (GLY)
    "H": "#8282D2",  # Histidine (HIS)
    "I": "#0F820F",  # Isoleucine (ILE)
    "L": "#0F820F",  # Leucine (LEU)
    "K": "#145AFF",  # Lysine (LYS)
    "M": "#E6E600",  # Methionine (MET)
    "F": "#3232AA",  # Phenylalanine (PHE)
    "P": "#DC9682",  # Proline (PRO)
    "S": "#FA9600",  # Serine (SER)
    "T": "#FA9600",  # Threonine (THR)
    "W": "#B45AB4",  # Tryptophan (TRP)
    "Y": "#3232AA",  # Tyrosine (TYR)
    "V": "#0F820F",  # Valine (VAL)
    "-": "#BEA06E",
}
"""dict[str, str]: Mapping amino acid to color using RasMol amino color scheme"""

# http://openrasmol.org/doc/rasmol.html#shapelycolours
COLORS_SHAPELY = {
    "A": "#8CFF8C",  # Alanine (ALA)
    "R": "#00007C",  # Arginine (ARG)
    "N": "#FF7C70",  # Asparagine (ASN)
    "D": "#A00042",  # Aspartic acid (ASP)
    "C": "#FFFF70",  # Cysteine (CYS)
    "Q": "#FF4C4C",  # Glutamine (GLN)
    "E": "#660000",  # Glutamic acid (GLU)
    "G": "#FFFFFF",  # Glycine (GLY)
    "H": "#7070FF",  # Histidine (HIS)
    "I": "#004C00",  # Isoleucine (ILE)
    "L": "#455E45",  # Leucine (LEU)
    "K": "#4747B8",  # Lysine (LYS)
    "M": "#B8A042",  # Methionine (MET)
    "F": "#534C42",  # Phenylalanine (PHE)
    "P": "#525252",  # Proline (PRO)
    "S": "#FF7042",  # Serine (SER)
    "T": "#B84C00",  # Threonine (THR)
    "W": "#4F4600",  # Tryptophan (TRP)
    "Y": "#8C704C",  # Tyrosine (TYR)
    "V": "#FF8CFF",  # Valine (VAL)
    "-": "#000000",
}
"""dict[str, str]: Mapping amino acid to color using Shapely palette"""


# https://www.jalview.org/help/html/colourSchemes/clustal.html
# hydrophobic: blue (A,I,L,M,F,W,V)
# positive: red (K,R)
# negative: magenta (D,E)
# polar: green (N,Q,S,T)
# aromatic: cyan (H,Y)
# cysteine: pink (C)
# glycine: orange (G)
# proline: yellow (P)
COLORS_CLUSTALX = {
    "A": "blue",  # Alanine (ALA)
    "R": "red",  # Arginine (ARG)
    "N": "green",  # Asparagine (ASN)
    "D": "magenta",  # Aspartic acid (ASP)
    "C": "pink",  # Cysteine (CYS)
    "Q": "green",  # Glutamine (GLN)
    "E": "magenta",  # Glutamic acid (GLU)
    "G": "orange",  # Glycine (GLY)
    "H": "cyan",  # Histidine (HIS)
    "I": "blue",  # Isoleucine (ILE)
    "L": "blue",  # Leucine (LEU)
    "K": "red",  # Lysine (LYS)
    "M": "blue",  # Methionine (MET)
    "F": "blue",  # Phenylalanine (PHE)
    "P": "yellow",  # Proline (PRO)
    "S": "green",  # Serine (SER)
    "T": "green",  # Threonine (THR)
    "W": "blue",  # Tryptophan (TRP)
    "Y": "cyan",  # Tyrosine (TYR)
    "V": "blue",  # Valine (VAL)
    "-": "white",
}
"""dict[str, str]: Mapping amino acid to color using Clustal X color scheme"""