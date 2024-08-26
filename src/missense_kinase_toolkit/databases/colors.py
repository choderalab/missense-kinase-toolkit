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
    # Check if 3-letter code provided
    if len(aa) == 3:
        aa_clean = aa.upper()
        dict_aa = {entry[2]: entry[1] for entry in AA_MAPPING}
        if aa_clean in dict_aa.keys():
            return dict_aa[aa_clean]
        else:
            print(f"Invalid 3-letter amino acid: {aa.upper()}")
    # Check if amino acid name provided
    elif len(aa) > 3:
        aa_clean = aa.lower()
        dict_aa = {entry[0]: entry[1] for entry in AA_MAPPING}
        if aa_clean in dict_aa.keys():
            return dict_aa[aa_clean]
        else:
            print(f"Invalid amino acid name: {aa.lower()}")
            return None
    # Check if single-letter code provided
    elif len(aa) == 1:
        aa_clean = aa.upper()
        if aa_clean in [entry[1] for entry in AA_MAPPING]:
            return aa.upper()
        else:
            print(f"Invalid single-letter amino acid: {aa_clean}")
            return None
    else:
        # Invalid amino acid input length (0 or 2 characters)
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

DICT_COLORS = {
    # https://en.wikipedia.org/wiki/Help:Distinguishable_colors
    "ALPHABET_PROJECT" : {
        "DICT_COLORS" : {
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
        },
        "DICT_ANNOTATION" : None
    },
    # asapdiscovery-genetics/asapdiscovery/genetics/seq_alignment.py
    "ASAP" : {
        "DICT_COLORS" : {
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
        },
        "DICT_ANNOTATION" : None
    },
    # http://openrasmol.org/doc/rasmol.html#aminocolours
    "RASMOL" : {
        "DICT_COLORS" : {
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
        },
        "DICT_ANNOTATION" : {
            
            "A"         :       "#C8C8C8",
            "R, K"      :       "#145AFF",
            "N, Q"      :       "#00DCDC",
            "D, E"      :       "#E60A0A",
            "C, M"      :       "#E6E600",
            "G"         :       "#EBEBEB",
            "H"         :       "#8282D2",
            "I, L, V"   :       "#0F820F",
            "F, Y"      :       "#3232AA",
            "P"         :       "#DC9682",
            "S, T"      :       "#FA9600",
            "W"         :       "#B45AB4",
            "-"         :       "#BEA06E",
        }
    },
    # http://openrasmol.org/doc/rasmol.html#shapelycolours
    # https://www.dnastar.com/manuals/MegAlignPro/17.3.1/en/topic/change-the-analysis-view-color-scheme
    "SHAPELY" : {
        "DICT_COLORS" : {
            "A": "#8CFF8C",  # Alanine (ALA)
            "R": "#00007C",  # Arginine (ARG)
            "N": "#FF7C70",  # Asparagine (ASN)
            "D": "#A00042",  # Aspartic acid (ASP)
            "C": "#FFFF70",  # Cysteine (CYS)
            "Q": "#FF7C70",  # Glutamine (GLN)
            "E": "#A00042",  # Glutamic acid (GLU)
            "G": "#FFFFFF",  # Glycine (GLY)
            "H": "#7070FF",  # Histidine (HIS)
            "I": "#004C00",  # Isoleucine (ILE)
            "L": "#004C00",  # Leucine (LEU)
            "K": "#00007C",  # Lysine (LYS)
            "M": "#FFFF70",  # Methionine (MET)
            "F": "#534C42",  # Phenylalanine (PHE)
            "P": "#525252",  # Proline (PRO)
            "S": "#FF7042",  # Serine (SER)
            "T": "#FF7042",  # Threonine (THR)
            "W": "#4F4600",  # Tryptophan (TRP)
            "Y": "#534C42",  # Tyrosine (TYR)
            "V": "#004C00",  # Valine (VAL)
            "-": "#000000",
        },
        "DICT_ANNOTATION" : {
            "A"         :       "#8CFF8C",
            "R, K"      :       "#00007C",
            "N, Q"      :       "#FF7C70",
            "D, E"      :       "#A00042",
            "C, M"      :       "#FFFF70",
            "G"         :       "#FFFFFF",
            "H"         :       "#7070FF",
            "I, L, V"   :       "#004C00",
            "F, Y"      :       "#534C42",
            "P"         :       "#525252",
            "S, T"      :       "#FF7042",
            "W"         :       "#4F4600",
            "-"         :       "#000000",
        }
    },
    # https://www.jalview.org/help/html/colourSchemes/clustal.html
    "CLUSTALX" : {
        "DICT_COLORS" : {
            "A": "blue",    # Alanine (ALA)
            "R": "red",     # Arginine (ARG)
            "N": "green",   # Asparagine (ASN)
            "D": "magenta", # Aspartic acid (ASP)
            "C": "pink",    # Cysteine (CYS)
            "Q": "green",   # Glutamine (GLN)
            "E": "magenta", # Glutamic acid (GLU)
            "G": "orange",  # Glycine (GLY)
            "H": "cyan",    # Histidine (HIS)
            "I": "blue",    # Isoleucine (ILE)
            "L": "blue",    # Leucine (LEU)
            "K": "red",     # Lysine (LYS)
            "M": "blue",    # Methionine (MET)
            "F": "blue",    # Phenylalanine (PHE)
            "P": "yellow",  # Proline (PRO)
            "S": "green",   # Serine (SER)
            "T": "green",   # Threonine (THR)
            "W": "blue",    # Tryptophan (TRP)
            "Y": "cyan",    # Tyrosine (TYR)
            "V": "blue",    # Valine (VAL)
            "-": "white",
        },
        "DICT_ANNOTATION" : {
            "hydrophobic": "blue",  # A,I,L,M,F,W,V
            "positive": "red",      # K,R
            "negative": "magenta",  # D,E
            "polar": "green",       # N,Q,S,T
            "aromatic": "cyan",     # H,Y
            "cysteine": "pink",     # C
            "glycine": "orange",    # G
            "proline": "yellow",    # P
        }
    },
    # https://www.jalview.org/help/html/colourSchemes/zappo.html
    "ZAPPO" : {
        "DICT_COLORS" : {
            "A": "#ffafaf",  # Alanine (ALA)
            "R": "#6464ff",  # Arginine (ARG)
            "N": "#02ff00",  # Asparagine (ASN)
            "D": "#ff0000",  # Aspartic acid (ASP)
            "C": "#ffff00",  # Cysteine (CYS)
            "Q": "#02ff00",  # Glutamine (GLN)
            "E": "#ff0000",  # Glutamic acid (GLU)
            "G": "#ff00ff",  # Glycine (GLY)
            "H": "#6464ff",  # Histidine (HIS)
            "I": "#ffafaf",  # Isoleucine (ILE)
            "L": "#ffafaf",  # Leucine (LEU)
            "K": "#6464ff",  # Lysine (LYS)
            "M": "#ffafaf",  # Methionine (MET)
            "F": "#ffc803",  # Phenylalanine (PHE)
            "P": "#ff00ff",  # Proline (PRO)
            "S": "#02ff00",  # Serine (SER)
            "T": "#02ff00",  # Threonine (THR)
            "W": "#ffc803",  # Tryptophan (TRP)
            "Y": "#ffc803",  # Tyrosine (TYR)
            "V": "#ffafaf",  # Valine (VAL)
            "-": "#000000",
        },
        "DICT_ANNOTATION" : {
            "aliphatic hydrophobic"     :       "#ffafaf", # A,I,L,M,V
            "aromatic"                  :       "#ffc803", # F,W,Y
            "positive"                  :       "#6464ff", # R,H,K
            "negative"                  :       "#ff0000", # D,E
            "hydrophilic"               :       "#02ff00", # N,Q,S,T
            "conformationally special"  :       "#ff00ff", # G,P
            "cysteine"                  :       "#ffff00", # C
        }
    }
}
"""dict[dict[str, str], dict[str, str]]: Mapping amino acid to color
    using specified dictionary. Dictionaries include color schemes from:
    - 2010 Colour Alphabet Project (ALPHABET_PROJECT)
    - ASAP Discovery palette (ASAP)
    - RasMol amino color scheme (RASMOL)
    - Shapely amino color scheme (SHAPELY)
    - Clustal X color scheme (CLUSTALX)
    - Zappo color scheme (ZAPPO)
    Dictionary keys include:
    - DICT_COLORS: Dictionary mapping single-letter amino acid to color
    - DICT_ANNOTATION: Dictionary mapping amino acid groups to color if one exists
"""
