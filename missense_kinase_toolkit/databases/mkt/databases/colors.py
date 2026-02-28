import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mkt.schema.io_utils import save_plot


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
            return None
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
    if aa_clean in dict_color.keys():
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
    "ALPHABET_PROJECT": {
        "DICT_COLORS": {
            "-": "#FFFFFF",
            "A": "#F0A3FF",
            "B": "#0075DC",
            "C": "#993F00",
            "D": "#4C005C",
            "E": "#191919",
            "F": "#005C31",
            "G": "#2BCE48",
            "H": "#FFCC99",
            "I": "#808080",
            "J": "#94FFB5",
            "K": "#8F7C00",
            "L": "#9DCC00",
            "M": "#C20088",
            "N": "#003380",
            "O": "#FFA405",
            "P": "#FFA8BB",
            "Q": "#426600",
            "R": "#FF0010",
            "S": "#5EF1F2",
            "T": "#00998F",
            "U": "#E0FF66",
            "V": "#740AFF",
            "W": "#990000",
            "X": "#FFFF80",
            "Y": "#FFE100",
            "Z": "#FF5005",
        },
        "DICT_ANNOTATION": None,
    },
    # asapdiscovery-genetics/asapdiscovery/genetics/seq_alignment.py
    "ASAP": {
        "DICT_COLORS": {
            "A": "red",  # Alanine (ALA)
            "R": "blue",  # Arginine (ARG)
            "N": "green",  # Asparagine (ASN)
            "D": "yellow",  # Aspartic acid (ASP)
            "C": "orange",  # Cysteine (CYS)
            "Q": "purple",  # Glutamine (GLN)
            "E": "cyan",  # Glutamic acid (GLU)
            "G": "magenta",  # Glycine (GLY)
            "H": "pink",  # Histidine (HIS)
            "I": "brown",  # Isoleucine (ILE)
            "L": "gray",  # Leucine (LEU)
            "K": "lime",  # Lysine (LYS)
            "M": "teal",  # Methionine (MET)
            "F": "navy",  # Phenylalanine (PHE)
            "P": "olive",  # Proline (PRO)
            "S": "maroon",  # Serine (SER)
            "T": "silver",  # Threonine (THR)
            "W": "gold",  # Tryptophan (TRP)
            "Y": "skyblue",  # Tyrosine (TYR)
            "V": "violet",  # Valine (VAL)
            "-": "white",
        },
        "DICT_ANNOTATION": None,
    },
    # http://openrasmol.org/doc/rasmol.html#aminocolours
    "RASMOL": {
        "DICT_COLORS": {
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
        "DICT_ANNOTATION": {
            "A": "#C8C8C8",
            "R, K": "#145AFF",
            "N, Q": "#00DCDC",
            "D, E": "#E60A0A",
            "C, M": "#E6E600",
            "G": "#EBEBEB",
            "H": "#8282D2",
            "I, L, V": "#0F820F",
            "F, Y": "#3232AA",
            "P": "#DC9682",
            "S, T": "#FA9600",
            "W": "#B45AB4",
            "-": "#BEA06E",
        },
    },
    # http://openrasmol.org/doc/rasmol.html#shapelycolours
    # https://www.dnastar.com/manuals/MegAlignPro/17.3.1/en/topic/change-the-analysis-view-color-scheme
    "SHAPELY": {
        "DICT_COLORS": {
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
        "DICT_ANNOTATION": {
            "A": "#8CFF8C",
            "R, K": "#00007C",
            "N, Q": "#FF7C70",
            "D, E": "#A00042",
            "C, M": "#FFFF70",
            "G": "#FFFFFF",
            "H": "#7070FF",
            "I, L, V": "#004C00",
            "F, Y": "#534C42",
            "P": "#525252",
            "S, T": "#FF7042",
            "W": "#4F4600",
            "-": "#000000",
        },
    },
    # https://www.jalview.org/help/html/colourSchemes/clustal.html
    "CLUSTALX": {
        "DICT_COLORS": {
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
        },
        "DICT_ANNOTATION": {
            "hydrophobic": "blue",  # A,I,L,M,F,W,V
            "positive": "red",  # K,R
            "negative": "magenta",  # D,E
            "polar": "green",  # N,Q,S,T
            "aromatic": "cyan",  # H,Y
            "cysteine": "pink",  # C
            "glycine": "orange",  # G
            "proline": "yellow",  # P
        },
    },
    # https://www.jalview.org/help/html/colourSchemes/zappo.html
    "ZAPPO": {
        "DICT_COLORS": {
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
        "DICT_ANNOTATION": {
            "aliphatic hydrophobic": "#ffafaf",  # A,I,L,M,V
            "aromatic": "#ffc803",  # F,W,Y
            "positive": "#6464ff",  # R,H,K
            "negative": "#ff0000",  # D,E
            "hydrophilic": "#02ff00",  # N,Q,S,T
            "conformationally special": "#ff00ff",  # G,P
            "cysteine": "#ffff00",  # C
        },
    },
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

DEFAULT_NULL_COLOR = "darkgray"
"""str: Default color for null/zero values in colormaps."""


def interpolate_color(
    norm_value: float, start_color_hex: str, end_color_hex: str
) -> str:
    """Interpolate between two colors based on normalized value.

    Parameters:
    -----------
    norm_value : float
        Normalized value between 0 and 1.
    start_color_hex : str
        Starting color in hex format (e.g., "#FFFFFF").
    end_color_hex : str
        Ending color in hex format (e.g., "#FF0000").

    Returns:
    --------
    str
        Interpolated color in hex format.
    """
    # convert hex to RGB
    start_r = int(start_color_hex[1:3], 16)
    start_g = int(start_color_hex[3:5], 16)
    start_b = int(start_color_hex[5:7], 16)

    end_r = int(end_color_hex[1:3], 16)
    end_g = int(end_color_hex[3:5], 16)
    end_b = int(end_color_hex[5:7], 16)

    # interpolate
    interp_r = int(start_r + (end_r - start_r) * norm_value)
    interp_g = int(start_g + (end_g - start_g) * norm_value)
    interp_b = int(start_b + (end_b - start_b) * norm_value)

    return f"#{interp_r:02x}{interp_g:02x}{interp_b:02x}"


def percentile_colormap(
    values: list[float],
    color_stops: dict[int, tuple[str, str]],
    zero_color: str = DEFAULT_NULL_COLOR,
) -> list[str]:
    """Map numeric values to colors using percentile-based interpolation.

    Divides non-zero values into N equal percentile bins (where N is the number
    of color stops) and interpolates within each bin's color range.

    Parameters:
    -----------
    values : list[float]
        Numeric values to map to colors.
    color_stops : dict[int, tuple[str, str]]
        Dict mapping bin number (1-indexed) to (start_hex, end_hex) tuples.
        The number of entries determines the number of percentile bins
        (e.g., 4 entries = quartiles, 5 entries = quintiles).
    zero_color : str
        Color for zero values (default: "darkgray").

    Returns:
    --------
    list[str]
        List of color strings (hex or named).
    """
    n_bins = len(color_stops)

    # calculate percentile boundaries from non-zero values
    non_zero_values = sorted([v for v in values if v > 0])
    if non_zero_values:
        n = len(non_zero_values)
        # boundaries at each 1/n_bins fraction (e.g., 4 bins -> 0.25, 0.50, 0.75)
        boundaries = [
            non_zero_values[max(0, int(n * (i + 1) / n_bins) - 1)]
            for i in range(n_bins - 1)
        ]
    else:
        boundaries = [(i + 1) / n_bins for i in range(n_bins - 1)]

    list_color = []
    for value in values:
        if value == 0:
            list_color.append(zero_color)
        else:
            # determine which bin and position within it
            bin_idx = n_bins  # default to last bin
            q_min = boundaries[-1] if boundaries else 0
            q_max = 1.0
            for i, boundary in enumerate(boundaries):
                if value <= boundary:
                    bin_idx = i + 1
                    q_min = boundaries[i - 1] if i > 0 else 0
                    q_max = boundary
                    break

            # interpolate position within bin (0 to 1)
            t = (value - q_min) / (q_max - q_min) if q_max > q_min else 0.5

            start_hex, end_hex = color_stops[bin_idx]
            list_color.append(interpolate_color(t, start_hex, end_hex))

    return list_color


DICT_QUARTILE_HEATMAP_COLORMAP = {
    1: ("#228B22", "#FFD700"),  # green → yellow
    2: ("#FFD700", "#FF8C00"),  # yellow → orange
    3: ("#FF8C00", "#FF0000"),  # orange → red
    4: ("#FF0000", "#8B0000"),  # red → dark red
}
"""dict[int, tuple[str, str]]: Quartile heatmap colormap for mutation visualization.
Maps quartile bin number (1-indexed) to (start_hex, end_hex) tuples for use with ``percentile_colormap``.
"""


def generate_colormap_legend(
    color_stops: dict[int, tuple[str, str]],
    output_path: str | None = None,
    zero_color: str = DEFAULT_NULL_COLOR,
    n_gradient_steps: int = 256,
    null_steps: int | None = None,
    figsize: tuple[float, float] = (0.75, 5.5),
) -> None:
    """Generate a vertical colormap legend image (SVG and PNG) from color stops.

    Creates a vertical gradient bar from bottom (null/zero color) to top (highest
    density), with percentile tick labels at 0, 0.25, 0.5, 0.75, and 1. Colors are
    generated via ``percentile_colormap`` with a synthetic uniform dataset so the
    legend is guaranteed to match live usage.

    Parameters:
    -----------
    color_stops : dict[int, tuple[str, str]]
        Dict mapping bin number (1-indexed) to (start_hex, end_hex) tuples,
        e.g., ``DICT_QUARTILE_HEATMAP_COLORMAP``.
    output_path : str | None
        Directory path to save the plot. If None, saves to the repo root.
    zero_color : str
        Color for the null/zero band at the bottom (default: ``DEFAULT_NULL_COLOR``).
    n_gradient_steps : int
        Number of interpolation steps per bin (default: 256).
    null_steps : int | None
        Height in pixels of the null/zero color band at the bottom. Defaults to
        1/10 the height of one bin (``n_gradient_steps // 10``).
    figsize : tuple[float, float]
        Figure size in inches (width, height). Default: (1, 5).
    """
    plt.rcParams["font.family"] = "Arial"

    n_bins = len(color_stops)
    # null band height: half the height of one bin by default
    if null_steps is None:
        null_steps = n_gradient_steps // 10

    # generate gradient via percentile_colormap with synthetic uniform values so
    # the legend is guaranteed consistent with live usage of percentile_colormap
    n_total = n_bins * n_gradient_steps
    synthetic_values = [(i + 1) / n_total for i in range(n_total)]
    gradient_hex = percentile_colormap(
        synthetic_values, color_stops, zero_color=zero_color
    )

    # build single-column RGBA array: null band at bottom, gradient above
    null_rgba = mcolors.to_rgba(zero_color)
    colors_col = [null_rgba] * null_steps + [mcolors.to_rgba(c) for c in gradient_hex]
    gradient = np.array(colors_col)[:, np.newaxis, :]  # shape: (total_rows, 1, 4)
    total_height = null_steps + n_total

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        gradient,
        aspect="auto",
        origin="lower",
        extent=[0, 1, 0, total_height],
    )

    # "0" tick at the null/gradient boundary (where green starts); then one tick
    # per bin boundary up to "1" at the top
    tick_positions = [null_steps + i * n_gradient_steps for i in range(n_bins + 1)]
    tick_labels = ["0", "0.25", "0.5", "0.75", "1"]

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.yaxis.set_ticks_position("right")
    ax.set_title("Mutational\ndensity", fontsize=10, loc="center", pad=6)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()

    filename = "colormap_legend"
    desc = "Colormap legend"
    if output_path is None:
        save_plot(fig=fig, output_filename=filename, plot_type=desc)
    else:
        save_plot(
            fig=fig,
            output_filename=filename,
            output_path=output_path,
            plot_type=desc,
            bool_force_local=False,
        )


DICT_BIOCHEM_PROP_COLORS = {
    "Charge": "#1f77b4",
    "Volume": "#ff7f0e",
    "Polarity": "#2ca02c",
}
"""dict[str, str]: Dictionary mapping biochemical properties to colors.
Keys are property names (e.g., "Charge", "Volume", "Polarity"), and values are hex color codes.
This dictionary can be used to look up colors for biochemical properties in visualizations.
"""
