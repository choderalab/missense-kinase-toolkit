import logging
from typing import TYPE_CHECKING

import webcolors
from mkt.databases.app.sequences import SequenceAlignment
from mkt.databases.app.structures import StructureVisualizer

if TYPE_CHECKING:
    from mkt.databases.app.schema import StructureConfig

logger = logging.getLogger(__name__)


def create_structure_visualizer(
    seq_align: SequenceAlignment,
    config_class: type["StructureConfig"],
    config_kwargs: dict | None = None,
) -> StructureVisualizer:
    """Create a StructureVisualizer from a SequenceAlignment and config class.

    This is the recommended way to create a StructureVisualizer with the new
    architecture. The flow is:
    1. Create SequenceAlignment
    2. Pass it to this function with a config class
    3. Get back a StructureVisualizer ready for visualization

    Parameters
    ----------
    seq_align : SequenceAlignment
        SequenceAlignment object with aligned sequences.
    config_class : Type[StructureConfig]
        The config class to instantiate (e.g., PhosphositesConfig, KLIFSConservedConfig).
    config_kwargs : dict | None, optional
        Additional keyword arguments to pass to the config class, by default None.

    Returns
    -------
    StructureVisualizer
        StructureVisualizer object ready for visualization.

    Examples
    --------
    >>> from mkt.databases.app.sequences import SequenceAlignment
    >>> from mkt.databases.app.schema import PhosphositesConfig
    >>>
    >>> # Create sequence alignment
    >>> seq_align = SequenceAlignment(str_kinase="EGFR", dict_color={"A": "blue", ...})
    >>>
    >>> # Create structure visualizer
    >>> viz = create_structure_visualizer(seq_align, PhosphositesConfig)
    >>>
    >>> # Get highlight data
    >>> list_idx, dict_color, dict_style = viz.get_highlight_data()
    """
    config_kwargs = config_kwargs or {}
    config = config_class(seq_align=seq_align, **config_kwargs)
    viz = StructureVisualizer(config)
    return viz


def convert_color_to_hex(color: str) -> str:
    """Convert named color to hex.

    Parameters
    ----------
    color : str
        Color name or hex string.

    Returns
    -------
    str
        Hex color string.
    """
    if color.startswith("#"):
        return color

    try:
        return webcolors.name_to_hex(color)
    except ValueError:
        logger.warning(f"Color '{color}' not recognized, defaulting to gray")
        return "#808080"


def interpolate_color(
    norm_value: float, start_color_hex: str, end_color_hex: str
) -> str:
    """Interpolate between two colors based on normalized value.

    Parameters
    ----------
    norm_value : float
        Normalized value between 0 and 1.
    start_color_hex : str
        Starting color in hex format (e.g., "#FFFFFF").
    end_color_hex : str
        Ending color in hex format (e.g., "#FF0000").

    Returns
    -------
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
