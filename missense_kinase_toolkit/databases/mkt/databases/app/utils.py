import logging

from mkt.databases.app.sequences import SequenceAlignment
from mkt.databases.app.structures import StructureVisualizer
from mkt.schema.io_utils import deserialize_kinase_dict

logger = logging.getLogger(__name__)


def generate_sequence_and_structure_viewers(
    str_kinase: str,
    dict_colors: dict[str, str],
    str_attr: str | None = None,
    sequence_kwargs: dict | None = None,
    structure_kwargs: dict | None = None,
) -> tuple[SequenceAlignment, StructureVisualizer]:
    """Generate a structure viewer for the given kinase.

    Parameters
    ----------
    str_kinase : str
        Gene name of the kinase.
    dict_colors : dict[str, str]
        Dictionary of colors for the residues to be highlighted.
    str_attr : str | None, optional
        Attribute to be highlighted in the structure, by default None.
    sequence_kwargs : dict | None, optional
        Additional keyword arguments to pass to SequenceAlignment, by default None.
    structure_kwargs : dict | None, optional
        Additional keyword arguments to pass to StructureVisualizer (e.g., dict_mutations), by default None.

    Returns
    -------
    tuple[SequenceAlignment, StructureVisualizer]
        Tuple containing SequenceAlignment and StructureVisualizer objects for the given kinase.
    """
    DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE")
    obj_temp = DICT_KINASE[str_kinase]

    # Initialize kwargs dictionaries if None
    sequence_kwargs = sequence_kwargs or {}
    structure_kwargs = structure_kwargs or {}

    obj_align = SequenceAlignment(obj_temp, dict_colors, **sequence_kwargs)
    obj_viz = StructureVisualizer(
        obj_kinase=obj_temp,
        dict_align=obj_align.dict_align,
        str_attr=str_attr,
        **structure_kwargs
    )

    return obj_align, obj_viz
