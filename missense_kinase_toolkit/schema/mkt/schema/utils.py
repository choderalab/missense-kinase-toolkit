import logging

logger = logging.getLogger(__name__)


def rgetattr(obj, attr, *args):
    """Get attribute from object recursively.

    Parameters
    ----------
    obj : Any
        Object to get attribute from.
    attr : str
        Attribute to get.
    *args : Any
        Any additional arguments to pass to getattr.

    Returns
    -------
    Any
        Value of attribute if found.
    """
    import functools

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    try:
        return functools.reduce(_getattr, [obj] + attr.split("."))
    except AttributeError:
        return None


def rsetattr(obj, attr, val):
    """Set attribute from object recursively.

    Parameters
    ----------
    obj : Any
        Object to get attribute from.
    attr : str
        Attribute to get.
    val : Any
        Value to set attribute to.

    Returns
    -------
    Any
        Value of attribute if found, otherwise default value.
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# adapted from: https://nathanielknight.ca/articles/consistent_random_uuids_in_python.html
def random_uuid():
    """Generate a random UUID that allows to set a seed.

    Returns
    -------
    str
        A random UUID as a string.
    """
    import random
    import uuid

    return uuid.UUID(bytes=bytes(random.getrandbits(8) for _ in range(16)), version=4)


def adjudicate_kinase_group(str_kinase: str, bool_lipid: bool = True) -> str | None:
    """Adjudicates the kinase group for a given kinase.

    Parameters
    ----------
    str_kinase : str
        The name of the kinase (e.g., "PIK3CA").
    bool_lipid : bool, optional
        Flag to indicate if lipid kinases should be classified as "Lipid" group, by default True.

    Returns
    -------
    str | None
        The adjudicated kinase group (e.g., "Lipid", "TK", "CMGC"), or None if the kinase is not found.
    """
    from mkt.schema.io_utils import deserialize_kinase_dict

    DICT_KINASE = deserialize_kinase_dict(str_name="DICT_KINASE", bool_verbose=False)

    if str_kinase not in DICT_KINASE:
        return None
    kinase = DICT_KINASE[str_kinase]
    if kinase.is_lipid_kinase() and bool_lipid:
        return "Lipid"
    return kinase.adjudicate_group()
