"""Schema utility helpers: recursive attribute access, UUID generation, and kinase-group adjudication.

Provides :func:`rgetattr`/:func:`rsetattr` for traversing nested Pydantic models,
:func:`random_uuid`, and :func:`adjudicate_kinase_group`.
"""

import logging

logger = logging.getLogger(__name__)

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)
"""Default tqdm bar format with comma-separated thousands in counts."""


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


def split_domain_suffix(name: str) -> tuple[str, str]:
    """Split a trailing multi-domain index suffix off a kinase name.

    Multi-domain kinases are represented with a ``_1`` / ``_2`` suffix denoting the
    individual kinase domains (e.g. ``"JAK1_1"`` is the first kinase domain of JAK1).

    Parameters
    ----------
    name : str
        Kinase name, optionally carrying a ``_<digit>`` domain suffix.

    Returns
    -------
    tuple[str, str]
        ``(base, suffix)`` where ``suffix`` is ``"_<digit>"`` or ``""``
        (e.g. ``"JAK1_1" -> ("JAK1", "_1")``, ``"BTK" -> ("BTK", "")``).
    """
    if len(name) >= 2 and name[-2] == "_" and name[-1].isdigit():
        return name[:-2], name[-2:]
    return name, ""


def group_name_homologs(
    names: list[str], min_prefix: int = 3
) -> list[tuple[str, list[str]]]:
    """Collapse prefix-homologous kinase names into compact labeled groups.

    Names are grouped when they share a base prefix of at least ``min_prefix``
    characters, differ only by a base suffix of at most one character, and carry the
    same multi-domain suffix (see :func:`split_domain_suffix`). Each group is labeled
    by factoring out the common prefix, e.g.
    ``["JAK1_1", "JAK2_1", "JAK3_1"] -> ("JAK1/2/3_1 (3)", [...])``. The default
    ``min_prefix`` of 3 keeps coincidental two-character matches apart (e.g. the
    unrelated ATM and ATR).

    Parameters
    ----------
    names : list[str]
        Kinase names to group. Order is not significant: names are sorted by
        ``(domain_suffix, name)`` internally so homologs are adjacent.
    min_prefix : int, optional
        Minimum shared base-prefix length required to merge, by default 3.

    Returns
    -------
    list[tuple[str, list[str]]]
        ``(label, members)`` pairs in sorted order; a singleton is ``(name, [name])``.
    """

    def _lcp(s: str, t: str) -> int:
        k = 0
        while k < len(s) and k < len(t) and s[k] == t[k]:
            k += 1
        return k

    items = sorted(
        (
            (base, dom, nm)
            for nm, (base, dom) in ((nm, split_domain_suffix(nm)) for nm in names)
        ),
        key=lambda it: (it[1], it[2]),
    )
    out, i = [], 0
    while i < len(items):
        j = i + 1
        while j < len(items) and items[j][1] == items[i][1]:  # same domain suffix
            bases = [items[k][0] for k in range(i, j + 1)]
            p = bases[0]
            for b in bases[1:]:
                p = p[: _lcp(p, b)]
            if len(p) < min_prefix or any(len(b[len(p) :]) > 1 for b in bases):
                break
            j += 1
        group = items[i:j]
        members = [it[2] for it in group]
        bases, dom = [it[0] for it in group], group[0][1]
        if len(group) == 1:
            label = bases[0] + dom
        else:
            p = bases[0]
            for b in bases[1:]:
                p = p[: _lcp(p, b)]
            label = (
                bases[0]
                + "/"
                + "/".join(b[len(p) :] for b in bases[1:])
                + dom
                + f" ({len(group)})"
            )
        out.append((label, members))
        i = j
    return out


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
