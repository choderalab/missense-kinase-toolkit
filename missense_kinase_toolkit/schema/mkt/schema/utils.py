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
    names: list[str], min_prefix: int = 3, show_count: bool = True
) -> list[tuple[str, list[str]]]:
    """Collapse prefix-homologous kinase names into compact labeled groups.

    Names are grouped when they share a common base stem of at least ``min_prefix``
    characters, each member's remaining variant is a single letter or a pure number
    (so distinct subfamilies such as EPHA10 / EPHB6 stay apart), and they carry the
    same multi-domain suffix (see :func:`split_domain_suffix`). The stem is trimmed
    back off any mid-number boundary so a shorter member number is not split out of a
    longer one (NEK1 vs NEK10/NEK11), and numeric variants are sorted numerically.
    Each group is labeled by factoring out the stem, e.g.
    ``["JAK1_1", "JAK2_1", "JAK3_1"] -> ("JAK1/2/3_1 (3)", [...])`` or
    ``["NEK1", "NEK10", "NEK2"] -> ("NEK1/2/10", [...])``. The default ``min_prefix``
    of 3 keeps coincidental two-character matches apart (e.g. the unrelated ATM, ATR).

    Parameters
    ----------
    names : list[str]
        Kinase names to group. Order is not significant: names are sorted by
        ``(domain_suffix, name)`` internally so homologs are adjacent.
    min_prefix : int, optional
        Minimum shared base-stem length required to merge, by default 3.
    show_count : bool, optional
        Append a ``" (N)"`` member count to each merged group's label, by default True.

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

    def _stem(bases: list[str]) -> str:
        """Common prefix, trimmed back off a mid-number boundary."""
        p = bases[0]
        for b in bases[1:]:
            p = p[: _lcp(p, b)]
        while (
            p
            and p[-1].isdigit()
            and any(len(b) > len(p) and b[len(p)].isdigit() for b in bases)
        ):
            p = p[:-1]
        return p

    def _is_homolog_set(bases: list[str]) -> bool:
        """True if the bases share a >=min_prefix stem with single-letter / pure-number
        variants (so e.g. FGFR1-4, PRKACA/B/G, NEK1/10 group; EPHA10/EPHB6 do not)."""
        p = _stem(bases)
        if len(p) < min_prefix:
            return False
        return all(
            (suf := b[len(p) :]) == ""
            or suf.isdigit()
            or (len(suf) == 1 and suf.isalpha())
            for b in bases
        )

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
        while (
            j < len(items)
            and items[j][1] == items[i][1]  # same domain suffix
            and _is_homolog_set([items[k][0] for k in range(i, j + 1)])
        ):
            j += 1
        group = items[i:j]
        members = [it[2] for it in group]
        bases, dom = [it[0] for it in group], group[0][1]
        if len(group) == 1:
            label = bases[0] + dom
        else:
            p = _stem(bases)
            variants = [b[len(p) :] for b in bases]
            if all(v == "" or v.isdigit() for v in variants):
                variants = sorted(variants, key=lambda v: int(v) if v else -1)
            else:
                variants = sorted(variants)
            label = (
                p
                + "/".join(variants)
                + dom
                + (f" ({len(group)})" if show_count else "")
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
