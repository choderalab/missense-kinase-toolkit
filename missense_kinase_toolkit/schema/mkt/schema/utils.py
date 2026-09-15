"""Schema utility helpers: recursive attribute access, UUID generation, and kinase-group adjudication.

Provides :func:`rgetattr`/:func:`rsetattr` for traversing nested Pydantic models,
:func:`random_uuid`, :func:`return_kinase_gene_set`,
:func:`adjudicate_kinase_group`, and the KLIFS-to-Dunbrack-MSA correspondence
helpers :func:`return_klifs2msa_dict`/:func:`return_catalytic_klifs2msa_dict`.
"""

import logging
import os
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mkt.schema.kinase_schema import KinaseInfo

logger = logging.getLogger(__name__)

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)
"""Default tqdm bar format with comma-separated thousands in counts."""


def query_date_from_file(path: str) -> str | None:
    """Return a source file's modification date (ISO) for :class:`~mkt.schema.kinase_schema.Provenance`.

    A freshly downloaded file's mtime is its download date; an existing local file's mtime is
    when it was last modified -- so this covers both the re-download and local-file cases.
    Shared by the databases source loaders (KinCoRe FASTA/CIF, the Dunbrack MSA, ...).

    Parameters
    ----------
    path : str
        Path to the source file.

    Returns
    -------
    str | None
        ISO date string (YYYY-MM-DD), or None if the file is absent.
    """
    if not os.path.exists(path):
        return None
    return date.fromtimestamp(os.path.getmtime(path)).isoformat()


def fill_missing_none(value: dict | None, keys) -> dict | None:
    """Return a dict with every key present (missing -> None), for TOML round-tripping.

    TOML has no null type, so serializing a dict then deserializing drops keys whose value is
    None. Rebuilding the full key set (missing -> None) makes the round-trip lossless. Shared by
    the KLIFS2UniProt, SASA, and MSA field validators. A None ``value`` (an unset optional field)
    is passed through as None rather than filled.

    Parameters
    ----------
    value : dict | None
        The (possibly gap-dropped) mapping, or None for an unset optional field.
    keys : Iterable
        The full ordered key set the mapping should cover.

    Returns
    -------
    dict | None
        ``value`` with every key present (missing -> None), or None if ``value`` is None.
    """
    if value is None:
        return None
    dict_temp = dict.fromkeys(keys, None)
    if isinstance(value, dict):
        dict_temp.update(value)
    return dict_temp


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


LIST_MANIFEST_EXTRA_PATHS = ["klifs.pocket_seq", "KLIFS2UniProtIdx", "KLIFS2UniProtSeq"]
"""list[str]: Non-sub-model :class:`KinaseInfo` fields also counted in the build manifest."""


def return_submodel_paths(model_cls=None, str_prefix: str = "") -> list[str]:
    """Return dotted paths to every nested Pydantic sub-model field, depth-first.

    ``Provenance`` fields are skipped; their versions are tallied by
    :func:`return_manifest_tallies`.

    Parameters
    ----------
    model_cls : type[BaseModel] | None, optional
        Model to walk, by default None (:class:`KinaseInfo`).
    str_prefix : str, optional
        Prefix prepended to each path, by default "".

    Returns
    -------
    list[str]
        Dotted sub-model paths (e.g. ``"kincore.cif.sasa"``).
    """
    from typing import get_args, get_origin

    from mkt.schema.kinase_schema import KinaseInfo, Provenance
    from pydantic import BaseModel

    if model_cls is None:
        model_cls = KinaseInfo

    list_paths = []
    for name, field in model_cls.model_fields.items():
        for sub_cls in get_args(field.annotation) or (field.annotation,):
            if (
                get_origin(sub_cls) is None
                and isinstance(sub_cls, type)
                and issubclass(sub_cls, BaseModel)
                and sub_cls is not Provenance
            ):
                path = f"{str_prefix}{name}"
                list_paths.append(path)
                list_paths.extend(return_submodel_paths(sub_cls, f"{path}."))
    return list_paths


def return_manifest_tallies(
    dict_kinase: dict[str, "KinaseInfo"],
    list_paths: list[str] | None = None,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Count non-None values and tally ``source.version`` per dotted path in one pass.

    Shared by the manifest writer and the load-time check so the two cannot drift.

    Parameters
    ----------
    dict_kinase : dict[str, KinaseInfo]
        Kinase dictionary to tally.
    list_paths : list[str] | None, optional
        Paths to tally, by default None (all sub-models plus
        :data:`LIST_MANIFEST_EXTRA_PATHS`).

    Returns
    -------
    tuple[dict[str, int], dict[str, dict[str, int]]]
        Path -> non-None count, and path -> version -> count (paths without a
        versioned source omitted).
    """
    from collections import Counter

    if list_paths is None:
        list_paths = return_submodel_paths() + LIST_MANIFEST_EXTRA_PATHS

    dict_counts = dict.fromkeys(list_paths, 0)
    dict_versions = {path: Counter() for path in list_paths}
    for obj in dict_kinase.values():
        for path in list_paths:
            if rgetattr(obj, path) is None:
                continue
            dict_counts[path] += 1
            version = rgetattr(obj, f"{path}.source.version")
            if version is not None:
                dict_versions[path][version] += 1

    return dict_counts, {
        path: dict(sorted(counter.items()))
        for path, counter in dict_versions.items()
        if counter
    }


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


def extract_sequence_from_cif(kincore) -> str | None:
    """Extract the one-letter sequence from a KinCoRe CIF, if present.

    Shared by ``KinaseInfo.extract_sequence_from_cif`` and the
    ``KinaseInfoKinaseDomainGenerator`` FASTA-to-CIF alignment validator so both read
    the CIF sequence the same way. Duck-typed on ``kincore.cif.cif`` to avoid importing
    the schema models here.

    Parameters
    ----------
    kincore : KinCoRe | None
        A ``KinCoRe`` object (or None); the sequence is read from
        ``kincore.cif.cif["_entity_poly.pdbx_seq_one_letter_code"]``.

    Returns
    -------
    str | None
        The one-letter CIF sequence with newlines stripped, or None if unavailable.
    """
    key_seq = "_entity_poly.pdbx_seq_one_letter_code"
    try:
        if kincore is not None and kincore.cif is not None:
            return kincore.cif.cif[key_seq][0].replace("\n", "")
    except Exception:
        pass
    return None


def split_domain_suffix(name: str) -> tuple[str, str]:
    """Split a trailing multi-domain index suffix off a kinase name.

    Multi-domain kinases carry a ``_<n>`` suffix denoting the individual kinase domain
    (e.g. ``"JAK1_1"``); ``n`` may have several digits.

    Parameters
    ----------
    name : str
        Kinase name, optionally carrying a ``_<digits>`` domain suffix.

    Returns
    -------
    tuple[str, str]
        ``(base, suffix)`` where ``suffix`` is ``"_<digits>"`` or ``""``
        (e.g. ``"JAK1_10" -> ("JAK1", "_10")``, ``"SGK1" -> ("SGK1", "")``).
    """
    base, sep, tail = name.rpartition("_")
    if sep and base and tail.isdigit():
        return base, f"{sep}{tail}"
    return name, ""


def return_kinase_gene_set(dict_kinase: dict | None = None) -> set[str]:
    """Return the set of kinase HGNC gene symbols for gene-level membership tests.

    Multi-kinase-domain genes are keyed with a ``_1`` / ``_2`` domain suffix (see
    :func:`split_domain_suffix`); those are collapsed to the base gene symbol
    (e.g. ``JAK1_1`` / ``JAK1_2`` -> ``JAK1``) so a symbol like ``"JAK1"`` is not
    missed when testing membership against cohort gene symbols.

    Parameters
    ----------
    dict_kinase : dict | None
        Mapping keyed by kinase name (optionally carrying a ``_<digit>`` domain
        suffix). If None, the canonical ``DICT_KINASE`` is deserialized.

    Returns
    -------
    set[str]
        Base HGNC gene symbols of every kinase.
    """
    if dict_kinase is None:
        from mkt.schema.io_utils import deserialize_kinase_dict

        dict_kinase = deserialize_kinase_dict(
            str_name="DICT_KINASE", bool_verbose=False
        )
    return {split_domain_suffix(name)[0] for name in dict_kinase}


SET_HOMOLOG_DO_NOT_MERGE = frozenset([frozenset({"BUB1", "BUB1B"})])
"""frozenset[frozenset[str]]: Hand-curated name pairs that :func:`group_name_homologs`
must never collapse together, overriding the prefix heuristic. BUB1 / BUB1B are distinct
genes (BUB1B is BUBR1) whose ``B`` suffix is not the receptor ``R`` case caught generically."""


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
    A complete gene name and its ``stem + "R"`` receptor paralog are never merged (e.g.
    INSR / INSRR stay apart), and any pair in :data:`SET_HOMOLOG_DO_NOT_MERGE` is held
    apart by hand (BUB1 / BUB1B).

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
        variants (so e.g. FGFR1-4, PRKACA/B/G, NEK1/10 group; EPHA10/EPHB6 do not).

        A bare stem member plus its ``stem + "R"`` receptor paralog is never merged
        (e.g. INSR / INSRR), nor is any pair in :data:`SET_HOMOLOG_DO_NOT_MERGE`."""
        base_set = set(bases)
        if any(pair <= base_set for pair in SET_HOMOLOG_DO_NOT_MERGE):
            return False
        p = _stem(bases)
        if len(p) < min_prefix:
            return False
        # a complete gene name and its receptor paralog (stem + "R") are distinct genes
        if p in base_set and (p + "R") in base_set:
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


def return_kinase_group_dict(dict_kinase: dict[str, "KinaseInfo"]) -> dict[str, str]:
    """Return each kinase's adjudicated group, adding bare multi-domain gene symbols.

    A bare symbol (e.g. ``"JAK1"`` for ``JAK1_1``/``JAK1_2``) takes its domains' shared
    group, or ``"Multiple"`` when they differ. Source of
    :data:`mkt.schema.constants.DICT_KINASE_GROUP`.

    Parameters
    ----------
    dict_kinase : dict[str, KinaseInfo]
        Mapping of kinase name to kinase object.

    Returns
    -------
    dict[str, str]
        Kinase name (suffixed or bare) -> group, sorted by name; kinases without a group
        are omitted.
    """
    dict_group: dict[str, str] = {}
    dict_base_groups: dict[str, set[str]] = {}
    for name, obj in dict_kinase.items():
        group = obj.adjudicate_group()
        if group is None:
            continue
        dict_group[name] = str(group)
        base, suffix = split_domain_suffix(name)
        if suffix:
            dict_base_groups.setdefault(base, set()).add(dict_group[name])
    for base, set_groups in dict_base_groups.items():
        dict_group[base] = set_groups.pop() if len(set_groups) == 1 else "Multiple"
    return dict(sorted(dict_group.items()))


def return_lipid_kinase_set(dict_kinase: dict[str, "KinaseInfo"]) -> frozenset[str]:
    """Return lipid kinase names, adding bare multi-domain symbols whose domains all qualify.

    Source of :data:`mkt.schema.constants.SET_LIPID_KINASE`.

    Parameters
    ----------
    dict_kinase : dict[str, KinaseInfo]
        Mapping of kinase name to kinase object.

    Returns
    -------
    frozenset[str]
        Lipid kinase names (suffixed or bare).
    """
    set_lipid = {name for name, obj in dict_kinase.items() if obj.is_lipid_kinase()}
    dict_base_names: dict[str, list[str]] = {}
    for name in dict_kinase:
        base, suffix = split_domain_suffix(name)
        if suffix:
            dict_base_names.setdefault(base, []).append(name)
    set_bare = {
        base
        for base, list_names in dict_base_names.items()
        if all(name in set_lipid for name in list_names)
    }
    return frozenset(set_lipid | set_bare)


def adjudicate_kinase_group(str_kinase: str, bool_lipid: bool = True) -> str | None:
    """Adjudicate the kinase group for a kinase name or bare multi-domain gene symbol.

    Reads the precomputed :data:`~mkt.schema.constants.DICT_KINASE_GROUP` and
    :data:`~mkt.schema.constants.SET_LIPID_KINASE`, so it never loads ``DICT_KINASE``. A
    bare multi-domain symbol returns its domains' shared group, or ``"Multiple"`` when
    they differ.

    Parameters
    ----------
    str_kinase : str
        Kinase name (e.g. ``"PIK3CA"``, ``"JAK1_1"``) or bare gene symbol (``"JAK1"``).
    bool_lipid : bool, optional
        Classify lipid kinases as ``"Lipid"``, by default True.

    Returns
    -------
    str | None
        The group (e.g. ``"Lipid"``, ``"TK"``, ``"Multiple"``), or None if the kinase is
        unknown.
    """
    from mkt.schema.constants import DICT_KINASE_GROUP, SET_LIPID_KINASE

    if bool_lipid and str_kinase in SET_LIPID_KINASE:
        return "Lipid"
    return DICT_KINASE_GROUP.get(str_kinase)


def return_klifs2msa_dict(
    dict_kinase: dict[str, "KinaseInfo"],
    bool_return_concordance: bool = False,
) -> dict[str, str] | tuple[dict[str, str], dict[str, float]]:
    """Assemble the empirical KLIFS-pocket -> Dunbrack-MSA position correspondence.

    For each KLIFS ``region:idx``, tallies which MSA ``region2uniprot`` key most often shares
    its UniProt index across the kinases carrying both maps, and returns that modal
    correspondence (built programmatically from ``dict_kinase`` -- not hard-coded).

    The concordance is **not 1:1**: it is ~99% at the core catalytic/structural anchors (e.g.
    ``III:17`` VAIK Lys, ``c.l:70`` HRD Asp, ``xDFG:81`` DFG Asp) but drops to ~90-95% across
    the variable alphaD/alphaE/linker region, where the two structure-based alignment conventions
    place insert-flanking residues differently, and for a handful of divergent (pseudo)kinases.
    Use it for cross-referencing/QA, not as an exact map. Pass ``bool_return_concordance`` to also
    get the per-position agreement fraction (share of kinases mapping to the modal MSA key).

    Parameters
    ----------
    dict_kinase : dict[str, KinaseInfo]
        Mapping of HGNC name to kinase object (needs both ``KLIFS2UniProtIdx`` and
        ``kincore.msa``).
    bool_return_concordance : bool, optional
        If True, also return the per-KLIFS-position agreement fraction, by default False.

    Returns
    -------
    dict[str, str] | tuple[dict[str, str], dict[str, float]]
        The KLIFS ``region:idx`` -> MSA ``region:idx`` map; with ``bool_return_concordance``,
        a ``(map, concordance)`` tuple where concordance is the modal-agreement fraction per
        KLIFS position.
    """
    from collections import Counter

    from mkt.schema.constants import LIST_KLIFS_REGION

    dict_counter: dict[str, Counter] = {label: Counter() for label in LIST_KLIFS_REGION}
    for obj in dict_kinase.values():
        msa = obj.kincore.msa if obj.kincore is not None else None
        if msa is None or obj.KLIFS2UniProtIdx is None:
            continue
        dict_idx2msa = {v: k for k, v in msa.region2uniprot.items() if v is not None}
        for klifs_label, uniprot_idx in obj.KLIFS2UniProtIdx.items():
            if uniprot_idx is not None and uniprot_idx in dict_idx2msa:
                dict_counter[klifs_label][dict_idx2msa[uniprot_idx]] += 1

    dict_map: dict[str, str] = {}
    dict_concordance: dict[str, float] = {}
    for label in LIST_KLIFS_REGION:
        counter = dict_counter[label]
        if not counter:
            continue
        msa_label, count = counter.most_common(1)[0]
        dict_map[label] = msa_label
        dict_concordance[label] = count / sum(counter.values())

    if bool_return_concordance:
        return dict_map, dict_concordance
    return dict_map


def return_catalytic_klifs2msa_dict(
    dict_kinase: dict[str, "KinaseInfo"],
) -> dict[str, str]:
    """Return the KLIFS -> MSA correspondence restricted to the catalytic positions.

    Subsets :func:`return_klifs2msa_dict` to :data:`LIST_KLIFS_CATALYTIC`; the source of
    the precomputed :data:`mkt.schema.constants.DICT_KLIFS2MSA_CATALYTIC` read by
    :meth:`mkt.schema.kinase_schema.KinaseInfo.return_catalytic_residues`.

    The catalytic anchors are where the two alignments agree most closely (~99% modal
    concordance at III:17, c.l:68-70 and xDFG:81-83; ~97% at the beta2 lysine II:13), which
    is what makes the fallback defensible where the general map is not.

    Parameters
    ----------
    dict_kinase : dict[str, KinaseInfo]
        Mapping of HGNC name to kinase object (see :func:`return_klifs2msa_dict`).

    Returns
    -------
    dict[str, str]
        KLIFS region:idx -> MSA region:idx, for the catalytic positions only.
    """
    from mkt.schema.constants import LIST_KLIFS_CATALYTIC

    dict_map = return_klifs2msa_dict(dict_kinase)
    return {
        label: dict_map[label] for label in LIST_KLIFS_CATALYTIC if label in dict_map
    }
