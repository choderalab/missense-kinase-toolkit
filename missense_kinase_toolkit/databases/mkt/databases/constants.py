"""Constants and helpers shared across the genomic-coordinate API clients.

Several clients (:mod:`mkt.databases.ensembl`, :mod:`mkt.databases.genomenexus`)
serve each genome build from a different REST host and therefore need the same
build-alias normalization and host lookup. Those shared pieces live here --
:data:`DICT_BUILD_ALIAS` with :func:`normalize_build` and :func:`resolve_rest_host`,
plus the JSON request headers -- so each client only declares its own build-to-host
mapping. It also holds :data:`RefSeqRNAPattern` for the cBioPortal ``refseqMrnaId``
field, :data:`EntrezGeneIDPattern` for its ``entrezGeneId`` field and
:data:`MissenseProteinChangePattern` for its ``proteinChange`` field.
"""

DICT_BUILD_ALIAS = {
    "GRCH37": "GRCh37",
    "37": "GRCh37",
    "HG19": "GRCh37",
    "B37": "GRCh37",
    "GRCH38": "GRCh38",
    "38": "GRCh38",
    "HG38": "GRCh38",
}
"""dict[str, str]: Upper-cased genome-build aliases to the canonical assembly name
(cBioPortal ``ncbiBuild`` is inconsistent -- ``"37"``/``"hg19"`` also mean GRCh37)."""

DICT_HEADER_JSON = {"Accept": "application/json"}
"""dict[str, str]: Header requesting a JSON response body."""

DICT_HEADER_JSON_POST = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}
"""dict[str, str]: Header for POST requests sending and receiving JSON."""

RefSeqRNAPrefixes = ("NM", "NR", "XM", "XR")
"""tuple[str, ...]: RefSeq RNA accession prefixes -- NM_/NR_ curated mRNA/ncRNA,
XM_/XR_ model (predicted) mRNA/ncRNA."""

RefSeqRNAPattern = rf"(?:{'|'.join(RefSeqRNAPrefixes)})_(?:\d{{6}}|\d{{9}})(?:\.\d+)?"
"""str: One RefSeq RNA accession -- 6 or 9 digits after the prefix, optional version
(e.g. ``NM_000546.5``). Unanchored so it can be matched against each element of a
cBioPortal ``refseqMrnaId`` cell; use ``re.fullmatch`` to validate a single value."""

RefSeqAccessionShape = r"[A-Z]{2}_\d+"
"""str: Any RefSeq accession (two-letter prefix, underscore, digits), RNA or not;
tells an unexpected accession apart from a placeholder such as ``"NA"`` or ``"."``."""

RefSeqCuratedCodingPrefixes = ("NM",)
"""tuple[str, ...]: Curated protein-coding RefSeq prefixes."""

RefSeqCodingPrefixes = ("NM", "XM")
"""tuple[str, ...]: Protein-coding RefSeq prefixes, curated and model; NR_/XR_ are
non-coding and have no protein translation."""

EntrezGeneIDPattern = r"[1-9]\d*"
"""str: One NCBI Entrez Gene ID -- a positive integer with no fixed width (``1`` for
A1BG to 9 digits); use ``re.fullmatch`` to validate a single value."""

AminoAcids = "ACDEFGHIKLMNPQRSTVWY"
"""str: One-letter codes of the 20 standard amino acids."""

MissenseProteinChangePattern = rf"([{AminoAcids}])(\d+)([{AminoAcids}])"
"""str: One single-residue substitution (e.g. ``V600E``): reference, position,
alternate."""


def normalize_build(build: object) -> str | None:
    """Return the canonical assembly name for a genome-build alias.

    Parameters
    ----------
    build : object
        Build as found in the cBioPortal ``ncbiBuild`` column (e.g. ``"37"``, ``"hg19"``,
        ``"GRCh38"``); missing values are allowed.

    Returns
    -------
    str | None
        ``"GRCh37"`` or ``"GRCh38"``, or None if ``build`` is not a known alias.
    """
    return DICT_BUILD_ALIAS.get(str(build).upper())


def resolve_rest_host(build: str, dict_host: dict[str, str]) -> str:
    """Return the REST host serving a genome build for a given API.

    Parameters
    ----------
    build : str
        Genome build/assembly name as found in the ``ncbiBuild`` column of
        cBioPortal mutations; common aliases (e.g. ``"37"``, ``"hg19"``) are
        normalized via :data:`DICT_BUILD_ALIAS`.
    dict_host : dict[str, str]
        Mapping of canonical assembly name to the API's base URL for that build.

    Returns
    -------
    str
        Base URL of the host serving that build.

    Raises
    ------
    ValueError
        If the build (after alias normalization) is not a key of ``dict_host``.
    """
    canonical = normalize_build(build)
    if canonical in dict_host:
        return dict_host[canonical]
    raise ValueError(
        f"Unsupported genome build {build!r}; expected one of "
        f"{sorted(dict_host)} (aliases: {sorted(DICT_BUILD_ALIAS)})."
    )
