"""Genome Nexus REST client for variant annotation and canonical transcripts.

Genome Nexus (``genomenexus.org``) is the VEP-based annotation engine behind
cBioPortal. This wraps its build-specific REST hosts
(:data:`DICT_GENOME_NEXUS_HOST`) for two uses:

- :func:`get_canonical_transcripts` -- the canonical Ensembl transcript per gene
  (exon/UTR structure, protein length, UniProt id), under an isoform-override
  source (default ``"uniprot"``) so the transcript aligns with the UniProt
  canonical sequence used by ``KinaseInfo``/KLIFS.
- :func:`annotate_variants` -- VEP-style consequence for genomic variants
  (protein position, coding change, consequence), matching how cBioPortal
  annotated the observed mutations.

Genome Nexus serves GRCh37 from the default host (``www.genomenexus.org``) and
GRCh38 from ``grch38.genomenexus.org``, so the build must match the coordinates.
"""

import json
import logging

from mkt.databases import requests_wrapper
from mkt.databases.constants import DICT_HEADER_JSON_POST, resolve_rest_host
from mkt.schema.utils import TQDM_BAR_FORMAT
from tqdm import tqdm

logger = logging.getLogger(__name__)


DICT_GENOME_NEXUS_HOST = {
    "GRCh37": "https://www.genomenexus.org",
    "GRCh38": "https://grch38.genomenexus.org",
}
"""dict[str, str]: Genome Nexus REST host per genome build; GRCh37 is the default host."""

GENOME_NEXUS_POST_MAX = 500
"""int: Conservative maximum number of items to send per POST request."""

DEFAULT_ISOFORM_OVERRIDE = "uniprot"
"""str: Isoform-override source; ``"uniprot"`` picks the UniProt-canonical transcript."""


def rest_host(build: str) -> str:
    """Return the Genome Nexus REST host for a genome build.

    Parameters
    ----------
    build : str
        Genome build/assembly name as found in the ``ncbiBuild`` column of
        cBioPortal mutations; common aliases (e.g. ``"37"``, ``"hg19"``) are
        normalized via :data:`mkt.databases.constants.DICT_BUILD_ALIAS`.

    Returns
    -------
    str
        Base URL of the Genome Nexus host serving that build.

    Raises
    ------
    ValueError
        If the build (after alias normalization) is not one of
        :data:`DICT_GENOME_NEXUS_HOST`.
    """
    return resolve_rest_host(build, DICT_GENOME_NEXUS_HOST)


def _post_chunks(url, params, payload, chunk_size, desc):
    """POST ``payload`` in chunks and yield each chunk's parsed JSON list.

    Parameters
    ----------
    url : str
        Endpoint URL.
    params : dict
        Query parameters (e.g. isoform override, fields).
    payload : list
        Items to send; split into ``chunk_size`` blocks.
    chunk_size : int
        Items per request (capped at :data:`GENOME_NEXUS_POST_MAX`).
    desc : str
        tqdm progress-bar description.

    Yields
    ------
    list
        The parsed JSON list returned for each chunk (empty on a failed chunk).
    """
    chunk_size = min(chunk_size, GENOME_NEXUS_POST_MAX)
    header = DICT_HEADER_JSON_POST
    session = requests_wrapper.get_cached_session()
    for idx in tqdm(
        range(0, len(payload), chunk_size),
        desc=desc,
        bar_format=TQDM_BAR_FORMAT,
    ):
        chunk = payload[idx : idx + chunk_size]
        res = session.post(url, params=params, headers=header, data=json.dumps(chunk))
        if not res.ok:
            logger.error("Error: %s", res.status_code)
            yield []
            continue
        yield res.json()


def get_canonical_transcripts(
    genes: list[str],
    build: str = "GRCh37",
    isoform_override: str = DEFAULT_ISOFORM_OVERRIDE,
    chunk_size: int = GENOME_NEXUS_POST_MAX,
) -> dict[str, dict]:
    """Fetch the canonical Ensembl transcript per gene from Genome Nexus.

    Parameters
    ----------
    genes : list[str]
        HGNC gene symbols.
    build : str
        Genome build selecting the host (e.g. ``"GRCh37"``, ``"GRCh38"``).
    isoform_override : str
        Isoform-override source; ``"uniprot"`` aligns transcripts with the UniProt
        canonical sequence.
    chunk_size : int
        Genes per request (capped at :data:`GENOME_NEXUS_POST_MAX`).

    Returns
    -------
    dict[str, dict]
        Mapping of gene symbol to its transcript record (``transcriptId``,
        ``proteinLength``, ``uniprotId``, ``refseqMrnaId``, ``exons``, ``utrs``,
        ...). Genes with no canonical transcript are absent.
    """
    url = f"{rest_host(build)}/ensembl/canonical-transcript/hgnc"
    params = {"isoformOverrideSource": isoform_override}
    dict_transcript: dict[str, dict] = {}
    for records in _post_chunks(
        url, params, genes, chunk_size, "Querying canonical transcripts in Genome Nexus"
    ):
        for rec in records:
            for symbol in rec.get("hugoSymbols") or []:
                dict_transcript[symbol] = rec
    return dict_transcript


def annotate_variants(
    variants: list[str],
    build: str = "GRCh37",
    isoform_override: str = DEFAULT_ISOFORM_OVERRIDE,
    chunk_size: int = GENOME_NEXUS_POST_MAX,
) -> dict[str, dict]:
    """Annotate genomic variants with Genome Nexus (VEP) consequence summaries.

    Parameters
    ----------
    variants : list[str]
        HGVS genomic strings, e.g. ``"7:g.140453136A>T"``.
    build : str
        Genome build selecting the host (e.g. ``"GRCh37"``, ``"GRCh38"``).
    isoform_override : str
        Isoform-override source; ``"uniprot"`` aligns transcripts with the UniProt
        canonical sequence.
    chunk_size : int
        Variants per request (capped at :data:`GENOME_NEXUS_POST_MAX`).

    Returns
    -------
    dict[str, dict]
        Mapping of variant string to its ``transcriptConsequenceSummary``
        (``proteinPosition``, ``hgvsc``, ``codonChange``, ``consequenceTerms``,
        ``refSeq``, ``transcriptId``, ``uniprotId``, ...). Variants that failed to
        annotate are absent.
    """
    url = f"{rest_host(build)}/annotation"
    params = {"isoformOverrideSource": isoform_override, "fields": "annotation_summary"}
    dict_annotation: dict[str, dict] = {}
    for records in _post_chunks(
        url, params, variants, chunk_size, "Annotating variants in Genome Nexus"
    ):
        for rec in records:
            summary = (rec.get("annotation_summary") or {}).get(
                "transcriptConsequenceSummary"
            )
            if summary is not None:
                dict_annotation[rec.get("variant")] = summary
    return dict_annotation
