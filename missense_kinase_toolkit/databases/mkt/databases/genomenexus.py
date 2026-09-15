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

import datetime
import json
import logging
import re

from mkt.databases import requests_wrapper
from mkt.databases.constants import DICT_HEADER_JSON_POST, resolve_rest_host
from mkt.schema.kinase_schema import Exon, Provenance
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


def build_exon_map(record: dict, protein_length: int) -> dict[int, int] | None:
    """Map each 1-based protein position to its exon number from a transcript record.

    Subtracts the UTR overlap from each exon to get its coding span, walks the exons in
    transcription (``rank``) order to lay out the CDS, and assigns residue ``r`` to the exon
    containing the central nucleotide of its codon (CDS position ``3r-1``) -- so a codon split
    across an exon boundary is assigned by its middle base. Strand-agnostic: ``rank`` is already
    coding order and the UTR overlap is a genomic-coordinate intersection.

    Parameters
    ----------
    record : dict
        A GenomeNexus canonical-transcript record with ``exons`` and ``utrs``.
    protein_length : int
        Expected protein length; the CDS must be ``protein_length * 3 + 3`` (with the stop codon)
        or the record is rejected (None) as inconsistent with the UniProt sequence.

    Returns
    -------
    dict[int, int] | None
        Mapping of 1-based protein position to 1-based exon number, or None if the exon/UTR
        structure is missing or inconsistent with ``protein_length``.
    """
    exons = record.get("exons") or []
    utrs = record.get("utrs") or []
    if not exons:
        return None

    # coding span per exon = exon length minus its overlap with any UTR
    spans: list[tuple[int, int]] = []  # (rank, coding_len)
    for exon in sorted(exons, key=lambda e: e["rank"]):
        start, end = exon["exonStart"], exon["exonEnd"]
        overlap = sum(
            max(0, min(end, u["end"]) - max(start, u["start"]) + 1) for u in utrs
        )
        spans.append((exon["rank"], (end - start + 1) - overlap))

    if sum(length for _, length in spans) != protein_length * 3 + 3:
        return None

    # lay out the CDS: rank -> [cds_start, cds_end] (1-based, coding 5'->3')
    ranges: list[tuple[int, int, int]] = []  # (rank, cds_start, cds_end)
    pos = 0
    for rank, length in spans:
        if length > 0:
            ranges.append((rank, pos + 1, pos + length))
            pos += length

    # residue r's codon centers on CDS position 3r-1; assign r to that base's exon
    idx2exon: dict[int, int] = {}
    for residue in range(1, protein_length + 1):
        center = 3 * residue - 1
        for rank, cds_start, cds_end in ranges:
            if cds_start <= center <= cds_end:
                idx2exon[residue] = rank
                break
    return idx2exon


def enrich_kinases_with_exons(
    dict_targets: dict,
    builds: tuple[str, ...] = ("GRCh37", "GRCh38"),
) -> None:
    """Annotate ``KinaseInfo`` objects with a per-residue exon map in place.

    Fetches the canonical transcript (UniProt isoform override) for each distinct gene, builds
    the protein-position -> exon-number map, and stamps it on ``obj.exon`` (recording the build
    it came from). Multi-domain entries (``HGNC_1``/``HGNC_2``) share the gene's transcript, so
    they share the same map. ``builds`` are tried in order: an entry unresolved on the primary
    build (no transcript, or a protein length disagreeing with the UniProt canonical sequence) is
    retried on the next, and left unset (logged) if no build resolves it.

    Parameters
    ----------
    dict_targets : dict
        Mapping of ``hgnc_name`` (possibly ``_1``/``_2``-suffixed) to ``KinaseInfo`` to enrich.
    builds : tuple[str, ...]
        Genome builds to try in order; GRCh37 first matches the cBioPortal MSK-IMPACT
        coordinates, GRCh38 recovers genes absent/renamed on GRCh37.

    Returns
    -------
    None
    """
    query_date = datetime.date.today().isoformat()
    remaining = set(dict_targets)

    for build in builds:
        if not remaining:
            break
        genes = sorted({re.sub(r"_\d+$", "", name) for name in remaining})
        dict_transcript = get_canonical_transcripts(genes, build=build)
        for hgnc_name in list(remaining):
            obj_kinase = dict_targets[hgnc_name]
            record = dict_transcript.get(re.sub(r"_\d+$", "", hgnc_name))
            if record is None:
                continue
            idx2exon = build_exon_map(record, len(obj_kinase.uniprot.canonical_seq))
            if idx2exon is None:
                continue
            obj_kinase.exon = Exon(
                transcript_id=record.get("transcriptId"),
                build=build,
                n_exons=len(record.get("exons") or []),
                idx2exon=idx2exon,
                source=Provenance(
                    name="GenomeNexus canonical transcript",
                    citation="de Bruijn et al., 2022.",
                    doi="https://doi.org/10.1200/CCI.21.00144",
                    query_date=query_date,
                ),
            )
            remaining.discard(hgnc_name)

    for hgnc_name in sorted(remaining):
        logger.warning(f"no consistent exon map for {hgnc_name} on {list(builds)}...")
