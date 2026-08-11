"""Ensembl REST client for reference-sequence and trinucleotide context.

Provides :class:`EnsemblSequence` (a single cached region fetch) plus batch and
helper functions for deriving the trinucleotide (SBS) context of genomic variants
from the Ensembl reference. The genome build selects the REST host
(:data:`DICT_ENSEMBL_REST_HOST`): GRCh37 has a dedicated host
(``grch37.rest.ensembl.org``) while GRCh38 uses the default ``rest.ensembl.org``,
so the build must match the coordinate system or the returned bases are wrong.
"""

import json
import logging
from dataclasses import field

from mkt.databases import requests_wrapper
from mkt.databases.api_schema import RESTAPIClient
from mkt.schema.utils import TQDM_BAR_FORMAT
from pydantic.dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)


DICT_ENSEMBL_REST_HOST = {
    "GRCh37": "https://grch37.rest.ensembl.org",
    "GRCh38": "https://rest.ensembl.org",
}
"""dict[str, str]: Ensembl REST host per genome build; GRCh38 uses the default host."""

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

ENSEMBL_POST_REGION_MAX = 50
"""int: Maximum number of regions accepted per ``/sequence/region`` POST request."""


def rest_host(build: str) -> str:
    """Return the Ensembl REST host for a genome build.

    Parameters
    ----------
    build : str
        Genome build/assembly name as found in the ``ncbiBuild`` column of
        cBioPortal mutations; common aliases (e.g. ``"37"``, ``"hg19"``) are
        normalized via :data:`DICT_BUILD_ALIAS`.

    Returns
    -------
    str
        Base URL of the Ensembl REST host serving that build.

    Raises
    ------
    ValueError
        If the build (after alias normalization) is not one of
        :data:`DICT_ENSEMBL_REST_HOST`.
    """
    canonical = DICT_BUILD_ALIAS.get(str(build).upper())
    if canonical in DICT_ENSEMBL_REST_HOST:
        return DICT_ENSEMBL_REST_HOST[canonical]
    raise ValueError(
        f"Unsupported genome build {build!r}; expected one of "
        f"{sorted(DICT_ENSEMBL_REST_HOST)} (aliases: {sorted(DICT_BUILD_ALIAS)})."
    )


DICT_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
"""dict[str, str]: Watson-Crick complement of each (upper-case) base."""


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a nucleotide sequence.

    Parameters
    ----------
    seq : str
        Nucleotide sequence (case-insensitive); ``N`` is passed through.

    Returns
    -------
    str
        Reverse-complemented, upper-cased sequence.
    """
    return "".join(DICT_COMPLEMENT[base] for base in reversed(seq.upper()))


def sbs_trinucleotide_context(context: str, ref: str, alt: str) -> dict:
    """Fold a trinucleotide context to the pyrimidine-reference SBS convention.

    The COSMIC single-base-substitution (SBS) 96-channel scheme requires the
    mutated (reference) base to be a pyrimidine (C or T). When ``ref`` is a purine
    (A or G) the trinucleotide and the substitution are reverse-complemented so the
    channel is strand-agnostic.

    Parameters
    ----------
    context : str
        Plus-strand trinucleotide (5' flank, reference base, 3' flank); the center
        base must equal ``ref``.
    ref : str
        Reference (wild-type) base on the plus strand.
    alt : str
        Variant (alternate) base on the plus strand.

    Returns
    -------
    dict
        ``{"context", "ref", "alt", "channel"}`` after pyrimidine folding, where
        ``channel`` is the ``"X[R>A]Y"`` label (e.g. ``"T[C>T]C"``).
    """
    context, ref, alt = context.upper(), ref.upper(), alt.upper()
    if ref in ("A", "G"):
        context = reverse_complement(context)
        ref = reverse_complement(ref)
        alt = reverse_complement(alt)
    return {
        "context": context,
        "ref": ref,
        "alt": alt,
        "channel": f"{context[0]}[{ref}>{alt}]{context[2]}",
    }


@dataclass
class EnsemblSequence(RESTAPIClient):
    """Fetch a single reference region via the Ensembl REST API.

    Wraps ``GET /sequence/region/{species}/{region}`` through the shared cached
    session, so repeated lookups of the same region are served from cache. The
    host is selected from :attr:`build` (see :func:`rest_host`). For many sites at
    once prefer :func:`get_trinucleotide_contexts`, which batches via POST.
    """

    chromosome: str | int
    """Chromosome name without a ``chr`` prefix (e.g. ``"14"``, ``"X"``)."""
    start: int
    """Region start (1-based, inclusive)."""
    end: int
    """Region end (1-based, inclusive)."""
    build: str = "GRCh37"
    """Genome build selecting the REST host (e.g. ``"GRCh37"``, ``"GRCh38"``)."""
    species: str = "human"
    """Ensembl species name."""
    strand: int = 1
    """Strand to return (``1`` plus, ``-1`` minus); MAF coordinates are plus-strand."""
    header: dict = field(default_factory=lambda: {"Accept": "application/json"})
    """Header for the API request."""

    def __post_init__(self):
        self.create_query_url()
        self.query_api()

    def create_query_url(self) -> None:
        """Build the sequence-region query URL from the coordinates and build."""
        region = f"{self.chromosome}:{self.start}..{self.end}:{self.strand}"
        self.url_query = (
            f"{rest_host(self.build)}/sequence/region/{self.species}/{region}"
        )

    def query_api(self) -> None:
        """Query the Ensembl sequence-region API and store the sequence."""
        res = requests_wrapper.get_cached_session().get(
            self.url_query,
            headers=self.header,
        )
        self._stamp_from_response(res)

        if res.ok:
            self.sequence = res.json().get("seq")
        else:
            logger.error("Error: %s", res.status_code)
            self.sequence = None


def get_cds_sequence(
    transcript_id: str,
    build: str = "GRCh37",
) -> str | None:
    """Return the coding sequence (CDS) of a transcript in coding 5'->3' order.

    Wraps ``GET /sequence/id/{id}?type=cds`` through the shared cached session.
    The sequence is already oriented in the coding direction (reverse-complemented
    for minus-strand transcripts), so index ``3k-3 .. 3k`` is the k-th codon.

    Parameters
    ----------
    transcript_id : str
        Ensembl transcript id (e.g. ``"ENST00000349310"``).
    build : str
        Genome build selecting the REST host (e.g. ``"GRCh37"``, ``"GRCh38"``).

    Returns
    -------
    str | None
        The upper-cased CDS nucleotide sequence, or None if the fetch failed.
    """
    res = requests_wrapper.get_cached_session().get(
        f"{rest_host(build)}/sequence/id/{transcript_id}",
        params={"type": "cds"},
        headers={"Accept": "application/json"},
    )
    if not res.ok:
        logger.error("Error: %s", res.status_code)
        return None
    seq = res.json().get("seq")
    return seq.upper() if seq else None


def get_trinucleotide_context(
    chromosome: str | int,
    position: int,
    build: str = "GRCh37",
    species: str = "human",
) -> str | None:
    """Return the plus-strand trinucleotide centered on a genomic position.

    Parameters
    ----------
    chromosome : str | int
        Chromosome name without a ``chr`` prefix.
    position : int
        1-based genomic position of the variant.
    build : str
        Genome build of the coordinate (e.g. ``"GRCh37"``, ``"GRCh38"``).
    species : str
        Ensembl species name.

    Returns
    -------
    str | None
        The 3-bp reference context ``[position-1, position, position+1]`` on the
        plus strand, or None if the fetch failed.
    """
    seq = EnsemblSequence(
        chromosome=chromosome,
        start=position - 1,
        end=position + 1,
        build=build,
        species=species,
    ).sequence
    return seq.upper() if seq else None


def get_trinucleotide_contexts(
    sites: list[tuple[str | int, int]],
    build: str = "GRCh37",
    species: str = "human",
    chunk_size: int = ENSEMBL_POST_REGION_MAX,
    progress: bool = True,
) -> dict[tuple[str, int], str | None]:
    """Batch-fetch plus-strand trinucleotide contexts for many genomic sites.

    Uses ``POST /sequence/region/{species}`` (up to
    :data:`ENSEMBL_POST_REGION_MAX` regions per request), which is far fewer round
    trips than one GET per site. All ``sites`` must share the same genome
    ``build`` (the host is build-specific); group by build before calling.
    De-duplicate ``sites`` before calling. POST responses are not written to the
    requests cache, so persist the returned table if it needs to survive runs.

    Parameters
    ----------
    sites : list[tuple[str | int, int]]
        ``(chromosome, position)`` pairs; positions are 1-based.
    build : str
        Genome build of the coordinates (e.g. ``"GRCh37"``, ``"GRCh38"``).
    species : str
        Ensembl species name.
    chunk_size : int
        Regions per POST request; capped at :data:`ENSEMBL_POST_REGION_MAX`.
    progress : bool
        Show the per-chunk progress bar; set False when called inside a caller's
        own progress loop to avoid a nested bar per call.

    Returns
    -------
    dict[tuple[str, int], str | None]
        Mapping of ``(str(chromosome), position)`` to its 3-bp plus-strand context
        (None where the region returned no sequence).
    """
    chunk_size = min(chunk_size, ENSEMBL_POST_REGION_MAX)
    url = f"{rest_host(build)}/sequence/region/{species}"
    header = {"Content-Type": "application/json", "Accept": "application/json"}
    session = requests_wrapper.get_cached_session()

    dict_context: dict[tuple[str, int], str | None] = {}
    for idx in tqdm(
        range(0, len(sites), chunk_size),
        desc="Querying trinucleotide context in Ensembl",
        bar_format=TQDM_BAR_FORMAT,
        disable=not progress,
    ):
        chunk = sites[idx : idx + chunk_size]
        regions = [f"{chrom}:{pos - 1}..{pos + 1}" for chrom, pos in chunk]
        res = session.post(url, headers=header, data=json.dumps({"regions": regions}))

        if not res.ok:
            logger.error("Error: %s", res.status_code)
            dict_context.update({(str(chrom), pos): None for chrom, pos in chunk})
            continue

        # the API preserves input order within a POST batch
        for (chrom, pos), entry in zip(chunk, res.json()):
            seq = entry.get("seq")
            dict_context[(str(chrom), pos)] = seq.upper() if seq else None

    return dict_context
