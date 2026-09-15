"""Reconcile cBioPortal protein positions onto canonical UniProt coordinates.

cBioPortal reports ``proteinChange`` against the transcript a study was annotated with (for
MSK-IMPACT, the MSKCC isoform override), which is not always the UniProt canonical.
:class:`CanonicalReconciler` maps each position through a cascade of source sequences
(:class:`SourceTier`), accepting a candidate only when the canonical residue equals the
reported reference residue, and :func:`select_domain_name` assigns a multi-domain gene's
position to the kinase domain that contains it.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from mkt.databases.aligners import BL2UniProtAligner
from mkt.databases.ensembl import get_protein_sequence
from mkt.databases.genomenexus import annotate_variants, get_canonical_transcripts
from mkt.databases.ncbi import get_cds_translation
from strenum import StrEnum

logger = logging.getLogger(__name__)


class SourceTier(StrEnum):
    """Source sequence that reconciled a position (the ``reconcile_source`` values)."""

    direct = "direct"
    mskcc = "mskcc"
    refseq = "refseq"
    genomenexus = "genomenexus"


TUPLE_DEFAULT_TIERS = (
    SourceTier.direct,
    SourceTier.mskcc,
    SourceTier.refseq,
    SourceTier.genomenexus,
)
"""tuple[SourceTier, ...]: Tiers tried in order; the Genome Nexus tier always runs last."""


def map_positions_by_alignment(
    seq_source: str, seq_target: str, aligner=None
) -> dict[int, int]:
    """Map 1-based positions of ``seq_source`` onto ``seq_target`` by alignment.

    Residues in aligned blocks map one-to-one; a source residue aligned to a gap (e.g. an
    isoform-only exon) has no key.

    Parameters
    ----------
    seq_source : str
        Sequence whose positions are mapped (e.g. an isoform's protein).
    seq_target : str
        Sequence mapped onto (e.g. the UniProt canonical).
    aligner : object, optional
        Aligner whose ``align(seq_source, seq_target)`` returns Biopython alignments, by
        default None (a global :class:`~mkt.databases.aligners.BL2UniProtAligner`).

    Returns
    -------
    dict[int, int]
        1-based source position -> 1-based target position.
    """
    if aligner is None:
        aligner = BL2UniProtAligner()
    alignment = aligner.align(seq_source, seq_target)[0]
    return {
        int(s0 + offset + 1): int(t0 + offset + 1)
        for (s0, s1), (t0, _) in zip(alignment.aligned[0], alignment.aligned[1])
        for offset in range(s1 - s0)
    }


def clean_refseq_accession(value: object) -> str | None:
    """Return the first RefSeq mRNA (``NM_``) accession in a ``refseqMrnaId`` cell.

    Parameters
    ----------
    value : object
        Raw cell value, possibly quoted, comma-separated, or missing.

    Returns
    -------
    str | None
        The first ``NM_`` accession, or None if there is none.
    """
    if not isinstance(value, str):
        return None
    accession = value.strip().strip('"').split(",")[0].strip()
    return accession if accession.startswith("NM_") else None


def select_domain_name(
    str_symbol: str,
    list_names: Sequence[str],
    dict_name2span: dict[str, tuple[int | None, int | None]],
    idx: int | None,
) -> tuple[str, bool | None]:
    """Return the kinase-domain name whose span contains a canonical position.

    Parameters
    ----------
    str_symbol : str
        Bare gene symbol, returned when the position lies in no domain.
    list_names : Sequence[str]
        The gene's kinase names (e.g. ``["JAK1_1", "JAK1_2"]`` or ``["BRAF"]``).
    dict_name2span : dict[str, tuple[int | None, int | None]]
        Kinase name -> adjudicated ``(start, end)``; spans with a None bound are skipped.
    idx : int | None
        1-based canonical position.

    Returns
    -------
    tuple[str, bool | None]
        ``(name, in_kinase_domain)``: the containing domain (the smallest span, with a
        warning, if spans overlap) and True, the bare symbol and False when no domain
        contains it, or the bare symbol and None when ``idx`` is None.
    """
    if idx is None:
        return str_symbol, None
    list_inside = []
    for name in list_names:
        start, end = dict_name2span.get(name, (None, None))
        if start is not None and end is not None and start <= idx <= end:
            list_inside.append((end - start, name))
    if not list_inside:
        return str_symbol, False
    if len(list_inside) > 1:
        logger.warning(
            f"{str_symbol} position {idx} lies in overlapping domains "
            f"{sorted(name for _, name in list_inside)}; using the smallest span."
        )
    return min(list_inside)[1], True


@dataclass
class CanonicalReconciler:
    """Map cBioPortal protein positions onto canonical UniProt coordinates."""

    dict_canonical: dict[str, str]
    """Gene symbol -> UniProt canonical sequence."""
    tuple_sources: tuple[SourceTier, ...] = TUPLE_DEFAULT_TIERS
    """Tiers to try, by default :data:`TUPLE_DEFAULT_TIERS`."""
    str_override: str = "mskcc"
    """Isoform-override source of the ``mskcc`` transcript tier, by default "mskcc"."""
    str_genomenexus_override: str = "uniprot"
    """Isoform-override source when re-annotating variants, by default "uniprot"."""
    str_build: str = "GRCh37"
    """Genome build for Genome Nexus and Ensembl, by default "GRCh37"."""
    fetch_transcripts: Callable[..., dict[str, dict]] = get_canonical_transcripts
    """Bulk gene -> transcript record fetcher, by default Genome Nexus."""
    fetch_ensembl_protein: Callable[..., str | None] = get_protein_sequence
    """Ensembl transcript -> protein fetcher, by default Ensembl REST."""
    fetch_refseq_protein: Callable[[str], str | None] = get_cds_translation
    """RefSeq mRNA -> protein fetcher, by default NCBI efetch."""
    fetch_annotations: Callable[..., dict[str, dict]] = annotate_variants
    """Bulk genomic variant -> consequence summary fetcher, by default Genome Nexus."""
    _dict_transcript: dict[str, str | None] = field(default_factory=dict, init=False)
    """Gene symbol -> override transcript id (None if the gene has none)."""
    _dict_seq: dict[str, str | None] = field(default_factory=dict, init=False)
    """Accession -> fetched protein sequence (None if the fetch failed)."""
    _dict_map: dict[tuple[str, str], dict[int, int]] = field(
        default_factory=dict, init=False
    )
    """(gene symbol, accession) -> source position -> canonical position."""

    def prime(self, genes: Sequence[str]) -> None:
        """Fetch the override transcript for every uncached gene in one bulk call.

        Parameters
        ----------
        genes : Sequence[str]
            Gene symbols; those without a canonical sequence are ignored.

        Returns
        -------
        None
        """
        if SourceTier.mskcc not in self.tuple_sources:
            return
        list_genes = sorted(
            {
                gene
                for gene in genes
                if gene in self.dict_canonical and gene not in self._dict_transcript
            }
        )
        if not list_genes:
            return
        dict_records = self.fetch_transcripts(
            list_genes, build=self.str_build, isoform_override=self.str_override
        )
        for gene in list_genes:
            record = dict_records.get(gene)
            self._dict_transcript[gene] = record.get("transcriptId") if record else None

    def _is_match(self, str_symbol: str, idx: int | None, str_aa_ref: str) -> bool:
        """Return whether canonical position ``idx`` holds the reported reference residue."""
        seq = self.dict_canonical.get(str_symbol)
        return (
            seq is not None
            and idx is not None
            and 1 <= idx <= len(seq)
            and seq[idx - 1] == str_aa_ref
        )

    def _position_map(
        self, str_symbol: str, accession: str, fetch: Callable[[str], str | None]
    ) -> dict[int, int]:
        """Return the cached source -> canonical position map for one accession."""
        key = (str_symbol, accession)
        if key not in self._dict_map:
            if accession not in self._dict_seq:
                self._dict_seq[accession] = fetch(accession)
            seq_source = self._dict_seq[accession]
            seq_canonical = self.dict_canonical[str_symbol]
            if seq_source is None:
                self._dict_map[key] = {}
            elif seq_source == seq_canonical:
                self._dict_map[key] = {i: i for i in range(1, len(seq_canonical) + 1)}
            else:
                self._dict_map[key] = map_positions_by_alignment(
                    seq_source, seq_canonical
                )
        return self._dict_map[key]

    def _fetch_ensembl(self, transcript_id: str) -> str | None:
        """Fetch an Ensembl transcript's protein on the configured build."""
        return self.fetch_ensembl_protein(transcript_id, build=self.str_build)

    def reconcile(
        self,
        str_symbol: str,
        idx_position: int | None,
        str_aa_ref: str | None,
        refseq: object = None,
    ) -> tuple[int | None, SourceTier | None]:
        """Reconcile one position through the direct, mskcc and refseq tiers.

        The Genome Nexus tier needs genomic coordinates and runs batched in
        :meth:`reconcile_many`.

        Parameters
        ----------
        str_symbol : str
            Gene symbol.
        idx_position : int | None
            1-based protein position as reported by cBioPortal.
        str_aa_ref : str | None
            Reported reference residue.
        refseq : object, optional
            Raw ``refseqMrnaId`` cell, by default None.

        Returns
        -------
        tuple[int | None, SourceTier | None]
            The canonical position and the tier that accepted it, or ``(None, None)``.
        """
        if str_symbol not in self.dict_canonical or idx_position is None:
            return None, None
        if not str_aa_ref:
            return None, None
        for tier in self.tuple_sources:
            if tier is SourceTier.direct:
                candidate = idx_position
            elif tier is SourceTier.mskcc:
                self.prime([str_symbol])
                transcript_id = self._dict_transcript.get(str_symbol)
                if transcript_id is None:
                    continue
                candidate = self._position_map(
                    str_symbol, transcript_id, self._fetch_ensembl
                ).get(idx_position)
            elif tier is SourceTier.refseq:
                accession = clean_refseq_accession(refseq)
                if accession is None:
                    continue
                candidate = self._position_map(
                    str_symbol, accession, self.fetch_refseq_protein
                ).get(idx_position)
            else:
                continue
            if self._is_match(str_symbol, candidate, str_aa_ref):
                return candidate, tier
        return None, None

    def reconcile_many(
        self,
        list_symbol: Sequence[str],
        list_position: Sequence[int | None],
        list_aa_ref: Sequence[str | None],
        list_refseq: Sequence[object] | None = None,
        list_hgvsg: Sequence[str | None] | None = None,
    ) -> tuple[list[int | None], list[SourceTier | None]]:
        """Reconcile many positions, batching the Genome Nexus tier over the residual.

        Parameters
        ----------
        list_symbol : Sequence[str]
            Gene symbol per row.
        list_position : Sequence[int | None]
            Reported 1-based protein position per row.
        list_aa_ref : Sequence[str | None]
            Reported reference residue per row.
        list_refseq : Sequence[object] | None, optional
            Raw ``refseqMrnaId`` cell per row, by default None.
        list_hgvsg : Sequence[str | None] | None, optional
            Genomic HGVS per row (e.g. ``"7:g.140453136A>T"``) for the Genome Nexus tier,
            by default None.

        Returns
        -------
        tuple[list[int | None], list[SourceTier | None]]
            Canonical position and accepting tier per row (None where irreconcilable).
        """
        n_rows = len(list_symbol)
        list_refseq = list(list_refseq) if list_refseq is not None else [None] * n_rows
        list_hgvsg = list(list_hgvsg) if list_hgvsg is not None else [None] * n_rows

        self.prime(list_symbol)
        list_idx: list[int | None] = []
        list_source: list[SourceTier | None] = []
        for symbol, position, aa_ref, refseq in zip(
            list_symbol, list_position, list_aa_ref, list_refseq
        ):
            idx, source = self.reconcile(symbol, position, aa_ref, refseq)
            list_idx.append(idx)
            list_source.append(source)

        if SourceTier.genomenexus not in self.tuple_sources:
            return list_idx, list_source
        dict_row2variant = {
            i: list_hgvsg[i]
            for i in range(n_rows)
            if list_source[i] is None
            and list_hgvsg[i]
            and list_symbol[i] in self.dict_canonical
            and list_aa_ref[i]
        }
        if not dict_row2variant:
            return list_idx, list_source
        dict_annotation = self.fetch_annotations(
            sorted(set(dict_row2variant.values())),
            build=self.str_build,
            isoform_override=self.str_genomenexus_override,
        )
        for i, variant in dict_row2variant.items():
            summary = dict_annotation.get(variant) or {}
            transcript_id = summary.get("transcriptId")
            position = summary.get("proteinPosition")
            if isinstance(position, dict):
                position = position.get("start")
            if transcript_id is None or position is None:
                continue
            candidate = self._position_map(
                list_symbol[i], transcript_id, self._fetch_ensembl
            ).get(int(position))
            if self._is_match(list_symbol[i], candidate, list_aa_ref[i]):
                list_idx[i], list_source[i] = candidate, SourceTier.genomenexus
        return list_idx, list_source
