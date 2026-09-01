"""Annotate kinases with the Modi & Dunbrack (2019) structure-based kinase MSA.

Sources the Human-PK alignment (497 canonical protein-kinase domains) on demand, maps each
domain's aligned row to UniProt canonical coordinates, and stores an :class:`MSA` on
``KinCore.msa`` -- creating an MSA-only ``KinCore`` shell for the few multi-KD second domains
that are in the alignment but lack KinCore structure. Positions are keyed KLIFS-style over the
229 Aurora-reference columns ("B1N:001".."HI:229") so the activation loop (DFG in ALN through
APE in ALC) is readable in UniProt coordinates. The handful of kinases whose UniProt
isoform/numbering differs from our canonical sequence are reconciled by local alignment.
"""

import logging
import os
import re
from collections import defaultdict

from mkt.databases.io_utils import DataSource
from mkt.schema.constants import DICT_MSA_ALIGNED_REGION, DICT_MSA_COL2LABEL
from mkt.schema.io_utils import get_repo_root
from mkt.schema.kinase_schema import MSA, KinCore, Provenance

logger = logging.getLogger(__name__)

MSA_SOURCE = DataSource(
    name="Human-PK-alignment.fasta",
    path=os.path.join(get_repo_root(), "data", "Human-PK-alignment.fasta"),
    url=(
        "https://dunbrack.fccc.edu/kincore/static/downloads/alignment-files/"
        "Human-PK-alignment.fasta"
    ),
    citation="10.1038/s41598-019-56499-4",  # Modi & Dunbrack, Sci Rep 2019
)
"""DataSource: Dunbrack KinCore Human-PK structure-based alignment (on-demand download)."""

ANNOTATION_PREFIX = "0ANNOTATION"
"""str: Header prefix of the alignment's annotation row (skipped)."""

_ALIGNED_ITEMS = list(DICT_MSA_ALIGNED_REGION.items())


def _parse_header(header: str) -> dict:
    """Parse an alignment header into a per-domain entry dict.

    ``>AGC_AKT1/150-408 AKT1_HUMAN AKT1 P31749`` ->
    ``{"hgnc": "AKT1", "uniprot": "P31749", "start": 150, "end": 408}``. Multi-KD domains keep
    their suffix (``TYR_JAK1_2/875-1151`` -> ``hgnc="JAK1_2"``), matching the ``hgnc_name`` keys
    of the KinaseInfo dict; the UniProt accession backs the synonym/alias fallback.

    Parameters
    ----------
    header : str
        The FASTA header line (without the leading ``>``).

    Returns
    -------
    dict
        ``{"hgnc", "uniprot", "start", "end"}`` for this domain.
    """
    parts = header.split()
    group_hgnc, rng = parts[0].split("/")
    start, end = (int(x) for x in rng.split("-"))
    return {
        "hgnc": group_hgnc.split("_", 1)[1],  # drop the leading group token
        "uniprot": parts[-1],
        "start": start,
        "end": end,
    }


def parse_msa(path: str | None = None) -> list[dict]:
    """Parse the alignment into a list of per-domain entries.

    Parameters
    ----------
    path : str | None, optional
        Path to the alignment FASTA; None resolves (and downloads) the default.

    Returns
    -------
    list[dict]
        One ``{"hgnc", "uniprot", "start", "end", "seq"}`` per aligned domain (annotation row
        skipped); ``seq`` is the gapped aligned row.
    """
    if path is None:
        path = MSA_SOURCE.resolve()
    entries: list[dict] = []
    header = None
    chunks: list[str] = []

    def _flush():
        if header is not None and not header.startswith(ANNOTATION_PREFIX):
            entry = _parse_header(header)
            entry["seq"] = "".join(chunks)
            entries.append(entry)

    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                _flush()
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        _flush()
    return entries


def _col2uniprot(
    aligned_seq: str, canonical_seq: str, start: int, end: int
) -> tuple[dict[int, int], int, int, bool]:
    """Map alignment columns to UniProt canonical indices for one domain.

    Assigns the header's UniProt numbering when the ungapped alignment row matches the
    canonical slice; otherwise reconciles by local alignment (the few isoform/numbering-shifted
    kinases) so the columns map to our canonical sequence at 100% identity.

    Parameters
    ----------
    aligned_seq : str
        The gapped aligned row for this domain.
    canonical_seq : str
        The UniProt canonical sequence of the kinase.
    start, end : int
        The header's 1-based UniProt start/end.

    Returns
    -------
    tuple[dict[int, int], int, int, bool]
        ``(col2uniprot, kd_start, kd_end, reconciled)`` -- a map of 1-based alignment column ->
        1-based UniProt index, the KD start/end in UniProt coords, and whether local alignment
        was needed.
    """
    # non-gap columns (1-based) and the residues they carry (upper-cased: lowercase = insert)
    cols = [(i + 1, ch.upper()) for i, ch in enumerate(aligned_seq) if ch != "-"]
    ungapped = "".join(ch for _, ch in cols)

    # direct: header numbering already matches our canonical slice
    if ungapped == canonical_seq[start - 1 : end]:
        col2uniprot = {col: start + off for off, (col, _) in enumerate(cols)}
        return col2uniprot, start, end, False

    # reconcile: local-align the ungapped row to our canonical (isoform/numbering shift)
    from mkt.databases.aligners import MSA2UniProtAligner

    alignment = MSA2UniProtAligner().align(canonical_seq, ungapped)[0]

    # map each ungapped position -> canonical index via the (gapless, 100%-identity) blocks
    pos2uniprot: dict[int, int] = {}
    for (t0, t1), (q0, q1) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(t1 - t0):
            pos2uniprot[q0 + offset] = t0 + offset + 1  # 1-based canonical index
    col2uniprot = {
        col: pos2uniprot[pos] for pos, (col, _) in enumerate(cols) if pos in pos2uniprot
    }
    mapped = sorted(col2uniprot.values())
    kd_start = mapped[0] if mapped else start
    kd_end = mapped[-1] if mapped else end
    return col2uniprot, kd_start, kd_end, True


def _build_regions(aligned_seq: str) -> dict[str, str]:
    """Slice the aligned row into ordered aligned + unaligned region strings.

    Parameters
    ----------
    aligned_seq : str
        The gapped aligned row for this domain.

    Returns
    -------
    dict[str, str]
        Ordered ``{region label: gapped substring}`` -- 17 aligned blocks ("B1N".."HI") and the
        16 unaligned regions between them ("B1N~B1C"...), in kinase-domain N->C order.
    """
    regions: dict[str, str] = {}
    for i, (name, (c0, c1)) in enumerate(_ALIGNED_ITEMS):
        regions[name] = aligned_seq[c0 - 1 : c1]
        if i + 1 < len(_ALIGNED_ITEMS):
            next_name, (nc0, _) = _ALIGNED_ITEMS[i + 1]
            if nc0 > c1 + 1:
                regions[f"{name}~{next_name}"] = aligned_seq[c1 : nc0 - 1]
    return regions


def build_msa_model(
    aligned_seq: str,
    canonical_seq: str,
    start: int,
    end: int,
    source: Provenance | None,
) -> MSA:
    """Build an :class:`MSA` for one domain from its aligned row.

    Parameters
    ----------
    aligned_seq : str
        The gapped aligned row for this domain.
    canonical_seq : str
        The UniProt canonical sequence of the kinase.
    start, end : int
        The header's 1-based UniProt start/end.
    source : Provenance | None
        Alignment provenance.

    Returns
    -------
    MSA
        The populated MSA model (region2uniprot gaps default to None via the model validator).
    """
    col2uniprot, kd_start, kd_end, reconciled = _col2uniprot(
        aligned_seq, canonical_seq, start, end
    )
    region2uniprot = {
        label: col2uniprot[col]
        for col, label in DICT_MSA_COL2LABEL.items()
        if col in col2uniprot
    }
    return MSA(
        regions=_build_regions(aligned_seq),
        region2uniprot=region2uniprot,
        start=kd_start,
        end=kd_end,
        reconciled=reconciled,
        source=source,
    )


def enrich_with_msa(
    obj, aligned_seq: str, start: int, end: int, source: Provenance
) -> None:
    """Attach an :class:`MSA` to a kinase, creating an MSA-only KinCore shell if needed.

    Parameters
    ----------
    obj : KinaseInfo
        The kinase to annotate (mutated in place).
    aligned_seq : str
        The gapped aligned row for this domain.
    start, end : int
        The header's 1-based UniProt start/end.
    source : Provenance
        Alignment provenance.

    Returns
    -------
    None
    """
    msa_model = build_msa_model(
        aligned_seq, obj.uniprot.canonical_seq, start, end, source
    )
    if obj.kincore is None:
        obj.kincore = KinCore(msa=msa_model)
    else:
        obj.kincore.msa = msa_model


def _base_accession(uniprot_id: str) -> str:
    """Strip the multi-KD ``_1``/``_2`` domain suffix to the bare UniProt accession."""
    return re.sub(r"_\d+$", "", uniprot_id)


def _match_target(entry: dict, by_hgnc: dict, by_base: dict) -> object | None:
    """Resolve an MSA entry to a KinaseInfo: by ``hgnc_name``, else by UniProt accession.

    The accession fallback absorbs gene-symbol synonyms (e.g. ICK->CILK1, PRPF4B->PRP4K)
    without a hardcoded alias map; when an accession maps to several domains (a single MSA
    entry for a multi-KD protein), the domain whose adjudicated KD bounds best overlap the
    MSA range is chosen.

    Parameters
    ----------
    entry : dict
        A parsed MSA entry (``hgnc``/``uniprot``/``start``/``end``/``seq``).
    by_hgnc : dict
        Targets keyed by ``hgnc_name``.
    by_base : dict
        Targets grouped by bare UniProt accession.

    Returns
    -------
    KinaseInfo | None
        The matched kinase, or None if unresolved.
    """
    obj = by_hgnc.get(entry["hgnc"])
    if obj is not None:
        return obj
    candidates = by_base.get(entry["uniprot"], [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        lo, hi = entry["start"], entry["end"]

        def _overlap(cand):
            start, end = cand.adjudicate_kd_start(), cand.adjudicate_kd_end()
            if start is None or end is None:
                return -1
            return min(hi, end) - max(lo, start)

        return max(candidates, key=_overlap)
    return None


def enrich_kinases_with_msa(dict_targets: dict) -> None:
    """Annotate a dict of kinases with the Dunbrack MSA.

    Matches each MSA domain to a target by ``hgnc_name`` first, then by UniProt accession (see
    :func:`_match_target`), so gene-symbol synonyms and single-entry multi-KD proteins resolve
    without a hardcoded alias map.

    Parameters
    ----------
    dict_targets : dict[str, KinaseInfo]
        Kinases to annotate, keyed by ``hgnc_name`` (mutated in place).

    Returns
    -------
    None
    """
    path = MSA_SOURCE.resolve()
    entries = parse_msa(path)
    source = MSA_SOURCE.provenance(path)
    by_base: dict[str, list] = defaultdict(list)
    for obj in dict_targets.values():
        by_base[_base_accession(obj.uniprot_id)].append(obj)

    n_ok = 0
    for entry in entries:
        obj = _match_target(entry, dict_targets, by_base)
        if obj is None:
            logger.warning(
                f"MSA entry {entry['hgnc']} ({entry['uniprot']}) unmatched; skipping."
            )
            continue
        try:
            enrich_with_msa(obj, entry["seq"], entry["start"], entry["end"], source)
            n_ok += 1
        except Exception as e:
            logger.error(
                f"MSA enrichment failed for {entry['hgnc']}: {e}", exc_info=True
            )
    logger.info(f"annotated {n_ok} kinase(s) with the Dunbrack MSA.")
