"""Client for retrieving AlphaFold structure predictions.

Provides :class:`AlphaFoldPrediction` and :class:`AlphaFoldStructure`, REST clients
that fetch AlphaFold model metadata and downloadable structure files for a given
UniProt accession.
"""

import ast
import io
import logging

from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import MMCIFIO, MMCIFParser, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from mkt.databases import requests_wrapper
from mkt.databases.api_schema import RESTAPIClient
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)

MAX_AF_MISMATCH_FRACTION = 0.10
"""float: reject an AlphaFold KD slice differing from the canonical UniProt sequence at more
than this fraction of positions (a larger divergence indicates a wrong isoform/sequence rather
than a near-match to flag on ``mismatch``)."""


@dataclass
class AlphaFoldPrediction(RESTAPIClient):
    """Class to query the AlphaFold EBI prediction API for a given UniProt accession.

    Parameters:
    -----------
    uniprot_id : str
        UniProt accession (e.g. "P00533").
    """

    uniprot_id: str
    """UniProt accession to query."""
    url: str = "https://alphafold.ebi.ac.uk/api"
    """Base URL for the AlphaFold EBI API."""
    headers: str = "{'Accept': 'application/json'}"
    """Header for the API request."""
    _json: dict | None = None
    """JSON response from the AlphaFold API."""

    def __post_init__(self):
        self.query_api()

    def query_api(self) -> None:
        """Query the AlphaFold prediction endpoint and populate _json.

        Returns:
        --------
        None
        """
        url = f"{self.url}/prediction/{self.uniprot_id}"
        res = requests_wrapper.get_cached_session().get(
            url,
            headers=ast.literal_eval(self.headers),
        )
        self._stamp_from_response(res)

        if res.ok:
            results = res.json()
            # the canonical AF2 model is entry "AF-<accession>-F1"; the AF DB may also return
            # a newer numeric-id (AF3) model under the same accession and/or isoform models --
            # select the AF2 fragment-1 entry specifically rather than requiring a single result
            f1 = [r for r in results if r.get("entryId") == f"AF-{self.uniprot_id}-F1"]
            if f1:
                self._json = f1[0]
            else:
                logger.warning(
                    "no canonical AF-<acc>-F1 AlphaFold model for %s (%d result(s))",
                    self.uniprot_id,
                    len(results),
                )
                self._json = None
        else:
            logger.error(
                "AlphaFold API error for %s: %s", self.uniprot_id, res.status_code
            )
            self._json = None


@dataclass
class AlphaFoldStructure(AlphaFoldPrediction):
    """Class to download the mmCIF file from the AlphaFold EBI server.

    Inherits from AlphaFoldPrediction and uses the cifUrl from the prediction
    JSON to download and store the CIF content.

    Parameters:
    -----------
    uniprot_id : str
        UniProt accession (e.g. "P00533").
    """

    _cif: str | None = None
    """mmCIF file content downloaded from AlphaFold."""

    def __post_init__(self):
        super().__post_init__()
        self._download_cif()

    def _download_cif(self) -> None:
        """Download the mmCIF file from the cifUrl in the prediction JSON.

        Returns:
        --------
        None
        """
        if self._json is None:
            logger.error("No prediction JSON available for %s", self.uniprot_id)
            return

        cif_url = self._json.get("cifUrl")
        if cif_url is None:
            logger.error("No cifUrl in prediction JSON for %s", self.uniprot_id)
            return

        res = requests_wrapper.get_cached_session().get(cif_url)
        self._stamp_from_response(res)
        if res.ok:
            self._cif = res.text
        else:
            logger.error(
                "Failed to download CIF for %s: %s", self.uniprot_id, res.status_code
            )
            self._cif = None


def slice_alphafold_cif_to_kd(
    cif_text: str,
    start: int,
    end: int,
) -> dict[str, str | list[str]]:
    """Slice a full-length AlphaFold mmCIF to the kinase-domain residue range.

    Keeps residues whose UniProt (auth) number is within ``[start, end]`` and returns the
    mmCIF dict, injecting the one-letter KD sequence under
    ``_entity_poly.pdbx_seq_one_letter_code`` (which ``MMCIFIO`` does not emit) so the
    CIF-sequence accessors work as they do for KinCoRe CIFs.

    Parameters
    ----------
    cif_text : str
        Full-length AlphaFold mmCIF content.
    start : int
        Inclusive kinase-domain start in UniProt numbering (from adjudication).
    end : int
        Inclusive kinase-domain end in UniProt numbering (from adjudication).

    Returns
    -------
    dict[str, str | list[str]]
        The KD-sliced mmCIF dictionary.
    """
    structure = MMCIFParser(QUIET=True).get_structure("af", io.StringIO(cif_text))

    class _KinaseDomainSelect(Select):
        def accept_residue(self, residue):
            return start <= residue.id[1] <= end

    mmcif_io = MMCIFIO()
    mmcif_io.set_structure(structure)
    buffer = io.StringIO()
    mmcif_io.save(buffer, _KinaseDomainSelect())
    buffer.seek(0)
    dict_cif = MMCIF2Dict(buffer)

    # MMCIFIO drops the one-letter sequence; rebuild it from the KD residues in order
    seq = "".join(
        protein_letters_3to1.get(residue.resname, "X")
        for residue in structure[0].get_residues()
        if start <= residue.id[1] <= end
    )
    dict_cif["_entity_poly.pdbx_seq_one_letter_code"] = [seq]
    return dict_cif


def fetch_alphafold_kd(
    uniprot_id: str,
    start: int,
    end: int,
    canonical_seq: str | None = None,
    max_mismatch_fraction: float = MAX_AF_MISMATCH_FRACTION,
):
    """Fetch the AlphaFold structure for a UniProt ID and slice it to the kinase domain.

    Parameters
    ----------
    uniprot_id : str
        Base UniProt accession (no multi-domain suffix).
    start : int
        Inclusive kinase-domain start in UniProt numbering.
    end : int
        Inclusive kinase-domain end in UniProt numbering.
    canonical_seq : str | None, optional
        Canonical UniProt sequence to compare against; when provided, KD-slice positions that
        differ from ``canonical_seq[start - 1 : end]`` are recorded on the model's ``mismatch``
        field (the AF DB model is numbered in UniProt coordinates). By default None (no check).
    max_mismatch_fraction : float, optional
        Reject the structure (return None) when more than this fraction of KD-slice positions
        differ from canonical -- a larger divergence indicates a wrong isoform/sequence rather
        than a near-match to flag. By default :data:`MAX_AF_MISMATCH_FRACTION`.

    Returns
    -------
    AlphaFold | None
        The KD-sliced AlphaFold model, or None if the structure could not be retrieved, its
        slice length differs from canonical, or it exceeds ``max_mismatch_fraction``.
    """
    from mkt.schema.kinase_schema import AlphaFold, Provenance

    structure = AlphaFoldStructure(uniprot_id=uniprot_id)
    if structure._cif is None or structure._json is None:
        logger.warning("no AlphaFold structure for %s", uniprot_id)
        return None

    dict_cif = slice_alphafold_cif_to_kd(structure._cif, start, end)

    # compare the KD slice to the canonical UniProt sequence; record differing positions as a
    # mismatch (mirroring KinCoRe) rather than discarding the structure over a few substitutions
    mismatch = None
    if canonical_seq is not None:
        seq_kd = dict_cif["_entity_poly.pdbx_seq_one_letter_code"][0]
        seq_expected = canonical_seq[start - 1 : end]
        if len(seq_kd) != len(seq_expected):
            # a length difference means residues are missing/extra in the slice; the KD slice
            # can no longer be aligned to canonical by position, so reject it
            logger.error(
                "AlphaFold KD slice for %s has length %d != canonical %d over [%d, %d]; "
                "rejecting structure.",
                uniprot_id,
                len(seq_kd),
                len(seq_expected),
                start,
                end,
            )
            return None
        # 0-indexed positions within the KD slice that differ from canonical (KinCoRe convention)
        mismatch = [i for i, (a, b) in enumerate(zip(seq_kd, seq_expected)) if a != b]
        if len(mismatch) / len(seq_expected) > max_mismatch_fraction:
            logger.error(
                "AlphaFold KD sequence for %s differs from canonical at %d/%d position(s) "
                "(> %.1f%%) over [%d, %d]; rejecting structure.",
                uniprot_id,
                len(mismatch),
                len(seq_expected),
                max_mismatch_fraction * 100,
                start,
                end,
            )
            return None
        mismatch = mismatch or None
        if mismatch:
            logger.warning(
                "AlphaFold KD sequence for %s differs from canonical at %d position(s) over "
                "[%d, %d]; recording as mismatch.",
                uniprot_id,
                len(mismatch),
                start,
                end,
            )
    tool_used = None
    for line in structure._cif.splitlines():
        if line.strip().startswith("_ma_model_list.model_group_name"):
            tool_used = line.split(None, 1)[1].strip().strip('"')
            break

    json = structure._json
    latest_version = json.get("latestVersion")
    query_date = (
        structure.query_datetime.date().isoformat()
        if getattr(structure, "query_datetime", None) is not None
        else None
    )
    # AlphaFold DB provenance from the returned JSON (tracks updates: model version + pipeline)
    source = Provenance(
        name="AlphaFold DB",
        version=f"v{latest_version}" if latest_version is not None else None,
        citation=json.get("toolUsed"),
        query_date=query_date,
    )
    return AlphaFold(
        cif=dict_cif,
        start=start,
        end=end,
        entry_id=json.get("entryId"),
        uniprot_accession=json.get("uniprotAccession"),
        global_metric_value=json.get("globalMetricValue"),
        model_created_date=json.get("modelCreatedDate"),
        latest_version=latest_version,
        tool_used=tool_used,
        mismatch=mismatch,
        source=source,
    )


def enrich_with_alphafold(obj_kinase, force: bool = False) -> None:
    """Populate ``obj_kinase.alphafold`` with the KD-sliced AlphaFold structure.

    Fetches an AlphaFold DB structure for **every** kinase with adjudicated KD bounds (so a
    KinCoRe-CIF kinase also gets an AF2 counterpart for structure/SASA comparison); the KinCoRe
    CIF remains the preferred active-state model for rendering (see :func:`adjudicate_structure`).
    The kinase-domain bounds come from the object's adjudication, so ``kincore`` must already be
    in place.

    Parameters
    ----------
    obj_kinase : KinaseInfo
        The kinase object to enrich (mutated in place).
    force : bool, optional
        Re-fetch and re-slice even when the stored structure's KD bounds are unchanged
        (``--force-regen``), by default False.

    Returns
    -------
    None
    """
    start = obj_kinase.adjudicate_kd_start()
    end = obj_kinase.adjudicate_kd_end()
    if start is None or end is None:
        logger.warning(
            "no kinase-domain bounds for %s; skipping AlphaFold", obj_kinase.hgnc_name
        )
        return

    # idempotent: re-slice only when the KD bounds changed since the structure was stored
    # (e.g. an enrichment upstream, such as msa, updated adjudication); otherwise keep it
    # unless a forced regeneration is requested.
    if (
        not force
        and obj_kinase.alphafold is not None
        and obj_kinase.alphafold.start == start
        and obj_kinase.alphafold.end == end
    ):
        return

    obj_kinase.alphafold = fetch_alphafold_kd(
        obj_kinase.uniprot_id.split("_")[0],
        start,
        end,
        canonical_seq=obj_kinase.uniprot.canonical_seq,
    )


def get_alphafold(obj_kinase):
    """Return the AlphaFold structure for a kinase, fetching on the fly if not stored.

    Entries without a KinCoRe CIF carry a stored ``alphafold``; entries with a KinCoRe CIF
    do not, so this fetches + slices the AF structure on demand (e.g. for the force-AF
    render override or AF-based rSASA).

    Parameters
    ----------
    obj_kinase : KinaseInfo
        The kinase object.

    Returns
    -------
    AlphaFold | None
        The stored or freshly fetched KD-sliced AlphaFold model, or None if unavailable.
    """
    if obj_kinase.alphafold is not None:
        return obj_kinase.alphafold
    start = obj_kinase.adjudicate_kd_start()
    end = obj_kinase.adjudicate_kd_end()
    if start is None or end is None:
        return None
    return fetch_alphafold_kd(
        obj_kinase.uniprot_id.split("_")[0],
        start,
        end,
        canonical_seq=obj_kinase.uniprot.canonical_seq,
    )


def adjudicate_structure(obj_kinase, prefer_alphafold: bool = False):
    """Return the KD structure to render/compute over and a provenance label.

    The KinCoRe active-state CIF is preferred; the AlphaFold structure (stored on
    KinCoRe-less entries, or fetched on the fly via :func:`get_alphafold`) is the fallback,
    or is forced when ``prefer_alphafold`` is True. Used by the app, PyMOL output, and rSASA.

    Parameters
    ----------
    obj_kinase : KinaseInfo
        The kinase object.
    prefer_alphafold : bool, optional
        Force the AlphaFold structure even when a KinCoRe CIF is present, by default False.

    Returns
    -------
    tuple[dict | None, str | None]
        ``(mmCIF dict, source label)`` where the label is ``"KinCoRe Active State"`` or
        ``"AF2 Database"``; ``(None, None)`` when no structure is available.
    """
    dict_kincore = (
        obj_kinase.kincore.cif.cif
        if obj_kinase.kincore is not None and obj_kinase.kincore.cif is not None
        else None
    )
    if dict_kincore is not None and not prefer_alphafold:
        return dict_kincore, "KinCoRe Active State"

    obj_alphafold = get_alphafold(obj_kinase)
    if obj_alphafold is not None:
        return obj_alphafold.cif, "AF2 Database"
    if dict_kincore is not None:
        return dict_kincore, "KinCoRe Active State"
    return None, None
