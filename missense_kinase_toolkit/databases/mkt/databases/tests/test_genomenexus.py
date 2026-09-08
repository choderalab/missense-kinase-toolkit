import pytest
from mkt.databases.genomenexus import build_exon_map, get_canonical_transcripts


def _record():
    """Return a synthetic 3-residue transcript record (CDS = 3*3 + 3 stop = 12 nt).

    Exon 1 (rank 1) is 15 bp with an 8 bp 5' UTR -> 7 coding bp (CDS 1-7); exon 2 (rank 2) is
    5 bp all coding (CDS 8-12). Residue 3's codon (CDS 7-9) straddles the boundary and is
    assigned by its central base (CDS 8) to exon 2.
    """
    return {
        "transcriptId": "ENST00000000001",
        "proteinLength": 3,
        "exons": [
            {"exonStart": 100, "exonEnd": 114, "rank": 1, "strand": 1},
            {"exonStart": 200, "exonEnd": 204, "rank": 2, "strand": 1},
        ],
        "utrs": [{"type": "five_prime_UTR", "start": 100, "end": 107, "strand": 1}],
    }


class TestBuildExonMap:
    def test_map_and_split_codon(self):
        assert build_exon_map(_record(), 3) == {1: 1, 2: 1, 3: 2}

    def test_strand_agnostic(self):
        # rank is coding order and UTR overlap is genomic, so a minus-strand record with the
        # same rank order yields the same map
        rec = _record()
        for exon in rec["exons"]:
            exon["strand"] = -1
        assert build_exon_map(rec, 3) == {1: 1, 2: 1, 3: 2}

    def test_inconsistent_length_returns_none(self):
        assert build_exon_map(_record(), 4) is None

    def test_no_exons_returns_none(self):
        assert build_exon_map({"exons": [], "utrs": []}, 3) is None


@pytest.mark.network
class TestGenomeNexusExons:
    def test_egfr_landmark_exons(self):
        # canonical EGFR clinical exon numbering: exon-19 deletion (~746), T790M (exon 20),
        # L858R (exon 21)
        rec = get_canonical_transcripts(["EGFR"], build="GRCh37")["EGFR"]
        idx2exon = build_exon_map(rec, rec["proteinLength"])
        assert idx2exon[746] == 19
        assert idx2exon[790] == 20
        assert idx2exon[858] == 21
        assert len(idx2exon) == rec["proteinLength"]

    def test_braf_v600_exon15(self):
        # BRAF is on the minus strand; V600E is exon 15
        rec = get_canonical_transcripts(["BRAF"], build="GRCh37")["BRAF"]
        idx2exon = build_exon_map(rec, rec["proteinLength"])
        assert idx2exon[600] == 15
