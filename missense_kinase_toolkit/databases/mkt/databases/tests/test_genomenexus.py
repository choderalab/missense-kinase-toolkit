import pytest
from mkt.databases.genomenexus import (
    _match_exonless_transcript,
    annotate_genomic_locations,
    build_exon_map,
    get_canonical_transcripts,
)


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


def _canonical(**kwargs):
    """Return an exon-less canonical record (a patch/alternate-contig copy)."""
    return {
        "transcriptId": "ENST_PATCH",
        "ccdsId": "CCDS1",
        "refseqMrnaId": "NM_1",
        "uniprotId": "P1",
        **kwargs,
    }


def _candidate(transcript_id, exons=True, **kwargs):
    """Return a candidate transcript record, with exons unless ``exons`` is False."""
    return {
        "transcriptId": transcript_id,
        "exons": _record()["exons"] if exons else [],
        **kwargs,
    }


class TestMatchExonlessTranscript:
    def test_same_ccds_with_exons(self):
        candidates = [
            _candidate("ENST_PATCH", exons=False, ccdsId="CCDS1", uniprotId="P1"),
            _candidate("ENST_OTHER", ccdsId="CCDS2", uniprotId="P1"),
            _candidate("ENST_PRIMARY", ccdsId="CCDS1", uniprotId="P1"),
        ]
        match = _match_exonless_transcript(_canonical(), candidates)
        assert match["transcriptId"] == "ENST_PRIMARY"

    def test_refseq_when_no_ccds(self):
        candidates = [_candidate("ENST_PRIMARY", refseqMrnaId="NM_1")]
        match = _match_exonless_transcript(_canonical(ccdsId=None), candidates)
        assert match["transcriptId"] == "ENST_PRIMARY"

    def test_uniprot_mismatch_rejected(self):
        candidates = [_candidate("ENST_PRIMARY", ccdsId="CCDS1", uniprotId="P2")]
        assert _match_exonless_transcript(_canonical(), candidates) is None

    def test_missing_uniprot_not_compared(self):
        candidates = [_candidate("ENST_PRIMARY", ccdsId="CCDS1", uniprotId=None)]
        match = _match_exonless_transcript(_canonical(uniprotId=None), candidates)
        assert match["transcriptId"] == "ENST_PRIMARY"

    def test_no_match_returns_none(self):
        candidates = [_candidate("ENST_OTHER", ccdsId="CCDS2", refseqMrnaId="NM_2")]
        assert _match_exonless_transcript(_canonical(), candidates) is None


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

    def test_ikbke_grch37_patch_canonical_resolved(self):
        # the GRCh37 canonical ENST00000581977 sits on HG1293_PATCH without exons;
        # the chr1 copy shares CCDS30996
        rec = get_canonical_transcripts(["IKBKE"], build="GRCh37")["IKBKE"]
        assert rec["transcriptId"] == "ENST00000367120"
        assert rec["ccdsId"] == "CCDS30996"
        assert len(build_exon_map(rec, rec["proteinLength"])) == 716

    def test_pip4k2b_grch38_alt_canonical_resolved(self):
        # the GRCh38 canonical ENST00000613180 sits on HSCHR17_7_CTG4 without exons;
        # the chr17 copy shares CCDS11329
        rec = get_canonical_transcripts(["PIP4K2B"], build="GRCh38")["PIP4K2B"]
        assert rec["transcriptId"] == "ENST00000619039"
        assert rec["ccdsId"] == "CCDS11329"
        assert len(build_exon_map(rec, rec["proteinLength"])) == 416


@pytest.mark.network
class TestAnnotateGenomicLocations:
    def test_snv_and_indel_keyed_by_location(self):
        # RET C634R (SNV) and E632_L633del (in-frame deletion) at their GRCh37 loci,
        # reported in cBioPortal's (MSKCC) frame and keyed by the input location
        list_loc = ["10,43609948,43609948,T,C", "10,43609942,43609947,GAGCTG,-"]
        dict_out = annotate_genomic_locations(
            list_loc, build="GRCh37", isoform_override="mskcc"
        )
        assert dict_out[list_loc[0]]["hugoGeneSymbol"] == "RET"
        assert dict_out[list_loc[0]]["hgvspShort"] == "p.C634R"
        assert dict_out[list_loc[1]]["hgvspShort"] == "p.E632_L633del"

    def test_wrong_build_does_not_annotate(self):
        # the same GRCh37 coordinates on GRCh38 miss the RET reference base
        dict_out = annotate_genomic_locations(
            ["10,43609948,43609948,T,C"], build="GRCh38", isoform_override="mskcc"
        )
        assert "10,43609948,43609948,T,C" not in dict_out
