"""Unit tests for the Dunbrack-MSA enrichment (parsing, matching, mapping)."""

from types import SimpleNamespace

from mkt.databases import msa


def test_parse_header_single_and_multidomain():
    """Header parsing yields hgnc/uniprot/range; multi-KD domains keep their suffix."""
    single = msa._parse_header("AGC_AKT1/150-408 AKT1_HUMAN AKT1 P31749")
    assert single == {"hgnc": "AKT1", "uniprot": "P31749", "start": 150, "end": 408}

    multi = msa._parse_header("TYR_JAK1_2/875-1151 JAK1_HUMAN JAK1 P23458")
    assert multi["hgnc"] == "JAK1_2" and multi["uniprot"] == "P23458"


def test_base_accession_strips_domain_suffix():
    assert msa._base_accession("P23458_2") == "P23458"
    assert msa._base_accession("P31749") == "P31749"


def _obj(uniprot_id, kd=(None, None)):
    """Minimal KinaseInfo stand-in exposing uniprot_id + adjudicated KD bounds."""
    return SimpleNamespace(
        uniprot_id=uniprot_id,
        adjudicate_kd_start=lambda: kd[0],
        adjudicate_kd_end=lambda: kd[1],
    )


def test_match_target_by_hgnc_name():
    obj = _obj("P31749")
    entry = {"hgnc": "AKT1", "uniprot": "P31749", "start": 150, "end": 408}
    assert msa._match_target(entry, {"AKT1": obj}, {}) is obj


def test_match_target_accession_fallback_synonym():
    """A gene-symbol synonym (MSA hgnc not a dict key) resolves by UniProt accession."""
    obj = _obj("Q9UPZ9")  # CILK1 (MSA labels it ICK)
    by_hgnc = {"CILK1": obj}
    by_base = {"Q9UPZ9": [obj]}
    entry = {"hgnc": "ICK", "uniprot": "Q9UPZ9", "start": 4, "end": 284}
    assert msa._match_target(entry, by_hgnc, by_base) is obj


def test_match_target_multidomain_by_range_overlap():
    """A single MSA entry for a multi-KD protein picks the domain overlapping its range."""
    dom1 = _obj("Q8IWB6_1", kd=(227, 512))
    dom2 = _obj("Q8IWB6_2", kd=(600, 850))
    by_base = {"Q8IWB6": [dom2, dom1]}  # order should not matter
    entry = {"hgnc": "TEX14", "uniprot": "Q8IWB6", "start": 227, "end": 512}
    assert msa._match_target(entry, {}, by_base) is dom1


def test_build_regions_orders_aligned_and_unaligned():
    """Region slicing yields the 17 aligned blocks + 16 unaligned regions, in order."""
    aligned = "-" * 2218
    regions = msa._build_regions(aligned)
    assert "B1N" in regions and "HI" in regions
    assert "B1N~B1C" in regions  # unaligned region between the first two blocks
    assert len(regions) == 33  # 17 aligned + 16 unaligned


def test_col2uniprot_direct_mapping():
    """A row matching the canonical slice maps columns to header UniProt numbering (no reconcile)."""
    canonical = "M" * 20 + "ACDEFG"  # residues 21-26 are the aligned stretch
    aligned = "AC-DEFG"  # cols 1-2,4-7 carry ACDEFG; col 3 is a gap
    col2uniprot, start, end, reconciled = msa._col2uniprot(aligned, canonical, 21, 26)
    assert not reconciled and start == 21 and end == 26
    assert col2uniprot == {1: 21, 2: 22, 4: 23, 5: 24, 6: 25, 7: 26}
