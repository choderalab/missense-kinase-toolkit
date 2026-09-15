"""Tests for canonical-coordinate reconciliation (:mod:`mkt.databases.isoform`), offline.

Synthetic isoforms of human ubiquitin stand in for real transcripts, and the network
fetchers are replaced by in-memory fakes that count their calls.
"""

import logging

from mkt.databases.isoform import (
    CanonicalReconciler,
    SourceTier,
    clean_refseq_accession,
    map_positions_by_alignment,
    select_domain_name,
)

CANONICAL = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
"""str: Canonical sequence (human ubiquitin, 76 aa)."""

ISOFORM_EXTENDED = "GSAHP" + CANONICAL
"""str: Isoform with a 5-residue N-terminal extension (canonical i -> isoform i + 5)."""

ISOFORM_DELETED = CANONICAL[:29] + CANONICAL[39:]
"""str: Isoform lacking canonical residues 30-39."""

ISOFORM_INSERTED = CANONICAL[:20] + "WWWWW" + CANONICAL[20:]
"""str: Isoform with 5 isoform-only residues after canonical residue 20."""


def _shifted_position(idx_canonical: int, int_shift: int) -> int:
    """Return an isoform position whose direct canonical residue differs from the target."""
    idx_isoform = idx_canonical + int_shift
    assert CANONICAL[idx_isoform - 1] != CANONICAL[idx_canonical - 1]
    return idx_isoform


class _Fakes:
    """In-memory stand-ins for the network fetchers, counting their calls."""

    def __init__(self, transcripts=None, ensembl=None, refseq=None, annotations=None):
        self.transcripts = transcripts or {}
        self.ensembl = ensembl or {}
        self.refseq = refseq or {}
        self.annotations = annotations or {}
        self.calls = {"transcripts": 0, "ensembl": 0, "refseq": 0, "annotations": []}

    def fetch_transcripts(self, genes, build, isoform_override):
        self.calls["transcripts"] += 1
        return {
            g: {"transcriptId": self.transcripts[g]}
            for g in genes
            if g in self.transcripts
        }

    def fetch_ensembl_protein(self, transcript_id, build):
        self.calls["ensembl"] += 1
        return self.ensembl.get(transcript_id)

    def fetch_refseq_protein(self, accession):
        self.calls["refseq"] += 1
        return self.refseq.get(accession)

    def fetch_annotations(self, variants, build, isoform_override):
        self.calls["annotations"].append(list(variants))
        return {v: self.annotations[v] for v in variants if v in self.annotations}

    def reconciler(self, **kwargs):
        return CanonicalReconciler(
            dict_canonical={"KIN": CANONICAL},
            fetch_transcripts=self.fetch_transcripts,
            fetch_ensembl_protein=self.fetch_ensembl_protein,
            fetch_refseq_protein=self.fetch_refseq_protein,
            fetch_annotations=self.fetch_annotations,
            **kwargs,
        )


class TestMapPositionsByAlignment:
    def test_identity(self):
        expected = {i: i for i in range(1, len(CANONICAL) + 1)}
        assert map_positions_by_alignment(CANONICAL, CANONICAL) == expected

    def test_n_terminal_extension_shifts(self):
        dict_map = map_positions_by_alignment(ISOFORM_EXTENDED, CANONICAL)
        assert dict_map[6] == 1
        assert dict_map[40 + 5] == 40
        assert all(i not in dict_map for i in range(1, 6))

    def test_internal_deletion_gives_two_blocks(self):
        dict_map = map_positions_by_alignment(ISOFORM_DELETED, CANONICAL)
        assert dict_map[29] == 29
        assert dict_map[30] == 40

    def test_isoform_only_residue_is_absent(self):
        dict_map = map_positions_by_alignment(ISOFORM_INSERTED, CANONICAL)
        assert all(i not in dict_map for i in range(21, 26))
        assert dict_map[26] == 21


class TestCanonicalReconciler:
    def test_direct_hit_does_no_alignment(self):
        fakes = _Fakes(
            transcripts={"KIN": "ENST_OVERRIDE"},
            ensembl={"ENST_OVERRIDE": ISOFORM_EXTENDED},
        )
        rec = fakes.reconciler()
        assert rec.reconcile("KIN", 10, CANONICAL[9]) == (10, SourceTier.direct)
        assert rec._dict_map == {}
        assert fakes.calls["ensembl"] == 0

    def test_override_shift_resolves_as_mskcc(self):
        fakes = _Fakes(
            transcripts={"KIN": "ENST_OVERRIDE"},
            ensembl={"ENST_OVERRIDE": ISOFORM_EXTENDED},
        )
        idx_isoform = _shifted_position(30, 5)
        assert fakes.reconciler().reconcile("KIN", idx_isoform, CANONICAL[29]) == (
            30,
            SourceTier.mskcc,
        )

    def test_refseq_rescues_an_override_miss(self):
        """No override transcript (the TGFBR2 case), but the row's RefSeq mRNA resolves it."""
        fakes = _Fakes(refseq={"NM_000001.2": ISOFORM_INSERTED})
        idx_isoform = _shifted_position(40, 5)
        result = fakes.reconciler().reconcile(
            "KIN", idx_isoform, CANONICAL[39], refseq='"NM_000001.2"'
        )
        assert result == (40, SourceTier.refseq)

    def test_isoform_only_position_returns_none(self):
        """An isoform-only residue has no canonical equivalent (the FGFR1 2-33 case)."""
        fakes = _Fakes(refseq={"NM_000001": ISOFORM_INSERTED})
        assert fakes.reconciler().reconcile("KIN", 23, "W", refseq="NM_000001") == (
            None,
            None,
        )

    def test_mismatched_reference_residue_is_rejected(self):
        """A mapped candidate whose canonical residue disagrees with the reported one is rejected."""
        fakes = _Fakes(
            transcripts={"KIN": "ENST_OVERRIDE"},
            ensembl={"ENST_OVERRIDE": ISOFORM_EXTENDED},
        )
        wrong_aa = "W" if CANONICAL[29] != "W" else "C"
        assert fakes.reconciler().reconcile("KIN", 35, wrong_aa) == (None, None)

    def test_one_alignment_per_accession(self):
        fakes = _Fakes(
            transcripts={"KIN": "ENST_OVERRIDE"},
            ensembl={"ENST_OVERRIDE": ISOFORM_EXTENDED},
        )
        rec = fakes.reconciler()
        list_canonical = [
            i for i in range(10, 70) if CANONICAL[i + 4] != CANONICAL[i - 1]
        ]
        list_idx, list_source = rec.reconcile_many(
            ["KIN"] * len(list_canonical),
            [i + 5 for i in list_canonical],
            [CANONICAL[i - 1] for i in list_canonical],
        )
        assert list_idx == list_canonical
        assert set(list_source) == {SourceTier.mskcc}
        assert len(rec._dict_map) == 1
        assert fakes.calls == {
            "transcripts": 1,
            "ensembl": 1,
            "refseq": 0,
            "annotations": [],
        }

    def test_internal_difference_requires_alignment(self):
        """A source differing internally from the canonical (the PIK3C2G case) still aligns."""
        fakes = _Fakes(
            transcripts={"KIN": "ENST_OVERRIDE"},
            ensembl={"ENST_OVERRIDE": ISOFORM_DELETED},
        )
        idx_isoform = 45 - 10
        assert CANONICAL[idx_isoform - 1] != CANONICAL[44]
        assert fakes.reconciler().reconcile("KIN", idx_isoform, CANONICAL[44]) == (
            45,
            SourceTier.mskcc,
        )

    def test_genomenexus_tier_batches_the_residual(self):
        fakes = _Fakes(
            ensembl={"ENST_GN": ISOFORM_EXTENDED},
            annotations={
                "1:g.100A>T": {
                    "transcriptId": "ENST_GN",
                    "proteinPosition": {"start": 35},
                }
            },
        )
        rec = fakes.reconciler()
        assert CANONICAL[34] != CANONICAL[29]
        list_idx, list_source = rec.reconcile_many(
            ["KIN", "KIN", "KIN"],
            [10, 35, 35],
            [CANONICAL[9], CANONICAL[29], CANONICAL[29]],
            list_hgvsg=[None, "1:g.100A>T", "1:g.100A>T"],
        )
        assert list_idx == [10, 30, 30]
        assert list_source == [
            SourceTier.direct,
            SourceTier.genomenexus,
            SourceTier.genomenexus,
        ]
        assert fakes.calls["annotations"] == [["1:g.100A>T"]]

    def test_unknown_gene_is_skipped(self):
        assert _Fakes().reconciler().reconcile("NOTAKINASE", 10, "K") == (None, None)


def test_clean_refseq_accession():
    assert clean_refseq_accession('"NM_003242.6,NM_001024847.3"') == "NM_003242.6"
    assert clean_refseq_accession("XM_011533856.1") is None
    assert clean_refseq_accession(float("nan")) is None
    assert clean_refseq_accession(None) is None


class TestSelectDomainName:
    DICT_SPAN = {
        "JAK1_1": (867, 1154),
        "JAK1_2": (583, 847),
        "BRAF": (457, 717),
        "TEX14_1": (227, 512),
        "TEX14_2": (None, None),
        "A_1": (1, 100),
        "A_2": (40, 60),
    }

    def test_inside_each_domain(self):
        assert select_domain_name(
            "JAK1", ["JAK1_1", "JAK1_2"], self.DICT_SPAN, 900
        ) == ("JAK1_1", True)
        assert select_domain_name(
            "JAK1", ["JAK1_1", "JAK1_2"], self.DICT_SPAN, 617
        ) == ("JAK1_2", True)

    def test_between_domains_keeps_bare_symbol(self):
        assert select_domain_name(
            "JAK1", ["JAK1_1", "JAK1_2"], self.DICT_SPAN, 855
        ) == ("JAK1", False)

    def test_single_domain_gene(self):
        assert select_domain_name("BRAF", ["BRAF"], self.DICT_SPAN, 600) == (
            "BRAF",
            True,
        )
        assert select_domain_name("BRAF", ["BRAF"], self.DICT_SPAN, 10) == (
            "BRAF",
            False,
        )

    def test_none_span_is_skipped(self):
        assert select_domain_name(
            "TEX14", ["TEX14_1", "TEX14_2"], self.DICT_SPAN, 300
        ) == ("TEX14_1", True)

    def test_overlap_picks_smallest_span_with_warning(self, caplog):
        caplog.set_level(logging.WARNING)
        assert select_domain_name("A", ["A_1", "A_2"], self.DICT_SPAN, 50) == (
            "A_2",
            True,
        )
        assert "overlapping domains" in caplog.text

    def test_missing_position(self):
        assert select_domain_name(
            "JAK1", ["JAK1_1", "JAK1_2"], self.DICT_SPAN, None
        ) == ("JAK1", None)
