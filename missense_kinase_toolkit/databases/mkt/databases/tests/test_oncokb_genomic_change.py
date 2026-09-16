"""Tests for annotating a mutation by genomic change.

Covers :class:`mkt.databases.oncokb.OncoKBGenomicChange` (URL construction and
response parsing, exercised offline against a stubbed response) and
:func:`mkt.databases.cbioportal.return_genomic_location_list`, which builds the
``genomicLocation`` strings the endpoint takes.

Genomic coordinates are the only frame-independent way to annotate a cohort:
OncoKB annotates FGFR1 on the MSKCC-override isoform (N577K) but TGFBR2 on the
UniProt canonical (R528H), so a ``proteinChange`` taken from one study transcript
asks about the wrong residue for some genes.
"""

import pandas as pd
import pytest
from mkt.databases import cbioportal
from mkt.databases.oncokb import OncoKBGenomicChange


def _annotated_json(hugo="FGFR1", alteration="N577K"):
    """Return a minimal byGenomicChange response for a reviewed mutation."""
    return {
        "query": {"hugoSymbol": hugo, "alteration": alteration},
        "geneExist": True,
        "variantSummary": f"{alteration} is likely oncogenic.",
        "oncogenic": "Likely Oncogenic",
        "vus": False,
        "mutationEffect": {"knownEffect": "Likely Gain-of-function"},
        "highestSensitiveLevel": "LEVEL_4",
        "highestResistanceLevel": None,
        "highestDiagnosticImplicationLevel": None,
        "highestPrognosticImplicationLevel": None,
        "highestFdaLevel": None,
        "treatments": [{"drugs": [{"drugName": "Erdafitinib"}]}],
    }


@pytest.fixture
def stub_query(monkeypatch):
    """Serve a canned response offline, leaving the real parsing in place.

    Only the two boundary methods are replaced -- ``set_api_key`` (so no token is
    needed) and ``query_api`` (the network hop) -- so ``__post_init__`` still runs
    the response handling these tests exercise.
    """

    def _install(dict_json):
        monkeypatch.setattr(OncoKBGenomicChange, "set_api_key", lambda self: {})

        def _fake_query_api(self):
            self._json = dict_json

        monkeypatch.setattr(OncoKBGenomicChange, "query_api", _fake_query_api)

    return _install


class TestGenomicLocationStrings:
    def _frame(self):
        return pd.DataFrame(
            {
                "chr": ["8", "3", "7"],
                "startPosition": [38274849, 30732970, 140453136],
                # an in-frame deletion spans more than one base
                "endPosition": [38274849, 30732972, 140453136],
                "referenceAllele": ["G", "GAA", "A"],
                "variantAllele": ["C", "-", "T"],
                "ncbiBuild": ["GRCh37", "GRCh37", "GRCh38"],
            }
        )

    def test_builds_chrom_start_end_ref_alt(self):
        list_out = cbioportal.return_genomic_location_list(self._frame(), "GRCh37")

        assert list_out[0] == "8,38274849,38274849,G,C"

    def test_covers_indels_that_hgvsg_skips(self):
        # return_hgvsg_list is single-base only; the deletion is None there
        df = self._frame()

        list_location = cbioportal.return_genomic_location_list(df, "GRCh37")
        list_hgvsg = cbioportal.return_hgvsg_list(df, "GRCh37")

        assert list_location[1] == "3,30732970,30732972,GAA,-"
        assert list_hgvsg[1] is None

    def test_other_build_rows_are_none(self):
        list_out = cbioportal.return_genomic_location_list(self._frame(), "GRCh37")

        assert list_out[2] is None

    def test_build_aliases_normalize(self):
        df = self._frame()
        df["ncbiBuild"] = ["37", "hg19", "GRCh38"]

        list_out = cbioportal.return_genomic_location_list(df, "GRCh37")

        assert list_out[0] is not None and list_out[1] is not None

    def test_missing_columns_return_none(self):
        # endPosition is absent from a frame built for the HGVSg path
        df = self._frame().drop(columns=["endPosition"])

        assert cbioportal.return_genomic_location_list(df, "GRCh37") == [None] * 3


class TestOncoKBGenomicChange:
    def test_url_carries_location_and_build(self, stub_query):
        stub_query(_annotated_json())

        obj = OncoKBGenomicChange(genomic_location="8,38274849,38274849,G,C")

        assert "annotate/mutations/byGenomicChange" in obj.url_query
        assert "genomicLocation=8,38274849,38274849,G,C" in obj.url_query
        assert "referenceGenome=GRCh37" in obj.url_query
        assert "tumorType" not in obj.url_query

    def test_tumor_type_is_appended_when_given(self, stub_query):
        stub_query(_annotated_json())

        obj = OncoKBGenomicChange(
            genomic_location="8,38274849,38274849,G,C", tumor_type="BLCA"
        )

        assert obj.url_query.endswith("&tumorType=BLCA")

    def test_records_the_alteration_oncokb_resolved(self, stub_query):
        # the caller cannot know OncoKB's frame otherwise: these coordinates are
        # FGFR1 N546K on the UniProt canonical but N577K in OncoKB's transcript
        stub_query(_annotated_json())

        obj = OncoKBGenomicChange(genomic_location="8,38274849,38274849,G,C")

        assert obj.gene_name == "FGFR1"
        assert obj.alteration == "N577K"

    def test_parses_the_shared_annotation_fields(self, stub_query):
        stub_query(_annotated_json())

        obj = OncoKBGenomicChange(genomic_location="8,38274849,38274849,G,C")

        assert obj.oncogenic == "Likely Oncogenic"
        assert obj.known_effect == "Likely Gain-of-function"
        assert obj.vus is False
        assert obj.dict_highest_level["Sensitive"] == 4
        assert obj.list_treatment == [["Erdafitinib"]]

    def test_unreviewed_variant_leaves_annotations_unset(self, stub_query):
        dict_json = _annotated_json()
        dict_json["variantSummary"] = (
            "The mutation has not specifically been reviewed by the OncoKB team."
        )
        stub_query(dict_json)

        obj = OncoKBGenomicChange(
            genomic_location="8,38274849,38274849,G,C", verbose=False
        )

        # the query echo still resolves, so the caller learns what it hit
        assert obj.gene_name == "FGFR1"
        assert obj.oncogenic is None
        assert obj.dict_highest_level["Sensitive"] is None

    def test_missing_location_is_rejected(self, caplog):
        obj = OncoKBGenomicChange()

        assert obj.url_query is None
        assert obj._json is None
