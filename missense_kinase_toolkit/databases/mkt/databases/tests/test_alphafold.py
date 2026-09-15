import pytest
from mkt.databases.alphafold import (
    AlphaFoldPrediction,
    AlphaFoldStructure,
    fetch_alphafold_kd,
)

# ---------------------------------------------------------------------------
# module-scoped fixtures – one API call per query
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def af_prediction_abl1():
    """Fetch AlphaFold prediction for ABL1 (P00519) once."""
    return AlphaFoldPrediction(uniprot_id="P00519")


@pytest.fixture(scope="module")
def af_prediction_egfr():
    """Fetch AlphaFold prediction for EGFR (P00533) once."""
    return AlphaFoldPrediction(uniprot_id="P00533")


@pytest.fixture(scope="module")
def af_prediction_invalid():
    """Fetch AlphaFold prediction for invalid ID once."""
    return AlphaFoldPrediction(uniprot_id="INVALID_ID_12345")


@pytest.fixture(scope="module")
def af_structure_abl1():
    """Fetch AlphaFold structure for ABL1 (P00519) once."""
    return AlphaFoldStructure(uniprot_id="P00519")


@pytest.fixture(scope="module")
def af_structure_invalid():
    """Fetch AlphaFold structure for invalid ID once."""
    return AlphaFoldStructure(uniprot_id="INVALID_ID_12345")


@pytest.fixture(scope="module")
def abl1_kinase():
    """Load the ABL1 KinaseInfo once (canonical sequence + KD bounds)."""
    from mkt.schema.io_utils import deserialize_kinase_dict

    return deserialize_kinase_dict(list_ids=["ABL1"])["ABL1"]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestAlphaFoldPredictionSingleIsoform:
    def test_json_not_none(self, af_prediction_abl1):
        assert af_prediction_abl1._json is not None

    def test_uniprot_accession(self, af_prediction_abl1):
        assert af_prediction_abl1._json["uniprotAccession"] == "P00519"

    def test_has_cif_url(self, af_prediction_abl1):
        assert "cifUrl" in af_prediction_abl1._json


@pytest.mark.network
class TestAlphaFoldPredictionMultipleIsoforms:
    def test_json_not_none(self, af_prediction_egfr):
        assert af_prediction_egfr._json is not None

    def test_uniprot_accession(self, af_prediction_egfr):
        assert af_prediction_egfr._json["uniprotAccession"] == "P00533"

    def test_has_cif_url(self, af_prediction_egfr):
        assert "cifUrl" in af_prediction_egfr._json


@pytest.mark.network
class TestAlphaFoldPredictionInvalid:
    def test_json_is_none(self, af_prediction_invalid):
        assert af_prediction_invalid._json is None


@pytest.mark.network
class TestAlphaFoldPredictionMultiModel:
    """FGR (P09769) returns both an AF2 (AF-P09769-F1) and a newer numeric-id AF3 model.

    The canonical AF2 fragment-1 entry must be selected rather than the query being rejected
    for having more than one result.
    """

    def test_selects_af2_f1_model(self):
        pred = AlphaFoldPrediction(uniprot_id="P09769")
        assert pred._json is not None
        assert pred._json["entryId"] == "AF-P09769-F1"


@pytest.mark.network
class TestAlphaFoldStructureDownload:
    def test_json_not_none(self, af_structure_abl1):
        assert af_structure_abl1._json is not None

    def test_cif_not_none(self, af_structure_abl1):
        assert af_structure_abl1._cif is not None

    def test_cif_starts_with_data(self, af_structure_abl1):
        assert af_structure_abl1._cif.startswith("data_")


@pytest.mark.network
class TestAlphaFoldStructureInvalid:
    def test_json_is_none(self, af_structure_invalid):
        assert af_structure_invalid._json is None

    def test_cif_is_none(self, af_structure_invalid):
        assert af_structure_invalid._cif is None


@pytest.mark.network
class TestFetchAlphaFoldKDValidation:
    """The KD-sliced AF sequence is compared to the canonical UniProt sequence."""

    def test_valid_matches_canonical(self, abl1_kinase):
        """A correct canonical sequence yields a model whose KD slice matches it (no mismatch)."""
        start = abl1_kinase.adjudicate_kd_start()
        end = abl1_kinase.adjudicate_kd_end()
        seq = abl1_kinase.uniprot.canonical_seq
        af = fetch_alphafold_kd("P00519", start, end, canonical_seq=seq)
        assert af is not None
        assert (
            af.cif["_entity_poly.pdbx_seq_one_letter_code"][0] == seq[start - 1 : end]
        )
        assert af.mismatch is None

    def test_mismatch_recorded_not_rejected(self, abl1_kinase, caplog):
        """A single-residue mismatch is recorded (like KinCoRe), not discarded."""
        start = abl1_kinase.adjudicate_kd_start()
        end = abl1_kinase.adjudicate_kd_end()
        seq = list(abl1_kinase.uniprot.canonical_seq)
        # flip one residue inside the KD slice -> canonical position `start` == slice index 1
        seq[start] = "A" if seq[start] != "A" else "G"
        af = fetch_alphafold_kd("P00519", start, end, canonical_seq="".join(seq))
        assert af is not None
        assert af.mismatch == [1]
        assert "recording as mismatch" in caplog.text

    def test_excessive_mismatch_rejected(self, abl1_kinase, caplog):
        """A KD slice diverging beyond max_mismatch_fraction is rejected (returns None)."""
        start = abl1_kinase.adjudicate_kd_start()
        end = abl1_kinase.adjudicate_kd_end()
        seq = list(abl1_kinase.uniprot.canonical_seq)
        # corrupt ~20% of the KD slice, well above the 5% default threshold
        for i in range(start - 1, end, 5):
            seq[i] = "A" if seq[i] != "A" else "G"
        af = fetch_alphafold_kd("P00519", start, end, canonical_seq="".join(seq))
        assert af is None
        assert "rejecting structure" in caplog.text
