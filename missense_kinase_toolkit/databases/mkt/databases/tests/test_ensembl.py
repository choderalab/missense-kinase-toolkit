import pytest
from Bio.Seq import Seq
from mkt.databases.ensembl import get_cds_sequence, get_protein_sequence
from mkt.schema.io_utils import deserialize_kinase_dict

STR_EGFR_TRANSCRIPT = "ENST00000275493"
"""str: EGFR canonical Ensembl transcript (GRCh37)."""


@pytest.fixture(scope="module")
def egfr_canonical():
    """UniProt canonical EGFR (P00533) from the packaged KinaseInfo archive."""
    dict_kinase = deserialize_kinase_dict(list_ids=["EGFR"], bool_verbose=False)
    return dict_kinase["EGFR"].uniprot.canonical_seq


@pytest.mark.network
class TestSequenceById:
    def test_protein_matches_uniprot_canonical(self, egfr_canonical):
        assert get_protein_sequence(STR_EGFR_TRANSCRIPT) == egfr_canonical

    def test_cds_translates_to_protein(self, egfr_canonical):
        cds = get_cds_sequence(STR_EGFR_TRANSCRIPT)
        # the Ensembl CDS includes the stop codon
        assert len(cds) == 3 * len(egfr_canonical) + 3
        assert str(Seq(cds).translate(to_stop=True)) == egfr_canonical

    def test_unknown_transcript_returns_none(self):
        assert get_protein_sequence("ENST00000000000") is None
