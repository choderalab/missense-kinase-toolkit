import pytest
import requests as req_lib
from mkt.databases import ncbi
from mkt.schema.io_utils import deserialize_kinase_dict


@pytest.fixture(scope="module")
def tgfbr2_canonical():
    """UniProt canonical TGFBR2 (P37173) from the packaged KinaseInfo archive."""
    dict_kinase = deserialize_kinase_dict(list_ids=["TGFBR2"], bool_verbose=False)
    return dict_kinase["TGFBR2"].uniprot.canonical_seq


@pytest.mark.network
class TestCDSTranslation:
    def test_canonical_isoform_matches_uniprot(self, tgfbr2_canonical):
        assert ncbi.get_cds_translation("NM_003242") == tgfbr2_canonical

    def test_longer_isoform_is_translated(self, tgfbr2_canonical):
        # TGFBR2 isoform A carries a 25-residue exon absent from the canonical
        seq = ncbi.get_cds_translation("NM_001024847")
        assert len(seq) == len(tgfbr2_canonical) + 25

    def test_unknown_accession_returns_none(self):
        assert ncbi.get_cds_translation("NM_999999999") is None


@pytest.fixture(scope="module")
def ncbi_brsk2():
    """Fetch NCBI protein record for EAX02438.1 once."""
    try:
        return ncbi.ProteinNCBI(accession="EAX02438.1")
    except req_lib.exceptions.RetryError as e:
        if "500 error responses" in str(e):
            pytest.skip("NCBI API returned 500 errors - skipping test")
        raise


@pytest.mark.network
class TestProteinNCBI:
    def test_header(self, ncbi_brsk2):
        assert ncbi_brsk2.list_headers == [
            "EAX02438.1 BR serine/threonine kinase 2, isoform CRA_c [Homo sapiens]"
        ]

    def test_sequence(self, ncbi_brsk2):
        assert ncbi_brsk2.list_seq == [
            "MTSTGKDGGAQHAQYVGPYRLEKTLGKGQTGLVKLGVHCVTCQKVAIKIVNREKLSESVLMKVEREIAILKLIEHPHVLKLHDVYENKKYLYLVLEHVSGGELFDYLVKKGRLTPKEARKFFRQIISALDFCHSHSICHRDLKPENLLLDEKNNIRIADFGMASLQVGDSLLETSCGSPHYACPEVIRGEKYDGRKADVWSCGVILFALLVGALPFDDDNLRQLLEKVKRGVFHMPHFIPPDCQSLLRGMIEVDAARRLTLEHIQKHIWYIGGKNEPEPEQPIPRKVQIRSLPSLEDIDPDVLDSMHSLGCFRDRNKLLQDLLSEEENQEKMIYFLLLDRKERYPSQEDEDLPPRNEIDPPRKRVDSPMLNRHGKRRPERKSMEVLSVTDGGSPVPARRAIEMAQHGQRSRSISGASSGLSTSPLSSPRVTPHPSPRGSPLPTPKGTPVHTPKESPAGTPNPTPPSSPSVGGVPWRARLNSIKNSFLGSPRFHRRKLQVPTPEEMSNLTPESSPELAKKSWFGNFISLEKEEQIFVVIKDKPLSSIKADIVHAFLSIPSLSHSVISQTSFRAEYKATGGPAVFQKPVKFQVDITYTEGGEAQKENGIYSVTFTLLSGPSRRFKRVVETIQAQLLSTHDPPAAQHLSEPPPPAPGLSWGAGLKGQKVATSYESSL"
        ]
