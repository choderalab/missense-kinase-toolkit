import logging


def test_cache_identity(dict_kinase):
    """Test that deserializing by cached name returns the same object."""
    from mkt.schema.io_utils import deserialize_kinase_dict

    # the session fixture already populated the cache under "DICT_KINASE";
    # a second by-name call must return that same cached object, not re-read
    assert deserialize_kinase_dict(str_name="DICT_KINASE") is dict_kinase


def test_dict_counts(dict_kinase):
    """Test deserialized dictionary size and per-source population counts."""
    assert len(dict_kinase) == 543
    assert (
        sum(["_" in i for i in dict_kinase.keys()]) == 28
    )  # 14 proteins with multiple KDs

    # missing data
    n_klifs = len([i.hgnc_name for i in dict_kinase.values() if i.klifs is not None])
    assert n_klifs == 539

    n_pocket = len(
        [
            i.hgnc_name
            for i in dict_kinase.values()
            if i.klifs is not None and i.klifs.pocket_seq is not None
        ]
    )
    assert n_pocket == 519

    n_kincore = len(
        [i.hgnc_name for i in dict_kinase.values() if i.kincore is not None]
    )
    assert n_kincore == 492

    n_pfam = len([i.hgnc_name for i in dict_kinase.values() if i.pfam is not None])
    assert n_pfam == 533

    n_klif2uniprot = len(
        [i.hgnc_name for i in dict_kinase.values() if i.KLIFS2UniProtIdx is not None]
    )
    assert n_klif2uniprot == 519


def test_abl1_fields(dict_kinase):
    """Test ABL1 attribute values across all data sources."""
    obj_abl1 = dict_kinase["ABL1"]

    assert obj_abl1.hgnc_name == "ABL1"

    assert obj_abl1.uniprot_id == "P00519"

    assert obj_abl1.kinhub.hgnc_name == "ABL1"
    assert obj_abl1.kinhub.kinase_name == "Tyrosine-protein kinase ABL1"
    assert obj_abl1.kinhub.manning_name == "ABL"
    assert obj_abl1.kinhub.xname == "ABL1"
    assert obj_abl1.kinhub.group == "TK"
    assert obj_abl1.kinhub.family == "Other"

    assert (
        obj_abl1.uniprot.canonical_seq
        == "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCVPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPGEGLKKPVLPATPKPQSAKPSGTPISPAPVPSTLPSASSALAGDQPSSTAFIPLISTRVSLRKTRQPPERIASGAITKGVVLDSTEALCLAISRNSEQMASHSAVLEAGKNLYTFCVSYVDSIQQMRNKFAFREAINKLENNLRELQICPATAGSGPAATQDFSKLLSSVKEISDIVQR"
    )
    assert obj_abl1.uniprot.phospho_sites == [
        50,
        70,
        115,
        128,
        139,
        172,
        185,
        215,
        226,
        229,
        253,
        257,
        393,
        413,
        446,
        559,
        569,
        618,
        619,
        620,
        659,
        683,
        718,
        735,
        751,
        781,
        814,
        823,
        844,
        852,
        855,
        917,
        977,
    ]
    assert (
        sum([i.startswith("Phospho") for i in obj_abl1.uniprot.phospho_description])
        == 33
    )
    assert len(obj_abl1.uniprot.phospho_evidence) == 33

    assert obj_abl1.klifs.gene_name == "ABL1"
    assert obj_abl1.klifs.name == "ABL1"
    assert (
        obj_abl1.klifs.full_name == "ABL proto-oncogene 1, non-receptor tyrosine kinase"
    )
    assert obj_abl1.klifs.group == "TK"
    assert obj_abl1.klifs.family == "Other"
    assert obj_abl1.klifs.iuphar == 1923
    assert obj_abl1.klifs.kinase_id == 392
    assert (
        obj_abl1.klifs.pocket_seq
        == "HKLGGGQYGEVYEVAVKTLEFLKEAAVMKEIKPNLVQLLGVYIITEFMTYGNLLDYLREYLEKKNFIHRDLAARNCLVVADFGLS"
    )

    assert obj_abl1.pfam.domain_name == "Protein tyrosine and serine/threonine kinase"
    assert obj_abl1.pfam.start == 242
    assert obj_abl1.pfam.end == 492
    assert obj_abl1.pfam.pfam_accession == "PF07714"
    assert obj_abl1.pfam.in_alphafold is True

    assert (
        obj_abl1.kincore.fasta.seq
        == "KWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSIS"
    )
    assert obj_abl1.kincore.fasta.start == 234
    assert obj_abl1.kincore.fasta.end == 503
    assert obj_abl1.kincore.mismatch is None
    assert obj_abl1.kincore.start == 1
    assert obj_abl1.kincore.end == 270


def test_abl1_klifs_mappings(dict_kinase):
    """Test ABL1 KLIFS-to-UniProt sequence and index mappings."""
    obj_abl1 = dict_kinase["ABL1"]

    str_dict = "".join(
        [
            v
            for k, v in obj_abl1.KLIFS2UniProtSeq.items()
            if v is not None and ":" not in k and "_intra" not in k
        ]
    )
    assert obj_abl1.klifs.pocket_seq == str_dict

    assert min(obj_abl1.KLIFS2UniProtIdx.values()) == 246
    assert max(obj_abl1.KLIFS2UniProtIdx.values()) == 385


def test_extract_sequence_from_cif(dict_kinase, caplog):
    """Test CIF sequence extraction and failure logging."""
    caplog.set_level(logging.INFO)

    assert (
        dict_kinase["ABL1"].extract_sequence_from_cif()
        == "KWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSIS"
    )

    # test logger messages for CIF extraction failures
    caplog.clear()
    assert (
        dict_kinase["BUB1B"].extract_sequence_from_cif(bool_verbose=True) is None
    )  # Kincore but no cif
    assert "No CIF sequence for BUB1" in caplog.text

    caplog.clear()
    assert (
        dict_kinase["ADCK1"].extract_sequence_from_cif(bool_verbose=True) is None
    )  # no Kincore
    assert "No CIF sequence for ADCK1" in caplog.text


def test_adjudicate_kd_sequence(dict_kinase, caplog):
    """Test kinase domain sequence adjudication across data-source priorities."""
    caplog.set_level(logging.INFO)

    assert (
        dict_kinase["ABL1"].adjudicate_kd_sequence()
        == "KWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSIS"
    )
    assert (
        dict_kinase["BUB1B"].adjudicate_kd_sequence()
        == "YCIKREYLICEDYKLFWVAPRNSAELTVIKVSSQPVPWDFYINLKLKERLNEDFDHFCSCYQYQDGCIVWHQYINCFTLQDLLQHSEYITHEITVLIIYNLLTIVEMLHKAEIVHGDLSPRCLILRNRIHDPYDCNKNNQALKIVDFSYSVDLRVQLDVFTLSGFRTVQILEGQKILANCSSPYQVDLFGIADLAHLLLFKEHLQVFWDGSFWKLSQNISELKDGELWNKFFVRILNANDEATVSVLGELAAEMNG"
    )
    assert (
        dict_kinase["MTOR"].adjudicate_kd_sequence()
        == "FVFLLKGHEDLRQDERVMQLFGLVNTLLANDPTSLRKNLSIQRYAVIPLSTNSGLIGWVPHCDTLHALIRDYREKKKILLNIEHRIMLRMAPDYDHLTLMQKVEVFEHAVNNTAGDDLAKLLWLKSPSSEVWFDRRTNYTRSLAVMSMVGYILGLGDRHPSNLMLDRLSGKILHIDFGDCFEVAMTREKFPEKIPFRLTRMLTNAMEVTGLDGNYRITCHTVMEVLREHKDSVMAVLEAFVYDPLLNWR"
    )
    caplog.clear()
    assert dict_kinase["PI4KAP1"].adjudicate_kd_sequence(bool_verbose=True) is None
    assert "No kinase domain sequence found for PI4KAP1" in caplog.text


def test_kincore_cif_backward_compatible():
    """KinCoreCIF loads both the pre-v2 (v1) and v2 KinCore CIF layouts.

    Every field that differs between the two Dunbrack releases is optional, so a v1
    record (with template_source/msa_size/msa_source/model_no/min_aloop_pLDDT) and a v2
    record (with model_confidence/dfg_conf/dihedral/snc/af_id) both validate, and each
    leaves the other release's fields as None. Guards against a future required-field
    regression breaking deserialization of the archived dict.
    """
    from mkt.schema.kinase_schema import KinCoreCIF

    cif = {"_entity_poly.pdbx_seq_one_letter_code": ["ABCDEF"]}

    # v1 (pre-AF2_Active_Models_v2) record
    v1 = KinCoreCIF.model_validate(
        {
            "cif": cif,
            "group": "TK",
            "hgnc": "ABL1",
            "min_aloop_pLDDT": 92.61,
            "template_source": "activeAF2",
            "msa_size": 5,
            "msa_source": "family",
            "model_no": 1,
        }
    )
    assert v1.min_aloop_pLDDT == 92.61 and v1.template_source == "activeAF2"
    assert v1.model_confidence is None and v1.dfg_conf is None and v1.af_id is None

    # v2 (AF2_Active_Models_v2) record
    v2 = KinCoreCIF.model_validate(
        {
            "cif": cif,
            "group": "TK",
            "hgnc": "ABL1",
            "model_confidence": 0.89,
            "dfg_conf": "DFGin",
            "dihedral": "BLAminus",
            "snc": "SNCiii",
            "af_id": "AF-P00519-K3A",
        }
    )
    assert v2.model_confidence == 0.89 and v2.dfg_conf == "DFGin"
    assert v2.min_aloop_pLDDT is None and v2.template_source is None


def test_no_legacy_kinases(dict_kinase):
    """Legacy (mostly Manning) kinases are excluded from the canonical dict."""
    from mkt.schema.constants import LIST_LEGACY_KINASES

    present = [name for name in LIST_LEGACY_KINASES if name in dict_kinase]
    assert present == []


def test_pseudokinase_kincore_cif_invariant(dict_kinase):
    """A KinCore active-state CIF marks a catalytically active kinase (never pseudo).

    Regression for the CIF-guard in ``is_pseudokinase``: no entry carrying a KinCore
    CIF may be labeled a pseudokinase, and the three previously-misclassified kinases
    (PDIK1L, SBK3, WNK4) are now catalytically active, while a genuine CIF-less
    pseudokinase (BUB1B) is still flagged.
    """
    violations = [
        name
        for name, info in dict_kinase.items()
        if info.kincore is not None
        and info.kincore.cif is not None
        and info.is_pseudokinase()
    ]
    assert violations == []

    for name in ("PDIK1L", "SBK3", "WNK4"):
        assert dict_kinase[name].is_pseudokinase() is False

    # a genuine pseudokinase without a CIF is still flagged
    assert dict_kinase["BUB1B"].is_pseudokinase() is True
