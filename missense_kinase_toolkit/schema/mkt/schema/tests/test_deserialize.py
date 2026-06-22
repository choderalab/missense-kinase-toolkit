import logging


def test_cache_identity(dict_kinase):
    """Test that deserializing by cached name returns the same object."""
    from mkt.schema.io_utils import deserialize_kinase_dict

    # the session fixture already populated the cache under "DICT_KINASE";
    # a second by-name call must return that same cached object, not re-read
    assert deserialize_kinase_dict(str_name="DICT_KINASE") is dict_kinase


def test_dict_counts(dict_kinase):
    """Test deserialized dictionary size and per-source population counts."""
    assert len(dict_kinase) == 566
    assert (
        sum(["_" in i for i in dict_kinase.keys()]) == 28
    )  # 14 proteins with multiple KDs

    # missing data
    n_klifs = len([i.hgnc_name for i in dict_kinase.values() if i.klifs is not None])
    assert n_klifs == 555

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
    assert n_pfam == 490

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
        dict_kinase["ABR"].extract_sequence_from_cif(bool_verbose=True) is None
    )  # no Kincore
    assert "No CIF sequence for ABR" in caplog.text


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
        == "VVEPYRKYPTLLEVLLNFLKTEQNQGTRREAIRVLGLLGALDPYKHKVNIGMIDQSRDASAVSLSESKSSQDSSDYSTSEMLVNMGNLPLDEFYPAVSMVALMRIFRDQSLSHHHTMVVQAITFIFKSLGLKCVQFLPQVMPTFLNVIRVCDGAIREFLFQQLGMLVSFVK"
    )
    caplog.clear()
    assert dict_kinase["ABR"].adjudicate_kd_sequence(bool_verbose=True) is None
    assert "No kinase domain sequence found for ABR" in caplog.text
