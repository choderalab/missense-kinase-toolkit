def test_rgetattr_rsetattr(mutable_kinase):
    """Test recursive attribute getter and setter helpers.

    Uses ``mutable_kinase`` because ``rsetattr`` mutates the ABL1 object.
    """
    from mkt.schema import utils

    obj = mutable_kinase("ABL1")

    # test rgetattr
    assert utils.rgetattr(obj, attr="hgnc_name") == "ABL1"
    assert utils.rgetattr(obj, attr="uniprot_id") == "P00519"
    assert utils.rgetattr(obj, attr="non_existent") is None

    # test rsetattr
    utils.rsetattr(obj=obj, attr="hgnc_name", val="ABL2")
    assert obj.hgnc_name == "ABL2"
    utils.rsetattr(obj=obj, attr="kincore.fasta.seq", val=None)
    assert obj.kincore.fasta.seq is None


def test_random_uuid():
    """Test deterministic UUID generation under a fixed seed."""
    import random

    from mkt.schema import utils

    random.seed(42)
    uuid = utils.random_uuid()
    assert str(uuid) == "a31c06bd-463e-4923-bc1a-adbde48b1697"


def test_group_name_homologs():
    """Test homolog grouping, receptor-paralog split, and hand-curated exceptions."""
    from mkt.schema import utils

    def _grouped(names):
        return {
            label: members
            for label, members in utils.group_name_homologs(names, show_count=False)
        }

    # numbered / single-letter subfamilies still collapse
    assert _grouped(["JAK1", "JAK2", "JAK3"]) == {"JAK1/2/3": ["JAK1", "JAK2", "JAK3"]}
    assert _grouped(["MYLK", "MYLK2", "MYLK3", "MYLK4"]) == {
        "MYLK/2/3/4": ["MYLK", "MYLK2", "MYLK3", "MYLK4"]
    }
    assert _grouped(["ACVR1", "ACVR1B", "ACVR1C"]) == {
        "ACVR1/B/C": ["ACVR1", "ACVR1B", "ACVR1C"]
    }

    # a complete gene name and its stem + "R" receptor paralog never merge
    assert _grouped(["INSR", "INSRR"]) == {"INSR": ["INSR"], "INSRR": ["INSRR"]}

    # hand-curated exception: BUB1 / BUB1B are distinct genes despite the "B" suffix
    assert _grouped(["BUB1", "BUB1B"]) == {"BUB1": ["BUB1"], "BUB1B": ["BUB1B"]}


def test_return_klifs2msa_dict(dict_kinase):
    """The empirical KLIFS->MSA map covers the pocket; core anchors are highly concordant."""
    from mkt.schema.utils import return_klifs2msa_dict

    dict_map, dict_concordance = return_klifs2msa_dict(
        dict_kinase, bool_return_concordance=True
    )
    # core catalytic anchors map to their known MSA columns
    assert dict_map["III:17"] == "B3:028"  # VAIK beta3 lysine
    assert dict_map["c.l:70"] == "CL:111"  # HRD catalytic aspartate
    assert dict_map["xDFG:81"] == "ALN:129"  # DFG aspartate
    # concordance is near-perfect at the anchors but not 1:1 across the pocket
    assert dict_concordance["xDFG:81"] >= 0.98
    assert min(dict_concordance.values()) >= 0.85


def test_catalytic_klifs2msa_constant_matches_corpus(dict_kinase):
    """The precomputed DICT_KLIFS2MSA_CATALYTIC matches the map derived from the corpus."""
    from mkt.schema.constants import DICT_KLIFS2MSA_CATALYTIC
    from mkt.schema.utils import return_catalytic_klifs2msa_dict

    assert return_catalytic_klifs2msa_dict(dict_kinase) == DICT_KLIFS2MSA_CATALYTIC


def test_split_domain_suffix():
    """Suffixes may have several digits; a trailing digit without an underscore is kept."""
    from mkt.schema.utils import split_domain_suffix

    assert split_domain_suffix("JAK1_1") == ("JAK1", "_1")
    assert split_domain_suffix("JAK1_10") == ("JAK1", "_10")
    assert split_domain_suffix("A_B_2") == ("A_B", "_2")
    assert split_domain_suffix("SGK1") == ("SGK1", "")
    assert split_domain_suffix("BTK") == ("BTK", "")
    assert split_domain_suffix("_1") == ("_1", "")


def test_adjudicate_kinase_group_without_corpus(monkeypatch):
    """Groups come from precomputed constants, so the corpus is never loaded."""
    from mkt.schema import io_utils
    from mkt.schema.constants import DICT_KINASE_GROUP, DICT_KINASE_GROUP_COLORS
    from mkt.schema.utils import adjudicate_kinase_group

    def _no_load(*args, **kwargs):
        raise AssertionError("adjudicate_kinase_group loaded DICT_KINASE")

    monkeypatch.setattr(io_utils, "deserialize_kinase_dict", _no_load)

    assert adjudicate_kinase_group("JAK1") == "TK"
    assert adjudicate_kinase_group("RPS6KA4") == "Multiple"
    assert adjudicate_kinase_group("RPS6KA4_1") == "AGC"
    assert adjudicate_kinase_group("RPS6KA4_2") == "CAMK"
    assert adjudicate_kinase_group("PIK3CA") == "Lipid"
    assert (
        adjudicate_kinase_group("PIK3CA", bool_lipid=False)
        == DICT_KINASE_GROUP["PIK3CA"]
    )
    assert adjudicate_kinase_group("NOTAKINASE") is None
    # every group the lookup can return has a palette color
    assert set(DICT_KINASE_GROUP.values()) | {"Lipid"} <= set(DICT_KINASE_GROUP_COLORS)


def test_kinase_group_constants_match_corpus(dict_kinase):
    """The precomputed group and lipid constants match those derived from the corpus."""
    from mkt.schema.constants import DICT_KINASE_GROUP, SET_LIPID_KINASE
    from mkt.schema.utils import return_kinase_group_dict, return_lipid_kinase_set

    assert return_kinase_group_dict(dict_kinase) == DICT_KINASE_GROUP
    assert return_lipid_kinase_set(dict_kinase) == SET_LIPID_KINASE
