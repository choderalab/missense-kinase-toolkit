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
