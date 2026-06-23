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
