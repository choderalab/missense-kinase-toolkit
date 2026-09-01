import os

import pytest
from mkt.schema import io_utils


def _assert_kinase_dicts_equal(expected, actual):
    """Assert two kinase dicts are equal, failing with a compact (value-free) message.

    A plain ``assert expected == actual`` on unequal ``KinaseInfo`` objects makes pytest
    render a diff of the full objects -- which embed large CIF strings and per-residue
    SASA/RSA dicts -- and effectively hangs. Instead compare per kinase and report only the
    differing kinase names and their differing top-level field names (never the values), so
    a round-trip regression fails fast with an actionable pointer.

    Parameters
    ----------
    expected : dict[str, KinaseInfo]
        The original kinase dictionary.
    actual : dict[str, KinaseInfo]
        The round-tripped kinase dictionary.
    """
    if set(expected) != set(actual):
        pytest.fail(
            "kinase key mismatch after round-trip: "
            f"{sorted(set(expected) ^ set(actual))}"
        )

    diffs = []
    for key in expected:
        if expected[key] == actual[key]:
            continue
        dump_expected = expected[key].model_dump()
        dump_actual = actual[key].model_dump()
        fields = sorted(
            name
            for name in dump_expected
            if dump_expected.get(name) != dump_actual.get(name)
        )
        diffs.append(f"{key}: {fields}")

    if diffs:
        pytest.fail(
            "serde round-trip mismatch (kinase: differing top-level fields):\n"
            + "\n".join(diffs)
        )


@pytest.fixture(scope="session")
def serde_sample(dict_kinase):
    """Curated subset of kinases for serialization round-trip tests.

    Covers a tyrosine kinase (ABL1), a CMGC kinase (CDK2), an atypical kinase
    (MTOR), and a Kincore-without-CIF entry (BUB1B), plus one multi-domain
    entry, one lipid kinase, and one pseudogene selected programmatically so the
    sample exercises the schema's edge cases without round-tripping all 566
    objects.

    Parameters
    ----------
    dict_kinase : dict[str, KinaseInfo]
        Session-scoped read-only kinase dictionary.

    Returns
    -------
    dict[str, KinaseInfo]
        Subset of the full kinase dictionary.
    """
    list_sample = ["ABL1", "CDK2", "MTOR", "BUB1B"]

    # add one multi-domain entry (hgnc_name contains "_")
    str_multi = next((k for k in dict_kinase if "_" in k), None)
    # add one lipid kinase and one pseudogene to exercise classification edges
    str_lipid = next((k for k, v in dict_kinase.items() if v.is_lipid_kinase()), None)
    str_pseudo = next((k for k, v in dict_kinase.items() if v.is_pseudogene()), None)

    for str_key in (str_multi, str_lipid, str_pseudo):
        if str_key is not None and str_key not in list_sample:
            list_sample.append(str_key)

    return {k: dict_kinase[k] for k in list_sample if k in dict_kinase}


@pytest.mark.parametrize("suffix", list(io_utils.DICT_FUNCS.keys()))
def test_serde_roundtrip(serde_sample, tmp_path, suffix):
    """Test serialize/deserialize round-trip for each supported format.

    Writes to a per-test ``tmp_path`` subdirectory so parallel xdist workers do
    not collide on a shared output directory.
    """
    if os.name == "nt" and suffix == "toml":
        pytest.skip("TOML serialization is not supported on Windows.")

    # yaml serialization is ~8s per kinase, so round-trip a single entry only
    dict_sample = serde_sample
    if suffix == "yaml":
        dict_sample = {"CDK2": serde_sample["CDK2"]}

    str_path = str(tmp_path / suffix)

    io_utils.serialize_kinase_dict(dict_sample, suffix=suffix, str_path=str_path)
    dict_temp = io_utils.deserialize_kinase_dict(suffix=suffix, str_path=str_path)

    _assert_kinase_dicts_equal(dict_sample, dict_temp)
