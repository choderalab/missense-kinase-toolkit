import copy

import pytest


@pytest.fixture(scope="session")
def dict_kinase():
    """Deserialize the bundled KinaseInfo dictionary once per session.

    Reuses the module-level ``_deserialization_cache`` in ``mkt.schema.io_utils``
    so each xdist worker reads the bundled tar.gz only once. This fixture is
    read-only; tests that mutate entries must use ``mutable_kinase`` instead to
    avoid contaminating other tests or parallel workers.

    Returns
    -------
    dict[str, KinaseInfo]
        Dictionary of KinaseInfo objects keyed by HGNC name.
    """
    from mkt.schema.io_utils import deserialize_kinase_dict

    return deserialize_kinase_dict(str_name="DICT_KINASE")


@pytest.fixture
def mutable_kinase(dict_kinase):
    """Return a factory that deep-copies a single KinaseInfo for mutation.

    Deep-copying one object (rather than the full 566-entry dict) keeps mutating
    tests fast while isolating changes from the session-scoped read-only dict.

    Parameters
    ----------
    dict_kinase : dict[str, KinaseInfo]
        Session-scoped read-only kinase dictionary.

    Returns
    -------
    Callable[[str], KinaseInfo]
        Function mapping an HGNC name to a deep-copied KinaseInfo object.
    """

    def _copy(hgnc_name):
        return copy.deepcopy(dict_kinase[hgnc_name])

    return _copy
