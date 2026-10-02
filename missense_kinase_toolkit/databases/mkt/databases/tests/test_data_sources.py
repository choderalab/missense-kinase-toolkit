import pytest
from mkt.databases.kincore import DICT_SEQ_SOURCE, DICT_STRUCTURE_SOURCE
from mkt.databases.msa import MSA_SOURCE
from mkt.databases.superpose import REFERENCE_SOURCE
from mkt.schema.io_utils import load_manifest, return_str_path_from_pkg_data

LIST_DATA_SOURCES = [
    *DICT_SEQ_SOURCE.values(),
    *DICT_STRUCTURE_SOURCE.values(),
    MSA_SOURCE,
    REFERENCE_SOURCE,
]
"""list[DataSource]: Every file-based source whose SHA-256 the build stamps into provenance."""


@pytest.mark.network
def test_source_files_match_packaged_manifest():
    """Each source file the shipped archive was built from still has the recorded SHA-256.

    A mismatch means the local copy is corrupt, or upstream replaced the file and the
    archive needs regenerating.
    """
    manifest = load_manifest(return_str_path_from_pkg_data())
    if manifest is None or not manifest.source_sha256:
        pytest.skip("packaged manifest records no source SHA-256s (pre-v2 archive)")

    list_checked, list_problems = [], []
    for source in LIST_DATA_SOURCES:
        if source.name not in manifest.source_sha256:
            continue
        # sources without a URL must already be present locally
        if source.url is None and source.sha256() is None:
            continue
        source.resolve()
        str_sha = source.sha256()
        if str_sha not in manifest.source_sha256[source.name]:
            list_problems.append(
                f"{source.name}: {str_sha}, archive built from "
                f"{sorted(manifest.source_sha256[source.name])}"
            )
        list_checked.append(source.name)

    assert list_checked, "no recorded source could be checked"
    assert list_problems == []
