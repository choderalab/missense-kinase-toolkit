import pytest
import requests


def pytest_configure(config):
    config.addinivalue_line("markers", "network: marks tests requiring network access")


# ---------------------------------------------------------------------------
# shared config helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def configured_output_dir():
    """Set OUTPUT_DIR once for the entire test session."""
    from mkt.databases import config

    config.set_output_dir(".")


@pytest.fixture(scope="session")
def configured_cbioportal(configured_output_dir):
    """Set cBioPortal config once for the entire test session."""
    from mkt.databases import config

    config.set_cbioportal_instance("www.cbioportal.org")


# ---------------------------------------------------------------------------
# EGFR fixtures (shared by test_kincore, test_klifs, test_uniprot)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def egfr_uniprot():
    """Fetch EGFR UniProt FASTA once (P00533)."""
    from mkt.databases.uniprot import UniProtFASTA

    return UniProtFASTA("P00533")


@pytest.fixture(scope="session")
def kincore_harmonized_dict():
    """Build harmonized KinCore FASTA/CIF dict once."""
    from mkt.databases.kincore import harmonize_kincore_fasta_cif

    return harmonize_kincore_fasta_cif()


@pytest.fixture(scope="session")
def egfr_kincore_alignment(kincore_harmonized_dict, egfr_uniprot):
    """Align EGFR KinCore sequence to UniProt once."""
    from mkt.databases.kincore import align_kincore2uniprot

    return align_kincore2uniprot(
        str_kincore=kincore_harmonized_dict["P00533"][0].fasta.seq,
        str_uniprot=egfr_uniprot._sequence,
    )


@pytest.fixture(scope="session")
def egfr_klifs_info():
    """Fetch EGFR from KLIFS API once."""
    from mkt.databases import klifs

    return klifs.KinaseInfo("EGFR")


@pytest.fixture(scope="session")
def egfr_klifs_pocket(egfr_uniprot, egfr_klifs_info, egfr_kincore_alignment):
    """Build EGFR KLIFSPocket once (depends on KLIFS + KinCore data)."""
    from mkt.databases import klifs

    if egfr_klifs_info.status_code != 200:
        pytest.skip("KLIFS API returned non-200 status")
    dict_egfr = egfr_klifs_info.get_kinase_info()[0]
    return klifs.KLIFSPocket(
        uniprotSeq=egfr_uniprot._sequence,
        klifsSeq=dict_egfr["pocket"],
        idx_kd=(
            egfr_kincore_alignment["start"] - 1,
            egfr_kincore_alignment["end"] - 1,
        ),
    )


# ---------------------------------------------------------------------------
# reusable bad-request response (used by test_utils_requests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def uniprot_bad_request_response():
    """Make a single bad request to UniProt for error-handling tests."""
    return requests.get("https://rest.uniprot.org/uniprotkb/TEST")
