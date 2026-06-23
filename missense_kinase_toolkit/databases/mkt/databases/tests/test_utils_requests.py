import pytest
from mkt.databases import utils_requests


@pytest.mark.network
class TestPrintStatusCode:
    def test_print_status_with_matching_code(
        self, uniprot_bad_request_response, capsys
    ):
        """Custom status message is printed when code is in dict_status_code."""
        utils_requests.print_status_code_if_res_not_ok(
            uniprot_bad_request_response,
            dict_status_code={400: "TEST"},
        )
        out, _ = capsys.readouterr()
        assert out == "Error code: 400 (TEST)\n"

    def test_print_status_without_matching_code(
        self, uniprot_bad_request_response, capsys
    ):
        """Generic status message is printed when code is not in dict_status_code."""
        utils_requests.print_status_code_if_res_not_ok(
            uniprot_bad_request_response,
            dict_status_code={200: "TEST"},
        )
        out, _ = capsys.readouterr()
        assert out == "Error code: 400\n"


@pytest.mark.network
class TestUniProtFASTAErrorHandling:
    def test_invalid_uniprot_id_prints_error(self, capsys):
        """UniProtFASTA prints error for invalid (but pattern-conforming) ID."""
        from mkt.databases.uniprot import UniProtFASTA

        uniprot_id = "L91119"
        UniProtFASTA(uniprot_id)
        out, _ = capsys.readouterr()
        assert out == f"Error code: 400 (Bad request)\nUniProt ID: {uniprot_id}\n\n"
