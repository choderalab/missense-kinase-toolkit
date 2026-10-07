import pytest
import requests
from mkt.databases import utils_requests


def _response(int_status_code):
    """A response with only a status code, built locally (no request is made)."""
    response = requests.Response()
    response.status_code = int_status_code
    return response


class TestPrintStatusCode:
    def test_print_status_with_matching_code(self, capsys):
        """Custom status message is printed when code is in dict_status_code."""
        utils_requests.print_status_code_if_res_not_ok(
            _response(400),
            dict_status_code={400: "TEST"},
        )
        out, _ = capsys.readouterr()
        assert out == "Error code: 400 (TEST)\n"

    def test_print_status_without_matching_code(self, capsys):
        """Generic status message is printed when code is not in dict_status_code."""
        utils_requests.print_status_code_if_res_not_ok(
            _response(400),
            dict_status_code={200: "TEST"},
        )
        out, _ = capsys.readouterr()
        assert out == "Error code: 400\n"

    def test_print_status_default_codes(self, capsys):
        """Without dict_status_code, the default table names the status."""
        utils_requests.print_status_code_if_res_not_ok(_response(503))
        out, _ = capsys.readouterr()
        assert out == "Error code: 503 (Service unavailable)\n"


@pytest.mark.network
class TestUniProtFASTAErrorHandling:
    def test_invalid_uniprot_id_prints_error(self, capsys):
        """UniProtFASTA prints error for invalid (but pattern-conforming) ID."""
        from mkt.databases.uniprot import UniProtFASTA

        uniprot_id = "L91119"
        UniProtFASTA(uniprot_id)
        out, _ = capsys.readouterr()
        # a 5xx means UniProt is down, not that the error handling is wrong
        if out.startswith("Error code: 5"):
            pytest.skip(f"UniProt unavailable: {out.splitlines()[0]}")
        assert out == f"Error code: 400 (Bad request)\nUniProt ID: {uniprot_id}\n\n"
