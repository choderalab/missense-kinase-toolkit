import requests


def print_status_code_if_res_not_ok(
    res_input: requests.models.Response,
    dict_status_code: dict[int, str] | None = None,
) -> None:
    """Print the status code and status message if the response is not OK

    Parameters
    ----------
    res_input : requests.models.Response
        Response object from an API request
    dict_status_code : dict[int, str] | None
        Dictionary of status codes and status messages; if None, defaults to a standard set of status codes

    Returns
    -------
    None
    """
    if dict_status_code is None:
        dict_status_code = {
            400: "Bad request",
            404: "Not found",
            415: "Unsupported media type",
            500: "Server error",
            503: "Service unavailable",
        }

    try:
        print(
            f"Error code: {res_input.status_code} ({dict_status_code[res_input.status_code]})"
        )
    except KeyError:
        print(f"Error code: {res_input.status_code}")
