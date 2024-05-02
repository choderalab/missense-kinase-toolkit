import requests

from missense_kinase_toolkit import requests_wrapper, utils_requests


def maybe_get_symbol_from_hgnc_search(
    input_symbol_or_id: str,
    input_is_hgnc_symbol: bool = True,
) -> list[str] | None:
    """Get gene name from HGNC REST API using either a gene symbol or an Ensembl gene ID.

    Parameters
    ----------
    input_symbol_or_id : str
        Gene symbol or Ensembl gene ID
    input_is_hgnc_symbol : bool
        If True, input_symbol_or_id is a gene symbol, otherwise it is an Ensembl gene ID

    Returns
    -------
    list[str] | None
        List of gene names that match input_symbol_or_id; empty list if no match and None if request fails
    """
    if input_is_hgnc_symbol:
        url = f"https://rest.genenames.org/search/symbol:{input_symbol_or_id}"
    else:
        url = f"https://rest.genenames.org/search/ensembl_gene_id:{input_symbol_or_id}"

    res = requests_wrapper.get_cached_session().get(
        url, headers={"Accept": "application/json"}
    )

    if res.ok:
        list_hgnc_gene_name = extract_list_from_hgnc_response_docs(res, "symbol")
    else:
        list_hgnc_gene_name = None
        utils_requests.print_status_code_if_res_not_ok(res)

    return list_hgnc_gene_name


def maybe_get_info_from_hgnc_fetch(
    hgnc_gene_symbol: str,
    list_to_extract: list[str] | None = None,
) -> dict | None:
    """Get gene information for a given HGNC gene name from gene symbol report using HGNC REST API.

    Parameters
    ----------
    hgnc_gene_symbol : str
        HGNC gene symbol
    list_to_extract : list[str] | None
        List of fields to extract from the response; if None, defaults to ["locus_type"]

    Returns
    -------
    dict | None
        Dictionary of gene information; empty list if no match and None if request fails or field not found
    """
    url = f"https://rest.genenames.org/fetch/symbol/{hgnc_gene_symbol}"
    res = requests_wrapper.get_cached_session().get(
        url, headers={"Accept": "application/json"}
    )

    if list_to_extract is None:
        list_to_extract = ["locus_type"]

    list_out = []
    if res.ok:
        set_keys = generate_key_set_hgnc_response_docs(res)
        for entry in list_to_extract:
            if entry not in set_keys:
                list_out.append(None)
            else:
                list_entry = extract_list_from_hgnc_response_docs(res, entry)
                list_out.append(list_entry)
    else:
        list_out = [None for _ in list_to_extract]
        utils_requests.print_status_code_if_res_not_ok(res)

    dict_out = dict(zip(list_to_extract, list_out))

    return dict_out


def extract_list_from_hgnc_response_docs(
    res_input: requests.models.Response,
    str_to_extract: str,
) -> list[str]:
    """Extract a list of values from the response documents of an HGNC REST API request.

    Parameters
    ----------
    res_input : requests.models.Response
        Response object from an HGNC REST API request
    str_to_extract : str
        Key to extract from the response documents

    Returns
    -------
    list[str]
        List of values extracted from the response documents
    """
    if res_input.json()["response"]["numFound"] >= 1:
        list_output = [
            doc[str_to_extract] for doc in res_input.json()["response"]["docs"]
        ]
    else:
        list_output = []
    return list_output


def generate_key_set_hgnc_response_docs(
    res_input: requests.models.Response,
) -> set[str]:
    """Generate a set of keys present in the response documents of an HGNC REST API request.

    Parameters
    ----------
    res_input : requests.models.Response
        Response object from an HGNC REST API request

    Returns
    -------
    set[str]
        Set of keys present in the response documents
    """
    list_keys = [set(doc.keys()) for doc in res_input.json()["response"]["docs"]]
    set_keys = set.union(*list_keys)
    return set_keys
