import requests

from missense_kinase_toolkit.databases import requests_wrapper, utils_requests


class HGNC:
    """Class to interact with the HGNC API."""
    def __init__(
            self,
            input_symbol_or_id: str,
            input_is_hgnc_symbol: bool = True,
    ) -> None:
        """Initialize HGNC Class object.

        Parameters
        ----------
        input_symbol_or_id : str
            Gene symbol or Ensembl gene ID
        input_is_hgnc_symbol : bool
            If True, input_symbol_or_id is a gene symbol, otherwise it is an Ensembl gene ID

        Attributes
        ----------
        url : str
            HGNC API URL
        hgnc : str | None
            HGNC gene symbol
        ensembl : str | None
            Ensembl gene ID

        """
        self.url = "https://rest.genenames.org"
        if input_is_hgnc_symbol:
            self.hgnc = input_symbol_or_id
            self.ensembl = None
        else:
            self.hgnc = None
            self.ensembl = input_symbol_or_id

    def maybe_get_symbol_from_hgnc_search(
        self,
        custom_field: str | None = None,
        custom_term: str | None = None,
    ) -> list[str] | None:
        """Get gene name from HGNC REST API using either a gene symbol or an Ensembl gene ID.

        Parameters
        ----------
        custom_field : str | None
            Optional: custom field to search for in the HGNC REST API; otherwise defaults to "symbol" or "ensembl_gene_id"
            See https://www.genenames.org/help/rest/ under "searchableFields" for options
        custom_term : str | None
            Optional: custom term to search for in the HGNC REST API

        Returns
        -------
        list[str] | None
            List of gene names that match input_symbol_or_id; empty list if no match and None if request fails

        """
        if custom_field is not None:
            url = f"{self.url}/search/{custom_field}:{custom_term}"
        else:
            if self.hgnc is not None:
                url = f"{self.url}/search/symbol:{self.hgnc}"
            else:
                url = f"{self.url}/search/ensembl_gene_id:{self.ensembl}"

        res = requests_wrapper.get_cached_session().get(
            url,
            headers={"Accept": "application/json"}
        )

        if res.ok:
            list_hgnc_gene_name = self._extract_list_from_hgnc_response_docs(res, "symbol")
            if len(list_hgnc_gene_name) == 1:
                if self.hgnc is not None:
                    print(f"Gene name found for {self.hgnc}: {list_hgnc_gene_name[0]}. Overwriting HGNC gene name...")
                else:
                    print(f"Gene name found for {self.ensembl}: {list_hgnc_gene_name[0]}. Adding HGNC gene name...")
                self.hgnc = list_hgnc_gene_name[0]
            elif len(list_hgnc_gene_name) == 0:
                print(f"No gene names found for {self.hgnc}")
            else:
                print(f"Multiple gene names found: {list_hgnc_gene_name}.")
        else:
            list_hgnc_gene_name = None
            utils_requests.print_status_code_if_res_not_ok(res)

        return list_hgnc_gene_name

    def maybe_get_info_from_hgnc_fetch(
        self,
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

        Notes
        -----
            The list of extractable fields can be found at https://www.genenames.org/help/rest/ under "storedFields" in the JSON or XML outputs.

        """
        if self.hgnc is not None:
            url = f"https://rest.genenames.org/fetch/symbol/{self.hgnc}"
            res = requests_wrapper.get_cached_session().get(
                url, headers={"Accept": "application/json"}
            )

            if list_to_extract is None:
                list_to_extract = ["locus_type"]

            list_out = []
            if res.ok:
                set_keys = self._generate_key_set_hgnc_response_docs(res)
                for entry in list_to_extract:
                    if entry not in set_keys:
                        list_out.append(None)
                    else:
                        list_entry = self._extract_list_from_hgnc_response_docs(res, entry)
                        list_out.append(list_entry)
            else:
                list_out = [None for _ in list_to_extract]
                utils_requests.print_status_code_if_res_not_ok(res)

            dict_out = dict(zip(list_to_extract, list_out))

        else:
            print("No HGNC gene symbol provided; cannot fetch gene information from HGNC API with Ensembl gene ID {self.ensembl}")
            dict_out = None

        return dict_out

    @staticmethod
    def _extract_list_from_hgnc_response_docs(
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

    @staticmethod
    def _generate_key_set_hgnc_response_docs(
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
