import json

import pandas as pd

from missense_kinase_toolkit import requests_wrapper, utils_requests


def retrieve_pfam(
    uniprot_id: str,
) -> pd.DataFrame | str | None:
    """Retrieve Pfam domain information for a given UniProt ID using InterPro REST API

    Parameters
    ----------
    uniprot_id : str
        UniProt ID

    Returns
    -------
    pd.DataFrame | str | None
        DataFrame with Pfam domain information if request is successful, UniProt ID if request fails;
          None if response is empty
    """
    url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/UniProt/{uniprot_id}"

    header = {"Accept": "application/json"}
    res = requests_wrapper.get_cached_session().get(
        url,
        headers=header
    )

    if res.ok:
        if len(res.text) == 0:
            print(f"No PFAM domains found: {uniprot_id}")
            return None
        else:
            list_json = json.loads(res.text)["results"]

            # metadata for UniProt ID
            list_metadata = [entry["metadata"] for entry in list_json]
            list_metadata = [{"pfam_accession" if k == "accession" else k:v for k,v in entry.items()} for entry in list_metadata]

            # Pfam domains locations
            list_locations = [entry["proteins"][0]["entry_protein_locations"][0]["fragments"][0] for entry in list_json]

            # model information
            list_model = [entry["proteins"][0]["entry_protein_locations"][0] for entry in list_json]
            [entry.pop("fragments", None) for entry in list_model]

            # protein information
            # do last because pop is an in-place operation
            list_protein = [entry["proteins"][0] for entry in list_json]
            [entry.pop("entry_protein_locations", None) for entry in list_protein]
            list_protein = [{"uniprot" if k == "accession" else k:v for k,v in entry.items()} for entry in list_protein]

            df_concat = pd.concat(
                [
                    pd.DataFrame(list_protein),
                    pd.DataFrame(list_metadata),
                    pd.DataFrame(list_locations),
                    pd.DataFrame(list_model)
                ],
                 axis=1
            )

            return df_concat
    else:
        utils_requests.print_status_code_if_res_not_ok(res)
        return None


# def find_pfam(
#     input_hgnc: str,
#     input_position: int,
#     df_ref: pd.DataFrame,
# ) -> str | None:
#     """Find Pfam domain for a given HGNC symbol and position

#     Parameters
#     ----------
#     input_hgnc : str
#         HGNC symbol
#     input_position : int
#         Codon position
#     df_ref : pd.DataFrame
#         DataFrame with Pfam domain information

#     Returns
#     -------
#     str | None
#         Pfam domain if found, None if not found
#     """
#     df_temp = df_ref.loc[df_ref["hgnc"] == input_hgnc].reset_index()
#     try:
#         domain = df_temp.loc[
#             ((input_position >= df_temp["start"]) & (input_position <= df_temp["end"])),
#             "name",
#         ].values[0]
#         return domain
#     except IndexError:
#         return None
