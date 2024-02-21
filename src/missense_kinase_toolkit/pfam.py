from __future__ import annotations

import json

import pandas as pd

from missense_kinase_toolkit import requests_wrapper


def retrieve_pfam(uniprot_id: str) -> pd.DataFrame | str | None:
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

    url = "https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/UniProt/" + uniprot_id

    res = requests_wrapper.get_cached_session().get(
        url, headers={"Accept": "application/json"}
    )

    if res.ok:
        dict_json = json.loads(res.text)["results"]
        try:
            df1_out = pd.DataFrame()
            df2_out = pd.DataFrame()

            for entry in dict_json:
                df1_temp = pd.DataFrame.from_dict(
                    entry["metadata"], orient="index"
                ).transpose()
                df1_out = pd.concat([df1_out, df1_temp]).reset_index(drop=True)

                df2_temp = pd.DataFrame.from_dict(
                    entry["proteins"][0], orient="index"
                ).transpose()
                df2_out = pd.concat([df2_out, df2_temp]).reset_index(drop=True)

            df1_out = df1_out.rename(columns={"accession": "pfam_accession"})
            df2_out = df2_out.rename(
                columns={
                    "accession": "uniprot_accession",
                    "source_database": "review_status",
                }
            )

            df_out = pd.concat([df1_out, df2_out], axis=1)
            df_out = df_out.explode("entry_protein_locations").reset_index(drop=True)

            list_entry = ["model", "score"]
            for entry in list_entry:
                df_out[entry] = df_out["entry_protein_locations"].apply(
                    lambda x: x[entry]
                )

            list_fragments = ["start", "end", "dc-status"]
            for entry in list_fragments:
                df_out[entry] = df_out["entry_protein_locations"].apply(
                    lambda x: x["fragments"][0][entry]
                )

            del df_out["entry_protein_locations"]

            return df_out
        except KeyError:
            print("Error:")
            print(dict_json)
            print()
            return None
    else:
        return uniprot_id


def concat_pfam(
    iter_uniprot: iter[str],
    iter_hgnc: iter[str],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Concatenate Pfam domain information for a list of UniProt IDs

    Parameters
    ----------
    iter_uniprot : iter[str]
        Iterable of UniProt IDs
    iter_hgnc : iter[str]
        Iterable of HGNC symbols

    Returns
    -------
    pd.DataFrame
        DataFrame with Pfam domain information
    dict[str, str]
        Dictionary of HGNC symbols and UniProt IDs with errors
    dict[str, str]
        Dictionary of HGNC symbols and UniProt IDs with missing information
    """
    dict_error = {}
    dict_missing = {}
    df = pd.DataFrame()

    for uniprot, hgnc in zip(iter_uniprot, iter_hgnc):
        temp = retrieve_pfam(uniprot)

        if temp is None:
            dict_error[hgnc] = uniprot
        if type(temp) is str:
            dict_missing[hgnc] = uniprot
        else:
            temp.insert(0, "hgnc", hgnc)
            df = pd.concat([df, temp]).reset_index(drop=True)

    return df, dict_error, dict_missing


def extract_numeric(
    input_string: str,
) -> str:
    """Extract numeric characters from a string

    Parameters
    ----------
    input_string : str
        Input string
    
    Returns
    -------
    str
        Numeric characters extracted from the input string
    """
    num = ""
    for i in input_string:
        if i.isdigit():
            num = num + i
    return num


def find_pfam(
    input_hgnc: str,
    input_position: int,
    df_ref: pd.DataFrame,
) -> str | None:
    """Find Pfam domain for a given HGNC symbol and position

    Parameters
    ----------
    input_hgnc : str
        HGNC symbol
    input_position : int
        Codon position
    df_ref : pd.DataFrame
        DataFrame with Pfam domain information

    Returns
    -------
    str | None
        Pfam domain if found, None if not found
    """
    df_temp = df_ref.loc[df_ref["hgnc"] == input_hgnc].reset_index()
    try:
        domain = df_temp.loc[
            ((input_position >= df_temp["start"]) & (input_position <= df_temp["end"])),
            "name",
        ].values[0]
        return domain
    except IndexError:
        return None
