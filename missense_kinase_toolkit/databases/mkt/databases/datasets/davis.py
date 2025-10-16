import pandas as pd

# don't currently have a good way to handle these
LIST_DAVIS_DROP = [
    "P.falciparum",  # not human kinase
    "M.tuberculosis",  # not human kinase
    "-phosphorylated",  # no way to handle phosphorylated residues yet
    "-cyclin",  # no way to handle cyclin proteins yet
]
"""List of terms to drop from the Davis DiscoverX dataset."""


def disambiguate_kinase_ids(
    df_in: pd.DataFrame,
    dict_in: dict,
    col_check: str | None = "Entrez Gene Symbol",
    bool_mono: bool = True,
):
    """Use DiscoverX supplemental file to disambiguate IDs.

    Parameters
    ----------
    df_in: pd.DataFrame
        Dataframe containing "DiscoverX Gene Symbol" column and col_check column if provided
    dict_in: dict
        DICT_KINASE object
    col_check: str | None
        Davis also supplies "Entrez Gene Symbol" to check against;
            PKIS2 does not so use None argument there.
    bool_mono: bool
        If True, only search for exact key matches;
            otherwise split on "_" to match multi-KD entries

    Returns
    -------
    list[str]
        List of strings containing matches to dictionary keys;
            no match returns empty string.
    """
    if bool_mono:
        list_ids = [v.hgnc_name for v in dict_in.values()]
    else:
        list_ids = list({v.hgnc_name.split("_")[0] for v in dict_in.values()})

    list_kinase_set = df_in["DiscoverX Gene Symbol"].apply(
        lambda x: {i for i in list_ids if i == x.split("(")[0]}
    )
    if col_check:
        list_check_set = df_in[col_check].apply(
            lambda x: {i for i in list_ids if i == x}
        )
        list_combo_set = [i | j for i, j in zip(list_check_set, list_kinase_set)]
    else:
        list_combo_set = list_kinase_set

    list_combo_str = ["".join(list(i)) for i in list_combo_set]

    return list_combo_str
