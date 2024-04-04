import pandas as pd

from missense_kinase_toolkit import requests_wrapper


def scrape_kinhub(
    url: str = "http://www.kinhub.org/kinases.html",
) -> pd.DataFrame:
    """Scrape the KinHub database for kinase information

    Parameters
    ----------
    url : str
        URL of the KinHub database

    Returns
    -------
    pd.DataFrame
        DataFrame of kinase information
    """
    import requests
    from bs4 import BeautifulSoup
    import numpy as np
    # TODO: to fix ImportError
    # .venv/lib/python3.11/site-packages/janitor.py line 6
    # "import ConfigParser" to "import configparser"
    # perhaps just write own function to clean column names
    # from janitor import clean_names

    page = requests_wrapper.get_cached_session().get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    list_header = [t for tr in soup.select('tr') for t in tr if t.name == 'th']
    dict_kinhub = {key.text.split('\n')[0]: [] for key in list_header}

    list_body = [t.text for tr in soup.select('tr') for t in tr if t.name == 'td']
    list_keys = list(dict_kinhub.keys())
    mult = len(list_keys)

    i = 1
    for entry in list_body:
        if entry == '' or entry == 'nan':
            dict_kinhub[list_keys[i-1]].append(np.nan)
        else:
            dict_kinhub[list_keys[i-1]].append(entry)

        if i % mult == 0:
            i = 1
        else:
            i +=1

    df_kinhub = pd.DataFrame.from_dict(dict_kinhub)
    # df_kinhub = clean_names(df_kinhub)

    return df_kinhub
