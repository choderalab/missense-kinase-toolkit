import pandas as pd

from missense_kinase_toolkit.databases import utils

# drop values that cannot be matched to protein kinase sequence
LIST_DROP = [
    "PI3-KINASE-ALPHA",  # lipid kinase
    "PI3-KINASE-DELTA",  # lipid kinase
    "PI3-KINASE-GAMMA",  # lipid kinase
    "PI4-K-BETA",  # lipid kinase
    "SPHK1",  # lipid kinase
    "SPHK2",  # lipid kinase
    "AMP-A1B1G1",  # B1G1 subunits not kinase entries
    "AMP-A2B1G1",  # B1G1 subunits not kinase entries
    "CK2",  # not sure if CSNK2A1 or CSNK2A2; catalog not available
]

# dictionary of {string match : string replace} for exact matches
DICT_EXACT = {
    "ARK5": "NUAK1",
    "CK1": "CK1a",
    "CRAF": "RAF1",
    "LRRK": "LRRK2",
    "MEK1": "MAP2K1",
    "PAR-1B-ALPHA": "MARK2",
    "PKC-BETA1": "PKC-B",
    "PRAK": "MAPKAPK5",
    "PTK5": "FRK",
    "SNF1LK2": "QIK",
}

# dictionary of {string match : string replace} for partial matches
DICT_PARTIAL = {
    "AURORA": "AURK",
    "ALPHA": "A",
    "BETA": "B",
    "DELTA": "D",
    "EPSILON": "E",
    "GAMMA": "G",
    "-ETA": "H",
    "IOTA": "I",
    "THETA": "T",
}


def read_xlsx_file(
    str_path: str = "../../../data/3. PKIS Nanosyn Assay Heatmaps.xlsx",
    str_sheet: str = "Assay and Panel information",
) -> pd.DataFrame:
    df = pd.read_excel(str_path, sheet_name=str_sheet)
    return df


# def main():
df_pkis2_km_atp = pd.read_csv("data/pkis2_km_atp.csv")

# plot stacked barchart of PKIS2 ATP Km values where X-axis is the
