from os import path
import pandas as pd
from tqdm import tqdm

from mkt.databases.chembl import return_chembl_id
from mkt.schema.io_utils import get_repo_root

def main():

    df_supp = pd.read_excel(path.join(get_repo_root(), "data/41587_2011_BFnbt1990_MOESM4_ESM.xls"), sheet_name=0)

    list_davis_drugs = list(
        map(
            lambda x, y: x if str(y) == "nan" else y, 
            df_supp["Compound Name"], 
            df_supp["Alternative Name"]
        )
    )

    dict_chembl_id = {drug: {"source": None, "ids": []} for drug in list_davis_drugs}
    for drug in tqdm(list_davis_drugs, desc="Querying drugs in ChEMBL"):
        drug_rev = drug.split(" (")[0]
        
        chembl_id, source = return_chembl_id(drug_rev)

        dict_chembl_id[drug]["source"] = source
        dict_chembl_id[drug]["ids"].extend(chembl_id)

    dict_chembl_id_rev = {k: v["ids"][0] for k, v in dict_chembl_id.items()}

    # manually reviewed first entry against MedChemExpress molecules/SMILES
    # {k: v for k, v in dict_chembl_id.items() if v["source"] == "search"}
    DICT_MANUAL = {
        "GSK-1838705A": "CHEMBL464552",
        "MLN-120B": "CHEMBL608154",
        "PD-173955": "CHEMBL386051",
        "PP-242": "CHEMBL1241674",
    }

    dict_chembl_id_rev.update(DICT_MANUAL)

    

if __name__ == "__main__":
    main()