# `mkt` data

## databases

| Data               | Source                                           | Description                                                                      |
| :------------------| :----------------------------------------------: | :--------------------------------------------------------------------------------|
| kinhub.csv         |  [Kinhub](http://www.kinhub.org/kinases.html)    | Database containing human kinases with properties and identifier annotations     |
| kinhub_uniprot.csv |  [UniProt](https://www.uniprot.org/uniprotkb)    | Database containing canonical sequence information                               |
| kinhub_pfam.csv    |  [Pfam](https://www.ebi.ac.uk/interpro/)         | Database containing annotated domains - kinase domain extracted here             |
| kinhub_klifs.csv   |  [KLIFS](https://klifs.net/)                     | Database containing Kinase-Ligand Interaction Fingerprint and Structures (KLIFS) |

The below provides code to generate and save these files:

### KinHub

```
from missense_kinase_toolkit.databases import scrapers

df_kinhub = scrapers.kinhub()

df_kinhub.to_csv("../data/kinhub.csv", index=False)
```

### UniProt

```
from tqdm import tqdm
import pandas as pd
from missense_kinase_toolkit.databases import uniprot

list_uniprot, list_hgnc, list_sequence = [], [], []

for index, row in tqdm(df_kinhub.iterrows(), total = df_kinhub.shape[0]):
    list_uniprot.append(row["UniprotID"])
    list_hgnc.append(row["HGNC Name"])
    list_sequence.append(uniprot.UniProt(row["UniprotID"])._sequence)

dict_uniprot = dict(zip(["uniprot_id", "hgnc_name", "canonical_sequence"], 
                        [list_uniprot, list_hgnc, list_sequence]))
df_uniprot = pd.DataFrame.from_dict(dict_uniprot)

df_uniprot.to_csv("../data/kinhub_uniprot.csv", index=False)
```

### Pfam

```
from tqdm import tqdm
import pandas as pd
from missense_kinase_toolkit.databases import pfam

df_pfam = pd.DataFrame()
for index, row in tqdm(df_kinhub.iterrows(), total = df_kinhub.shape[0]):
    df_temp = pfam.Pfam(row["UniprotID"])._pfam
    df_pfam = pd.concat([df_pfam, df_temp]).reset_index(drop=True)
df_pfam["uniprot"] = df_pfam["uniprot"].str.upper()

df_pfam.to_csv("../data/kinhub_pfam.csv", index=False)
```

### KLIFS

```
from tqdm import tqdm
import pandas as pd
from missense_kinase_toolkit.databases import klifs

df_klifs = pd.DataFrame()
for index, row in tqdm(df_kinhub.iterrows(), total=df_kinhub.shape[0]):
    df_temp = pd.DataFrame(klifs.KinaseInfo(row["UniprotID"], "uniprot")._kinase_info, index=[0])
    df_klifs = pd.concat([df_klifs, df_temp]).reset_index(drop=True)

df_klifs.to_csv("../data/kinhub_klifs.csv", index=False)
```

### Generating Pydantic model

In the `kinase_schema` module, we provide a `KinaseInfo` Pydantic model (values in the dictionary generated below) that can be used to combine the above dataframes using the UniProt ID as the key on which to merge. Each database is contained in a sub-model. This also incorporates the kinase domains derived from the `Human-PK.fasta` file on [KinCore](http://dunbrack.fccc.edu/kincore/home) as the Pfam annotations may be computationally determined and insufficient.

This will also validate, correct, and generate the following:
+ Correct 3 KLIFS pocket residues manually adjudicated to be incorrect (ADCK3, LRRK2, and CAMKK1)
+ Validate that the canonical UniProt sequence length matches the protein length in Pfam
+ Provide a dictionary that aligns the 85 residue in the `KLIFS2UniProt` attribute

```
from missense_kinase_toolkit.databases import kinase_schema

dict_kinase = kinase_schema.create_kinase_models_from_df()
```