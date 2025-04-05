# `mkt` data

| Data  | Source | Description |
| :---- | :----: | :---------- |
| 3. PKIS Nanosyn Assay Heatmaps.xlsx | [DOI: 10.1371/journal.pone.0181585](https://pmc.ncbi.nlm.nih.gov/articles/PMC5540273/) | Provides approximate $k_{M, ATP}$ values |
| AF2-active.fasta | [DOI: 10.1101/2023.07.21.550125](http://dunbrack.fccc.edu/kincore/activemodels) | Provides the sequence information for the active state AF2-generated active structures |
| Human-PK.fasta   | [DOI: https://doi.org/10.1038/s41598-019-56499-4](http://dunbrack.fccc.edu/kincore/alignment) |  Provides the sequence information used to generate initial Dunbrack multiple sequence alignments in 2019 |
| AF2-active.fasta | [DOI: 10.1101/2023.07.21.550125](http://dunbrack.fccc.edu/kincore/activemodels) | Compressed folder containing CIF files of AF2-generated active structures |

## Package data

In the `mkt.schema` sub-package, the `kinase_schema` module provides a `KinaseInfo` Pydantic model that are stored as `json` files as part of the package data. In the `mkt.databases` sub-package, the `kinase_schema` module provides Pydantic models to generate these objects using webscrapers and API clients to extract data from `Kinhub`, `KlIFS`, `Kincore`, `UniProt`, and `Pfam`. The generators will also validate, correct, and generate the following:
+ Correct KLIFS pocket residues manually adjudicated to be incorrect
+ Validate that the canonical UniProt sequence length matches the protein length in Pfam
+ Provide a dictionary that aligns the 85 residue in the `KLIFS2UniProt` attribute

Load these KinaseInfo objects using the following code after installing the `mkt.schema` sub-package:
```
from mkt.schema import io_utils

dict_kinase = io_utils.deserialize_kinase_dict()
```
