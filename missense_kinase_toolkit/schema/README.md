# mkt.schema

The `schema` sub-package provides harmonized and pre-processed sequence and structure data along with `Pydantic` models to load, query, and validate this data. A notebook

This latest version of this package can be installed directly from Github via `pip` using the following:
```
pip install git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/schema
```

To load the package data in the form of a dictionary where the kinase HGNC names are the keys and the `KinaseInfo` `Pydantic` models are the values use the following:
```
from mkt.schema import io_utils
dict_kinase = io_utils.deserialize_kinase_dict()
```

The `KinaseInfo` object contains the following relevant fields:
| Field        | Description                                                                                                                                                             |
| :-:          | :-                                                                                                                                                                      |
| `hgnc_name`  | Hugo Gene Nomenclature Commitee gene name                                                                                                                               |
| `uniprot_id` | UniProt ID                                                                                                                                                              |
| `kinhub`     | Information scraped from [KinHub](http://www.kinhub.org/)                                                                                                               |
| `uniprot`    | Canonical sequence from [UniProt](https://www.uniprot.org/)                                                                                                             |
| `klifs`      | Information from [KLIFS](https://klifs.net/) API query, including KLIFS pocket sequence                                                                                 |
| `pfam`       | Annotated kinase domain from [Pfam](https://www.ebi.ac.uk/interpro/entry/pfam) (includes "Protein kinase domain" and <br>"Protein tyrosine and serine/threonine kinase" only), aligned to UniProt canonical sequence |
| `kincore`    | Annotated kinase domain from Dunbrack lab's [KinCore](http://dunbrack.fccc.edu/kincore/activemodels), aligned to UniProt canonical sequence                             |

The code to generate these can be found in the `databases` pacakge. This sub-package is designed to
