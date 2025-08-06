# mkt.databases

The `databases` sub-package provides the API clients and scraper to query and harmonize various kinase and general protein resources. It can be used to generate the `KinaseInfo` objects in the `mkt.databases` sub-package.

Note that `mkt.schema` is a dependency of `mkt.databases`, so the former package needs to be installed first. The latest version of these packages can be installed directly from Github via `pip` using the following:
```
pip install git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/schema
pip install git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/databases
```

Please note that the `ClustalOmegaAligner` in the `aligners` module requires a local installation of [Clustal Omega](http://www.clustal.org/omega/). By default, we assume the local executable is `/usr/local/bin/clustalo` though a different path can be specified with the `path_bin` argument.

The database module allows users to query relevant database APIs to extract clinically relevant mutational data and protein annotations from various sources. The `mkt.databases` package contains the following modules:
| Module                                            | Description                                                                    |
| :-:                                               | :-                                                                             |
| [`KinHub`](http://www.kinhub.org/)                | Curated list of human kinases and corresponding information                    |
| [UniProt](https://www.uniprot.org/)               | Obtain canonical protein sequence information                                  |
| [Pfam](https://www.ebi.ac.uk/interpro/entry/pfam) | Annotate protein domains                                                       |
| [HGNC](https://www.genenames.org/)                | Standardize gene naming conventions                                            |
| [KLIFS](https://klifs.net/)                       | Kinase-ligand interaction annotations                                          |
| [cBioPortal](https://www.cbioportal.org/)         | Multi-institutional repository of sequencing data for cancer genomics          |
| [KinCore](http://dunbrack.fccc.edu/kincore/home)  | Harmonized sequence and structure kinase representations from the Dunbrack lab |

## CLI scripts

### `generate_kinaseinfo_objects`

Use this script to query API and scrapers and harmonize into gzipped `KinaseInfo` objects for `mkt.schema` package data:
```
generate_kinaseinfo_objects \
    --pathObjects <PATH_TO_SAVE_GZIP_KINASEINFO_OBJECTS> \
    --pathReports <PATH_TO_SAVE_REPORT_PLOTS>
```

### `extract_cbioportal_missense_kinases`

Use this script to generate a CSV file containing all missense kinase mutations for a given `studyId` within the provided `cbioportalInstance`:
```
extract_cbioportal_missense_kinases \
    --cbioportalInstance    # if not provided, default to www.cbioportal.org
    --cbioportalToken       # only if using private instance; CBIO_PASSWORD=$(cat '<PATH_TO_FILE>/cbioportal_data_access_token.txt' | sed 's/^token: //')
    --studyId               # required study ID within given instance; if not provided logger will return options in instance
    --cache                 # path to requests cache; otherwise will use root repo or current working directory if not Github repo
    --outputPath            # path to save extracted missense kinase data; otherwise will use root repo or current working directory if not Github repo
    --generateHeatmap       # use this flag to generate heatmap figure (store True)
    --dictHeatmap           # pass heatmap dict as a string for ast.literal_eval
```
