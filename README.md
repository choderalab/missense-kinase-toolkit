missense-kinase-toolkit (mkt)
==============================
[//]: # (Badges)
[![codecov](https://codecov.io/gh/choderalab/missense-kinase-toolkit/branch/main/graph/badge.svg)](https://codecov.io/gh/choderalab/missense-kinase-toolkit/branch/main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/choderalab/missense-kinase-toolkit/main.svg)](https://results.pre-commit.ci/latest/github/choderalab/missense-kinase-toolkit/main)
[![Documentation Status](https://readthedocs.org/projects/missense-kinase-toolkit/badge/?version=latest)](https://missense-kinase-toolkit.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/758694808.svg)](https://zenodo.org/doi/10.5281/zenodo.11495030)<br />
[![schema-ci](https://github.com/choderalab/missense-kinase-toolkit/actions/workflows/schema-ci.yaml/badge.svg)](https://github.com/choderalab/missense-kinase-toolkit/actions/workflows/schema-ci.yaml)
[![databases-ci](https://github.com/choderalab/missense-kinase-toolkit/actions/workflows/databases-ci.yaml/badge.svg)](https://github.com/choderalab/missense-kinase-toolkit/actions/workflows/databases-ci.yaml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mkt-app.streamlit.app)

## Intro

`mkt` is a Python package to generate sequence and structure-based representations for human kinase property prediction. While our application uses this data to predict the impact of clinically observed missense mutations on human kinase activity, we note that many of these tools can be used more extensively to characterize wild-type human kinases and any mutant forms. The `databases` sub-package can be used to query the APIs of a variety of protein resources that are not exclusive to either kinases or humans, including UniProt, Pfam, and cBioPortal.

Additional documentation can be found [here](https://missense-kinase-toolkit.readthedocs.io/en/latest/).

## Getting started

`mkt` is structured as a monorepo with sub-packages and directories described below for specific tasks.

| Subpackages   | Description                                                                                                                         |
| :-:           | :-                                                                                                                                  |
| `app`         | [Streamlit app](https://mkt-app.streamlit.app/) to visualize data contained in harmonized `Pydantic` models                         |
| `schema`      | Harmonized and pre-processed sequence and structure data along with `Pydantic` models to load, query, and validate this data        |
| `databases`   | Package containing API clients and scrapers to collect and harmonize kinase data from various sources and generate `schema` objects |
| `ml`          | **In-progress** package to build machine learning models to predict kinase properties                                               |
| `experiments` | **In-progress** package to analyze experimental results for project                                                                 |

Sub-packages can be installed directly from Github via `pip` using the following:
```
pip install git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/<sub-package directory>
```

### Copyright

Copyright (c) 2024, Jess White

#### Acknowledgements

We would like to express gratitude to the creators of the following resources on which we heavily rely:
+ [UniProt](https://www.uniprot.org/)
+ [Pfam](https://www.ebi.ac.uk/interpro/entry/pfam)
+ [cBioPortal](https://www.cbioportal.org/)
+ [KinHub](http://www.kinhub.org/)
+ [KLIFS](https://klifs.net/)
+ [KinCore](http://dunbrack.fccc.edu/kincore/home)
+ [KinoML](https://github.com/openkinome/kinoml)

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
