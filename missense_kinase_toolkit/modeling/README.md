# mkt.ml

This package is meant to facilitate modeling experiments using kinase representations derived from `mkt.databases` and consolidated in `mkt.schema`.

## Setting up environment(s)

In the `missense_kinase_toolkit/modeling` sub-directory, use the following:

## `boltz2`

This will create a clean, GPU-compatible environment with `boltz2` installed:
```
./create_env.sh
```

## `mkt_modeling`

This creates an environment with the necessary modeling packages (e.g., `PyMol`):
```
conda env create -f environment.yaml
```
