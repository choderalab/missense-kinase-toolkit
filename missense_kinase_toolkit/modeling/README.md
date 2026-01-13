# mkt.modeling

This package is meant to facilitate modeling experiments.

## Setting up environment(s)

In the `missense_kinase_toolkit/modeling` sub-directory, use the following:

### `boltz2`

This will create a clean, GPU-compatible environment with `boltz2` installed:
```
./create_env.sh
```

### `mkt_modeling`

This creates an environment with the necessary modeling packages (e.g., `PyMol`):
```
cd <REPO_ROOT>/missense-kinase-toolkit/missense_kinase_toolkit/modeling/create_env.sh
conda env create -f environment.yaml
conda activate mkt_modeling
pip install -e .
```
