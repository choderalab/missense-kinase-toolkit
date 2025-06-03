# mkt.ml

This package is meant to facilitate machine learning experiments using kinase representations derived from `mkt.databases` and consolidated in `mkt.schema`.

The following drug and protein language models are supported. Theoretically, the `CombinedPoolingModel` in the `mkt.ml.models.pooling` module should be compatible with any pretrained SMILES and amino acid language models with a pooling layer if support is implemented in the `mkt.ml.constants` module (i.e., models added to `DrugModel` and/or `KinaseModel`).

| Model name                 | Type        | Description                 |
| :------------------------: | :---------: | :-------------------------- |
| esm2_t6_8M_UR50D           | ESM2        | 6 layers / 8M parameters    |
| esm2_t12_35M_UR50D         | ESM2        | 12 layers / 35M parameters  |
| esm2_t30_150M_UR50D        | ESM2        | 30 layers / 150M parameters |
| esm2_t33_650M_UR50D        | ESM2        | 33 layers / 650M parameters |
| esm2_t36_3B_UR50D          | ESM2        | 36 layers / 3B parameters   |
| esm2_t48_15B_UR50D         | ESM2        | 48 layers / 15B parameters  |
| DeepChem/ChemBERTa-77M-MTR | ChemBERTa-2 | Multi-task regression       |
| DeepChem/ChemBERTa-77M-MLM | ChemBERTa-2 | Masked language model       |

## Running on cluster

Running an experiment consists of the following steps:

1. Generate a config yaml file (2 examples are shown below).
2. Submit the job via bash script shown below.

### Configs

#### Train-test kinase split

Hold out all tyrosine kinases (TK) and tyrosine kinase-like (TKL) samples in the test set. Replace the following placeholders:
+ <PATH_TO_DATA>: Path where csv file of data resides
+ <WANDB_ENTITY_NAME>: If using, username or team name under which the runs will be logged; else None
+ <WANDB_PROJECT_NAME>: The name of the project under which this run will be logged

```
seed: 42
data:
  type: "PKIS2_Kinase_Split"
  configs:
    filepath: "<PATH_TO_DATA>/pkis_data.csv"
    col_labels: "percent_displacement"
    col_kinase: "klifs"
    col_drug: "Smiles"
    model_drug: "CHEMBERTA_MTR"
    model_kinase: "ESM2_T6_8M"
    col_kinase_split: "kincore_group"
    list_kinase_split: ["TK", "TKL"]
model:
  type: "pooling"
  configs:
    hidden_size: 256
    bool_drug_freeze: False
    bool_kinase_freeze: False
    dropout_rate: 0.1
trainer:
  configs:
    batch_size: 32
    epochs: 100
    learning_rate: 2e-5
    weight_decay: 0.01
    percent_warmup: 0.1
    bool_clip_grad: True
    save_every: 1
    moving_avg_window: 100
    log_interval: 10
    validation_step_interval: 1000
    best_models_to_keep: 5
    entity_name: <WANDB_ENTITY_NAME>
    project_name: <WANDB_PROJECT_NAME>
```

#### Cross-validation

Five-fold cross-validation. Replace the following placeholders:
+ <PATH_TO_DATA>: Path where csv file of data resides
+ <WANDB_ENTITY_NAME>: If using, username or team name under which the runs will be logged; else None
+ <WANDB_PROJECT_NAME>: The name of the project under which this run will be logged

```
seed: 42
data:
  type: "PKIS2_CV_Split"
  configs:
    filepath: "<PATH_TO_DATA>/pkis_data.csv"
    col_labels: "percent_displacement"
    col_kinase: "klifs"
    col_drug: "Smiles"
    model_drug: "CHEMBERTA_MTR"
    model_kinase: "ESM2_T6_8M"
    k_folds: 5
model:
  type: "pooling"
  configs:
    hidden_size: 256
    bool_drug_freeze: False
    bool_kinase_freeze: False
    dropout_rate: 0.1
trainer:
  configs:
    batch_size: 32
    epochs: 100
    learning_rate: 2e-5
    weight_decay: 0.01
    percent_warmup: 0.1
    bool_clip_grad: True
    save_every: 1
    moving_avg_window: 100
    log_interval: 10
    validation_step_interval: 1000
    best_models_to_keep: 5
    entity_name: <WANDB_ENTITY_NAME>
    project_name: <WANDB_PROJECT_NAME>
```

### Run `trainer` via bash script

Note that the default output directory in `run_trainer` is the directory in which this bash script is run.

```
#!/bin/bash
#SBATCH --partition=<PARTITION>
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=run_trainer
#SBATCH --output=<PATH_TO_OUTPUT>/stdout/%x_%j.out
#SBATCH --error=<PATH_TO_OUTPUT>/stderr/%x_%j.err

# Usage: sbatch run_trainer.sh <PATH_TO_CONFIG>

FILE_CONFIG=$1

# error if no argument is given
if [ $# -eq 0 ]; then
    echo "You must enter exactly 1 command line arguments: <PATH_TO_CONFIG>"
    exit 1
fi

if [ ! -f "$FILE_CONFIG" ]; then
    echo "File ${FILE_CONFIG} does not exist. Exiting..."
    exit 1
fi

source ~/.bashrc
mamba activate mkt_ml_plus

run_trainer \
    --config ${FILE_CONFIG} \
    --job_name ${SLURM_JOB_NAME}
```
