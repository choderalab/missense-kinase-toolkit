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