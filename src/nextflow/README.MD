# NextFlow workflow to run pipeline

To run: `nextflow run main.nf -params-file params.json`

Generate own `params.json` file using the following parameters:
```
{
    "CBIOPORTAL_COHORT"     : "TODO",
    "OUTPUT_DIR"            : "TODO",
    "CBIOPORTAL_INSTANCE"   : "TODO",
    "CBIOPORTAL_TOKEN"      : "TODO",
    "REQUESTS_CACHE"        : "TODO"
}
```

Below is a description of what each variable should contain. If variable is optional and not in use, do not create any entry in the `json` file.

| Variable             | Optional | Description                                |
| :--------------------| :------: | :----------------------------------------- |
| CBIOPORTAL_COHORT    |    No    | cBioPortal cohort to analyze               |
| OUTPUT_DIR           |    No    | Path to outdir to save data                |
| CBIOPORTAL_INSTANCE  |    Yes   | `cbioportal.org` if none provided          |
| CBIOPORTAL_TOKEN     |    Yes   | Data Access Token if using private instance|
| REQUESTS_CACHE       |    Yes   | Boolean of whether to cache requests data  |
