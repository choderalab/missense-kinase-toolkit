process PROCESS_CBIOPORTAL {
    input:
    tuple val(cbio_cohort), path(out_dir), val(cbio_inst), val(cbio_token), path(request_cache)

    output:
    path("${out_dir}/cbioportal")
    """
    export PYTHONHASHSEED=0
    process_cbioportal \
        --cohort ${cbio_cohort} \
        --outDir ${out_dir} \
        --instance ${cbio_inst} \
        --token ${cbio_token} \
        --requestsCache ${request_cache}
    """
}
