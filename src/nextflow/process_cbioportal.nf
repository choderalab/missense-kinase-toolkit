process PROCESS_CBIOPORTAL {
    // tag "$meta.id"
    // label 'process_medium'
    // container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
    //     'docker://jeffquinnmsk/pan_preclinical_etl:latest' :
    //     'docker.io/jeffquinnmsk/pan_preclinical_etl:latest' }"

    input:
    tuple val(meta), path(raw_data), path(studies), path(source_files), path(request_cache), val(study_name)

    output:
    tuple val(meta), path("${prefix}/per_study_results/${study_name}"), emit: etl_results

    """
    export PYTHONHASHSEED=0
    mkdir -p "${prefix}/per_study_results/${study_name}"
    ${cache_flag}process_dataset \
        --data-dir ${raw_data} \
        --output-dir ${prefix}/per_study_results/${study_name} \
        --studies ${studies} \
        --source-files ${source_files} \
        --study-id ${study_name}
    """
}
