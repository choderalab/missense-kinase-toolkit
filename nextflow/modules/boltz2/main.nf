process CREATE_YAML {
    label 'cpu'

    conda "${params.envModeling}"
    publishDir "${params.OUTPUT}", mode: 'copy', overwrite: true

    input:
    tuple val(uuid),
          val(smiles),
          val(kinase_name),
          val(y),
          val(klifs),
          val(kincore_kd),
          val(group_consensus),
          val(source),
          val(z_score)

    output:
    tuple val(uuid), path("${uuid}.yaml"), emit: yaml

    script:
    """
    create_yaml.py \\
        --targetSequence "${kincore_kd}" \\
        --ligandSMILES "${smiles}" \\
        --uuid ${uuid}
    """
}

process RUN_BOLTZ2 {
    label 'gpu'

    conda "${params.envBoltz}"
    publishDir "${params.OUTPUT}", mode: 'copy', overwrite: true

    input:
    tuple val(uuid), path(yaml_file)

    output:
    tuple val(uuid), path("boltz_results_${uuid}"), emit: all_results
    tuple val(uuid), path("boltz_results_${uuid}/predictions/${uuid}/*.{cif,pdb}"), emit: structure

    script:
    def output_format_flag = params.usePDB ? "--output_format pdb" : "--output_format mmcif"
    def msa_server_flag = params.useMSAServer ? "--use_msa_server" : ""
    """
    echo "Running boltz with:"
    echo "YAML file: ${yaml_file}"
    echo "Output dir: boltz_results_${uuid}"
    echo "MSA server flag: ${msa_server_flag}"
    echo "BOLTZ_CACHE: \$BOLTZ_CACHE"

    boltz predict ${yaml_file} \\
        --accelerator gpu \\
        --cache \$BOLTZ_CACHE \\
        ${output_format_flag} \\
        ${msa_server_flag}
    """
}
