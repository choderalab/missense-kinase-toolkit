process SAVE_RESULTS_CSV {
    label 'cpu'
    
    conda "${params.envModeling}"
    publishDir "${params.outputDir}/results", mode: 'copy', overwrite: true
    
    input:
    tuple val(uuid),
          val(smiles),
          val(kinase_name),
          val(y),
          val(klifs),
          val(kincore_kd),
          val(group_consensus),
          val(source),
          val(z_score),
          path(boltz_results)
    
    output:
    path "results_with_uuid.csv", emit: results_csv
    
    script:
    """
    # Create CSV header if it doesn't exist
    if [ ! -f results_with_uuid.csv ]; then
        echo "uuid,smiles,kinase_name,y,klifs,kincore_kd,group_consensus,source,z-score,boltz_output_dir" > results_with_uuid.csv
    fi
    
    # Append this row to the CSV
    echo "${uuid},${smiles},${kinase_name},${y},${klifs},${kincore_kd},${group_consensus},${source},${z_score},${boltz_results}" >> results_with_uuid.csv
    """
}