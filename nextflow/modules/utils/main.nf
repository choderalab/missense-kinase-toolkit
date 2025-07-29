process WRITE_CSV_ROW {
    label 'cpu'
    
    conda "${params.envModeling}"
    
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
    path "${uuid}.csv", emit: csv_row
    
    script:
    """
    echo "${uuid},${smiles},${kinase_name},${y},${klifs},${kincore_kd},${group_consensus},${source},${z_score},${boltz_results}" > ${uuid}.csv
    """
}

process COMBINE_CSV_ROWS {
    label 'cpu'
    
    conda "${params.envModeling}"
    publishDir "${params.OUTPUT}", mode: 'copy', overwrite: true
    
    input:
    path csv_rows, stageAs: "row_*.csv"
    
    output:
    path "results_with_uuid.csv", emit: results_csv
    
    script:
    """
    # Create header
    echo "uuid,smiles,kinase_name,y,klifs,kincore_kd,group_consensus,source,z-score,boltz_output_dir" > results_with_uuid.csv
    
    # Combine all row files
    cat row_*.csv >> results_with_uuid.csv
    """
}