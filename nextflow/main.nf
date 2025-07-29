#!/usr/bin/env nextflow

/*
========================================================================================
    Boltz-2 Kinase NextFlow Pipeline for Chodera/Tansey Labs
========================================================================================
    Github   :  https://github.com/choderalab/missense-kinase-toolkit
    Contact  :  Jess White
----------------------------------------------------------------------------------------
*/

log.info """\
        Boltz-2 Kinase Pipeline
        ======================================
        fileName     : ${params.fileName}
        outputDir    : ${params.outputDir}

        envBoltz     : ${params.envBoltz}
        envModeling  : ${params.envModeling}
        """
        .stripIndent()

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { CREATE_YAML       } from './modules/boltz2/main.nf'   addParams(OUTPUT: "${params.outputDir}/outputs/yaml")
include { RUN_BOLTZ2        } from './modules/boltz2/main.nf'   addParams(OUTPUT: "${params.outputDir}/outputs/boltz_results")

include { WRITE_CSV_ROW     } from './modules/utils/main.nf'
include { COMBINE_CSV_ROWS  } from './modules/utils/main.nf'   addParams(OUTPUT: "${params.outputDir}/outputs")

/*
========================================================================================
    Define the workflow
========================================================================================
*/

workflow BOLTZ_CSV {
    Channel.fromPath("${params.fileName}", checkIfExists: true)
        .splitCsv(header: true)
        .map { row ->
            [
                UUID.randomUUID().toString(),
                row.smiles,
                row.kinase_name,
                row.y,
                row.klifs,
                row.kincore_kd,
                row.group_consensus,
                row.source,
                row.'z-score'
            ]
        }
        .set { csv_tuples }

    CREATE_YAML( csv_tuples )

    RUN_BOLTZ2( CREATE_YAML.out.yaml )

    csv_tuples
        .join(RUN_BOLTZ2.out.all_results, by: 0)
        .set { joined_results }

    WRITE_CSV_ROW( joined_results )

    WRITE_CSV_ROW.out.csv_row
        .collect()
        .set { all_csv_rows }

    COMBINE_CSV_ROWS( all_csv_rows )
}
