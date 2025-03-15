#!/bin/bash
#SBATCH --partition=componc_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=7G
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --job-name=batch_esm_km_atp
#SBATCH --output=/data1/tanseyw/projects/whitej/esm_km_atp/src/stdout/%x_%j.out
#SBATCH --error=/data1/tanseyw/projects/whitej/esm_km_atp/src/stderr/%x_%j.err

while IFS=, read -r model col_seq run_name
do
    sbatch -J ${run_name} run.sh ${model} ${col_seq} ${run_name}
done < batch_jobs.csv
