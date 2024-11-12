#!/bin/bash
#SBATCH --partition=componc_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=7G
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=/data1/tanseyw/projects/whitej/esm_km_atp/src/stdout/%x_%j.out
#SBATCH --error=/data1/tanseyw/projects/whitej/esm_km_atp/src/stderr/%x_%j.err

# take command line arguments
if [ "$#" -ne 3 ]; then
    echo "You must enter exactly 3 command line arguments: <MODEL> <COL_SEQ> <RUN_NAME>"
    exit
fi

MODEL=$1
COL_SEQ=$2
RUN_NAME=$3

source ~/.bashrc
mamba activate hf_torch

python main.py \
    --model ${MODEL} \
    --columnSeq ${COL_SEQ} \
    --wandbRun ${RUN_NAME}