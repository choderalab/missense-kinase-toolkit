#!/usr/bin/env bash
# Usage: ./create_env.sh # use GPU

ENV_NAME="boltz2"

source ~/.bashrc

# check if mamba is installed
if command -v mamba &> /dev/null; then  
    echo "mamba is installed, proceeding with environment creation..."
else
    echo "mamba is not installed. Please install mamba first."
    exit 1
fi

# check if boltz2 env already exists
if conda env list | grep -q ${ENV_NAME}; then
    echo "${ENV_NAME} environment already exists. Please remove it first."
    exit 1
else
    echo "${ENV_NAME} environment does not exist, proceeding with creation..."
fi

# check if GPU is available
if command -v nvidia-smi &> /dev/null; then
  echo "NVIDIA GPU detected and nvidia-smi is available, proceeding with environment creation..."
  nvidia-smi
else
  echo "NVIDIA GPU or nvidia-smi not found. Please install with a GPU available."
fi

mamba create -n ${ENV_NAME} python=3.12
source ~/.bashrc
conda activate ${ENV_NAME}

if [ ! -f .env ]; then
    echo ".env file not found, using default for cache."
else
    source .env
    if [ ! -d "${BOLTZ_CACHE}" ]; then
        echo "Creating BOLTZ_CACHE directory at ${BOLTZ_CACHE}"
        mkdir -p "${BOLTZ_CACHE}"
    else
        echo "BOLTZ_CACHE directory already exists at ${BOLTZ_CACHE}"
    fi
    conda env config vars set BOLTZ_CACHE=${BOLTZ_CACHE}
fi

pip install boltz -U

if [ $? -eq 0 ]; then
    echo "Environment boltz2 created and activated successfully."
    echo "To activate the environment, use: conda activate boltz2"
    mamba deactivate
else
    echo "pip install failed."
    exit 1
fi
