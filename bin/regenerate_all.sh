#!/usr/bin/env/ bash
# usage: ./regenerate_all.sh <config_file.yaml>

# must have 1 arg
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file.yaml>"
    exit 1
fi

CONFIG_FILE=$1

# check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file $CONFIG_FILE does not exist."
    exit 1
fi

# make sure config if yaml
if [[ "$CONFIG_FILE" != *.yaml ]]; then
    echo "Config file $CONFIG_FILE is not a yaml file."
    exit 1
fi

# source the virtual environment
PATH_TO_VENV="missense_kinase_toolkit/VE/bin/activate"
if [ -f "$PATH_TO_VENV" ]; then
    source "$PATH_TO_VENV"
else
    # run create_venv.sh to create the virtual environment
    echo "Virtual environment not found. Need to run ../create_venv.sh to generate it."
    #TODO how to run nested scripts in bash?
    ../create_venv.sh
    source "$PATH_TO_VENV"
fi

# source .env file if it exists
if [ -f ".env" ]; then
    export $(cat .env | xargs)
fi

# generate kinase_info
python -m generate_kinaseinfo_objects

# generate the kinase plots
python -m plot_dict_kinase \
    --config "$CONFIG_FILE"

# generate the dataset CSV files
python -m generate_dataset_csv_files

# run the generation of the plots
python -m plot_dataset_data \
    --config "$CONFIG_FILE"
