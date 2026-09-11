#!/usr/bin/env bash
# Regenerate the mkt data artifacts and/or figures from one shared study config.
#
# usage: ./bin/regenerate_all.sh <config_file.yaml> [--figs-only TASKS] [--only TASKS] [--skip TASKS]
#   tasks: kinaseinfo | conservation | dataset | pymol   (comma-separated; "all" allowed)
#   --figs-only TASKS   these tasks render figures only (no data rebuild); others do a full
#                       regen. bare --figs-only means all. pymol has no data step (always figures).
#   --only TASKS        run only these tasks.
#   --skip TASKS        run everything except these tasks.
#
# Each CLI reads its own task namespace from the shared YAML and writes figures to
#   <output.subdir>/<config-stem>/<task>/
#
# Run from the repo root.

set -uo pipefail

# --- constants (edit here) ---
ALL_TASKS="kinaseinfo conservation dataset pymol"
PATH_TO_VENV="missense_kinase_toolkit/VE/bin/activate"
CREATE_VENV_SCRIPT="missense_kinase_toolkit/create_venv.sh"

CONFIG=""
FIGS_ONLY_TASKS=""
ONLY_TASKS=""
SKIP_TASKS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --figs-only)
            # optional value; bare --figs-only == all
            if [ $# -ge 2 ] && [ "${2#--}" = "$2" ]; then
                FIGS_ONLY_TASKS="$2"
                shift 2
            else
                FIGS_ONLY_TASKS="all"
                shift
            fi
            ;;
        --only)
            ONLY_TASKS="${2:-}"
            shift 2
            ;;
        --skip)
            SKIP_TASKS="${2:-}"
            shift 2
            ;;
        *)
            CONFIG="$1"
            shift
            ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Usage: $0 <config_file.yaml> [--figs-only TASKS] [--only TASKS] [--skip TASKS]"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "Config file $CONFIG does not exist."
    exit 1
fi
if [[ "$CONFIG" != *.yaml ]]; then
    echo "Config file $CONFIG is not a yaml file."
    exit 1
fi

# --- environment ---
if [ -f "$PATH_TO_VENV" ]; then
    # shellcheck disable=SC1090
    source "$PATH_TO_VENV"
else
    echo "Virtual environment not found at $PATH_TO_VENV."
    echo "Create it first:  bash $CREATE_VENV_SCRIPT"
    exit 1
fi
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# return 0 if $1 is in the comma-list $2 (or $2 == "all")
in_csv() {
    [ "$2" = "all" ] && return 0
    case ",$2," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# map a task name to its CLI entry point
cli_for() {
    case "$1" in
        kinaseinfo) echo generate_kinaseinfo_objects ;;
        conservation) echo generate_conservation_data ;;
        dataset) echo generate_dataset_csv_files ;;
        pymol) echo generate_pymol_files ;;
    esac
}

# run a step, reporting failure without aborting the rest of the run
run_step() {
    echo ">>> $*"
    if ! "$@"; then
        echo "!!! step failed: $*"
    fi
}

for task in $ALL_TASKS; do
    # --only wins; otherwise run everything not in --skip
    if [ -n "$ONLY_TASKS" ]; then
        in_csv "$task" "$ONLY_TASKS" || continue
    elif [ -n "$SKIP_TASKS" ] && in_csv "$task" "$SKIP_TASKS"; then
        echo "--- skipping $task ---"
        continue
    fi

    cli=$(cli_for "$task")
    # pymol has no data step; the figs-only flag only applies to the data-bearing tasks
    if [ "$task" != "pymol" ] && in_csv "$task" "$FIGS_ONLY_TASKS"; then
        run_step "$cli" --figs-only --config "$CONFIG"
    else
        run_step "$cli" --config "$CONFIG"
    fi
done

echo "=== regenerate_all.sh complete ==="
