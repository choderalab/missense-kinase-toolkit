#!/usr/bin/env bash
# Usage: ./create_venv.sh [--python X.Y] [--[no-]schema] [--[no-]databases]
#                         [--[no-]ml] [--[no-]app]
#
# Creates the project virtual environment with editable installs of the
# selected mono-repo sub-packages, each with its [dev,test] extras.
#
# Sub-package selection (positive/negative flag pairs; later flags win):
#   --schema    / --no-schema     mkt-schema           (default: on)
#   --databases / --no-databases  mkt-databases        (default: on)
#   --ml        / --no-ml         mkt-ml               (default: off)
#   --app       / --no-app        Streamlit app deps   (default: on)
# databases, ml, and app all depend on mkt-schema, which is not on PyPI, so
# schema is required whenever any of them is selected.
#
# --python X.Y picks the interpreter (3.9-3.12); without it you are prompted.
#
# - If `uv` is installed, uses `uv venv` + a single `uv pip install` so the
#   local schema checkout satisfies the other sub-packages' mkt.schema dep.
# - Otherwise falls back to `python3 -m venv` + `pip install -e`.
#
# Prompts before deleting an existing venv. On completion, .env vars are
# appended to the activate script and the pre-commit hook is installed if
# pre-commit is on PATH.

set -euo pipefail

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

VENV_DIR="${VENV_DIR:-VE}"
EXTRAS="[dev,test]"
APP_REQUIREMENTS="app/requirements.txt"

# defaults mirror the long-standing dev venv: schema + databases + app deps
WITH_SCHEMA=1
WITH_DATABASES=1
WITH_ML=0
WITH_APP=1
py_version=""

while [ $# -gt 0 ]; do
  case "$1" in
    --schema) WITH_SCHEMA=1 ;;
    --no-schema) WITH_SCHEMA=0 ;;
    --databases) WITH_DATABASES=1 ;;
    --no-databases) WITH_DATABASES=0 ;;
    --ml) WITH_ML=1 ;;
    --no-ml) WITH_ML=0 ;;
    --app) WITH_APP=1 ;;
    --no-app) WITH_APP=0 ;;
    --python)
      [ $# -ge 2 ] || { echo "error: --python needs a version" >&2; exit 1; }
      py_version="$2"
      shift
      ;;
    --python=*) py_version="${1#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument '$1' (see --help)" >&2; exit 1 ;;
  esac
  shift
done

# run from the mono-repo package root regardless of the caller's cwd
cd "$(dirname "${BASH_SOURCE[0]}")"

# every other sub-package needs the local schema checkout
if [ "$WITH_SCHEMA" = "0" ] && { [ "$WITH_DATABASES" = "1" ] || [ "$WITH_ML" = "1" ] || [ "$WITH_APP" = "1" ]; }; then
  echo "error: --no-schema conflicts with databases/ml/app, which depend on mkt-schema" >&2
  exit 1
fi
if [ "$WITH_SCHEMA$WITH_DATABASES$WITH_ML$WITH_APP" = "0000" ]; then
  echo "error: no sub-packages selected" >&2
  exit 1
fi

# load .env so env vars are visible to this script
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# pick toolchain
if command -v uv >/dev/null 2>&1; then
  USE_UV=1
  echo "uv detected ($(uv --version)); using uv"
else
  USE_UV=0
  echo "uv not found; falling back to python3 -m venv + pip"
fi

# prompt for python version unless given via --python
if [ -z "$py_version" ]; then
  read -r -p "Python version to use [3.9/3.10/3.11/3.12] (default: 3.12): " py_version
  py_version="${py_version:-3.12}"
fi

# validate the chosen version
case "$py_version" in
  3.9|3.10|3.11|3.12) ;;
  *)
    echo "error: unsupported Python version '$py_version' (must be 3.9-3.12)" >&2
    exit 1
    ;;
esac

# resolve the python executable (uv can fetch a managed interpreter itself)
if [ "$USE_UV" = "1" ]; then
  PYTHON_EXE="$py_version"
elif command -v "python${py_version}" >/dev/null 2>&1; then
  PYTHON_EXE="python${py_version}"
elif command -v python3 >/dev/null 2>&1; then
  actual=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  if [ "$actual" != "$py_version" ]; then
    echo "error: python${py_version} not found and python3 is $actual" >&2
    exit 1
  fi
  PYTHON_EXE="python3"
else
  echo "error: no suitable Python interpreter found" >&2
  exit 1
fi

# prompt before deleting existing VE
if [ -d "$VENV_DIR" ]; then
  read -r -p "$VENV_DIR/ exists. Delete and recreate? [y/N] " reply
  case "$reply" in
    [yY]|[yY][eE][sS])
      echo "removing existing $VENV_DIR/"
      rm -rf "$VENV_DIR"
      ;;
    *)
      echo "aborting; existing $VENV_DIR/ left in place"
      exit 1
      ;;
  esac
fi

# create venv
if [ "$USE_UV" = "1" ]; then
  # --seed adds pip so `VE/bin/python -m pip` keeps working
  uv venv "$VENV_DIR" --python "$PYTHON_EXE" --seed
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PIP=(uv pip install)
else
  "$PYTHON_EXE" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python3 -m pip install --upgrade pip
  PIP=(python3 -m pip install)
fi
echo "using $(python --version) in $VENV_DIR/"

# assemble one install so the resolver sees the local schema alongside its
# dependents (schema first; the others declare mkt.schema as a dep)
install_args=()
selected=()
[ "$WITH_SCHEMA" = "1" ] && install_args+=(-e "./schema${EXTRAS}") && selected+=(schema)
[ "$WITH_DATABASES" = "1" ] && install_args+=(-e "./databases${EXTRAS}") && selected+=(databases)
[ "$WITH_ML" = "1" ] && install_args+=(-e "./ml${EXTRAS}") && selected+=(ml)

# app requirements pin mkt-schema/mkt-databases to git main for Streamlit
# Cloud; drop those lines so the editable local checkouts are used instead
if [ "$WITH_APP" = "1" ]; then
  app_reqs=$(mktemp)
  trap 'rm -f "$app_reqs"' EXIT
  grep -v "subdirectory=missense_kinase_toolkit/" "$APP_REQUIREMENTS" > "$app_reqs"
  install_args+=(-r "$app_reqs")
  selected+=(app)
fi

echo "installing: ${selected[*]} (sub-packages editable with extras $EXTRAS)"
"${PIP[@]}" "${install_args[@]}"

# install the pre-commit git hook (idempotent); pre-commit is not a declared
# dep, so only if it is already on PATH
if command -v pre-commit >/dev/null 2>&1; then
  echo "installing pre-commit git hook"
  (cd .. && pre-commit install)
else
  echo "pre-commit not on PATH; skipping hook install"
fi

# append .env to the activate script so vars load on activation
if [ -f .env ]; then
  {
    echo ""
    echo "# load project environment variables"
    cat .env
  } >> "$VENV_DIR/bin/activate"
else
  echo ".env not found, skipping append to $VENV_DIR/bin/activate"
fi

echo ""
echo "done. activate with: source $VENV_DIR/bin/activate"
