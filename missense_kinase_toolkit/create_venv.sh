#!/usr/bin/env bash
# Usage: ./create_venv.sh
#
# Creates the project virtual environment and installs mkt-schema then
# mkt-databases as editable sub-packages (order matters: databases depends
# on schema).
#
# - If `uv` is installed AND uv.lock exists, uses `uv sync` for reproducible
#   installs, then overwrites the two sub-packages with editable installs
#   so local changes are reflected immediately.
# - If `uv` is installed but no uv.lock, uses `uv venv` + `uv pip install -e`.
# - Otherwise falls back to `python3 -m venv` + `pip install -e`.
#
# The script prompts for a Python version (3.9-3.12) and before deleting an
# existing venv. On completion, .env vars are appended to the activate script.

set -euo pipefail

VENV_DIR="VE"
SCHEMA_DIR="schema"
DATABASES_DIR="databases"

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

# prompt for python version
read -r -p "Python version to use [3.9/3.10/3.11/3.12] (default: 3.12): " py_version
py_version="${py_version:-3.12}"

# validate the chosen version
case "$py_version" in
  3.9|3.10|3.11|3.12) ;;
  *)
    echo "error: unsupported Python version '$py_version' (must be 3.9-3.12)" >&2
    exit 1
    ;;
esac

# resolve the python executable
if command -v "python${py_version}" >/dev/null 2>&1; then
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
echo "using $PYTHON_EXE ($($PYTHON_EXE --version))"

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

# create venv and install deps
if [ "$USE_UV" = "1" ] && [ -f uv.lock ]; then
  export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
  uv sync --python "$PYTHON_EXE" --all-extras
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [ "$USE_UV" = "1" ]; then
  uv venv "$VENV_DIR" --python "$PYTHON_EXE"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  "$PYTHON_EXE" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python3 -m pip install --upgrade pip
fi

# apply editable overrides (post-install so local changes are reflected
# immediately; schema first since databases depends on it)
override_editable() {
  local pkg_path="$1"
  local pkg_name="$2"
  if [ ! -d "$pkg_path" ]; then
    echo "skipping $pkg_name: '$pkg_path' is not a directory"
    return
  fi
  echo "installing $pkg_name editable from $pkg_path"
  if [ "$USE_UV" = "1" ]; then
    uv pip install -e "$pkg_path"
  else
    python3 -m pip install -e "$pkg_path"
  fi
}

override_editable "./$SCHEMA_DIR" "mkt-schema"
override_editable "./$DATABASES_DIR" "mkt-databases"

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
