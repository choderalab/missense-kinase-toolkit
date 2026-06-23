#!/usr/bin/env bash
# Usage: docs/create_docs_venv.sh [python_version]   (default: 3.12)
#
# Creates an isolated virtual environment for building the Sphinx docs at
# docs/.venv (gitignored) and installs:
#   - the doc toolchain: sphinx + sphinx-rtd-theme
#   - editable installs of mkt-schema then mkt-databases (order matters:
#     databases depends on schema), which autodoc imports at build time.
#
# Uses uv if available, otherwise falls back to python3 -m venv + pip.
# Build the docs afterwards with docs/build_docs.sh.
#
# This replaces the old root-level docs_env/ venv (which OneDrive corrupted by
# turning its interpreter symlinks into plain text files).
set -euo pipefail

# resolve paths from this script's location so it runs from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
SCHEMA_DIR="$REPO_ROOT/missense_kinase_toolkit/schema"
DATABASES_DIR="$REPO_ROOT/missense_kinase_toolkit/databases"

# mkt requires Python >=3.9,<3.13
PY_VERSION="${1:-3.12}"
case "$PY_VERSION" in
  3.9|3.10|3.11|3.12) ;;
  *)
    echo "error: unsupported Python '$PY_VERSION' (mkt requires >=3.9,<3.13)" >&2
    exit 1
    ;;
esac

if command -v uv >/dev/null 2>&1; then
  USE_UV=1
  echo "uv detected ($(uv --version)); using uv"
else
  USE_UV=0
  echo "uv not found; falling back to python3 -m venv + pip"
fi

# prompt before clobbering an existing venv
if [ -d "$VENV_DIR" ]; then
  read -r -p "$VENV_DIR exists. Delete and recreate? [y/N] " reply
  case "$reply" in
    [yY]|[yY][eE][sS]) echo "removing $VENV_DIR"; rm -rf "$VENV_DIR" ;;
    *) echo "aborting; existing venv left in place"; exit 1 ;;
  esac
fi

# create the venv and resolve the pip install command
if [ "$USE_UV" = "1" ]; then
  uv venv "$VENV_DIR" --python "$PY_VERSION"
  pip_install() { uv pip install --python "$VENV_DIR/bin/python" "$@"; }
else
  if command -v "python${PY_VERSION}" >/dev/null 2>&1; then
    "python${PY_VERSION}" -m venv "$VENV_DIR"
  else
    python3 -m venv "$VENV_DIR"
  fi
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  pip_install() { "$VENV_DIR/bin/python" -m pip install "$@"; }
fi

# doc toolchain
pip_install sphinx sphinx-rtd-theme

# editable sub-packages (schema first since databases depends on it)
pip_install -e "$SCHEMA_DIR"
pip_install -e "$DATABASES_DIR"

echo ""
echo "done. build the docs with: docs/build_docs.sh"
