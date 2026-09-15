#!/usr/bin/env bash
# Usage: ./editable_overrides.sh
#
# Re-applies editable installs of mkt-schema and mkt-databases from the
# local sub-package directories (schema/ and databases/). Use this after
# running `uv sync` manually, which reinstalls those packages from the
# lockfile and undoes any prior editable install.
#
# The venv VE/ must already exist; this script does not create or activate
# it — it just installs into whichever environment is active.

set -euo pipefail

SCHEMA_DIR="schema"
DATABASES_DIR="databases"

# load .env so env vars are visible
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# pick toolchain (matches create_venv.sh behavior)
if command -v uv >/dev/null 2>&1; then
  USE_UV=1
else
  USE_UV=0
fi

# warn if no venv is active — installing into the system Python is almost
# certainly not what the user wants
if [ -z "${VIRTUAL_ENV:-}" ]; then
  if [ -d "VE" ]; then
    # shellcheck disable=SC1091
    source VE/bin/activate
    echo "activated VE/ for this script"
  else
    echo "warning: no virtualenv active and no VE/ directory; aborting" >&2
    exit 1
  fi
fi

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

echo ""
echo "done."
