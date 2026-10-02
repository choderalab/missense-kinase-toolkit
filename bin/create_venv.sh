#!/usr/bin/env bash
# Usage: ./bin/create_venv.sh [--overrides-only] [--python X.Y]
#                             [--[no-]schema] [--[no-]databases] [--[no-]ml]
#                             [--[no-]app] [EXTRA ...]
#
# Creates the project virtual environment (missense_kinase_toolkit/VE/) with
# editable installs of the selected mono-repo sub-packages and installs the
# pre-commit hook. EXTRA args pick pyproject optional-dependency groups applied
# to every selected sub-package (default: all extras each one defines).
#
# Sub-package selection (positive/negative flag pairs; later flags win):
#   --schema / --no-schema        mkt-schema          (default: on)
#   --databases / --no-databases  mkt-databases       (default: on)
#   --ml / --no-ml                mkt-ml              (default: off)
#   --app / --no-app              Streamlit app deps  (default: on)
# databases, ml, and app all depend on mkt-schema, which is not on PyPI, so
# schema is required whenever any of them is selected.
#
# --python X.Y: interpreter to use (3.9-3.12); prompted for if omitted.
#
# --overrides-only: skip venv creation and only re-apply the editable installs
# of the selected sub-packages into the existing VE/ (e.g. after a manual
# install that replaced them with non-editable copies).
#
# - If `uv` is installed, uses `uv venv --seed` + a single `uv pip install`, so
#   the local schema checkout satisfies the other sub-packages' mkt.schema dep.
# - Otherwise falls back to `python3 -m venv` + `pip install`.
#
# In either case, the app requirements are installed minus their git pins of
# mkt-schema/mkt-databases, so the editable local checkouts are kept.

set -euo pipefail

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

VENV_DIR="${VENV_DIR:-VE}"
APP_REQUIREMENTS="app/requirements.txt"

# parse args: flags + positional pyproject extras (none => all)
OVERRIDES_ONLY=0
WITH_SCHEMA=1
WITH_DATABASES=1
WITH_ML=0
WITH_APP=1
py_version=""
EXTRAS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --overrides-only) OVERRIDES_ONLY=1 ;;
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
    -*) echo "unknown option: $1" >&2; exit 1 ;;
    *) EXTRAS+=("$1") ;;
  esac
  shift
done

# work from the mono-repo package dir (where VE/, .env, and the sub-packages
# live) regardless of the caller's cwd
cd "$(dirname "${BASH_SOURCE[0]}")/../missense_kinase_toolkit"

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

# selected sub-package dirs, schema first (the others declare mkt.schema)
SUBPACKAGES=()
[ "$WITH_SCHEMA" = "1" ] && SUBPACKAGES+=(schema)
[ "$WITH_DATABASES" = "1" ] && SUBPACKAGES+=(databases)
[ "$WITH_ML" = "1" ] && SUBPACKAGES+=(ml)

# "[a,b]" suffix for a sub-package: the requested extras, else all it defines
extras_suffix() {
  local pkg_dir="$1"
  local extras_csv
  if [ "${#EXTRAS[@]}" -gt 0 ]; then
    extras_csv=$(IFS=,; echo "${EXTRAS[*]}")
  else
    extras_csv=$(sed -n '/^\[project.optional-dependencies\]/,/^\[/p' "$pkg_dir/pyproject.toml" \
      | grep -oE '^[A-Za-z0-9_-]+ *=' | tr -d ' =' | paste -sd, -)
  fi
  [ -n "$extras_csv" ] && echo "[$extras_csv]"
  return 0
}

# --overrides-only: re-apply editable installs into the existing venv, then stop
if [ "$OVERRIDES_ONLY" = "1" ]; then
  if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -d "$VENV_DIR" ]; then
      # shellcheck disable=SC1091
      source "$VENV_DIR/bin/activate"
      echo "activated $VENV_DIR/ for this script"
    else
      echo "error: no virtualenv active and no $VENV_DIR/ directory" >&2
      exit 1
    fi
  fi
  override_args=()
  for pkg in "${SUBPACKAGES[@]}"; do
    override_args+=(-e "./$pkg")
  done
  if [ "${#override_args[@]}" -eq 0 ]; then
    echo "no sub-packages selected for editable overrides (app has none)"
    exit 0
  fi
  echo "installing editable: ${SUBPACKAGES[*]}"
  if [ "$USE_UV" = "1" ]; then
    uv pip install "${override_args[@]}"
  else
    python3 -m pip install "${override_args[@]}"
  fi
  echo ""
  echo "done."
  exit 0
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

# install deps: one resolve so the local schema satisfies its dependents
install_args=()
selected=()
for pkg in "${SUBPACKAGES[@]}"; do
  suffix=$(extras_suffix "$pkg")
  install_args+=(-e "./${pkg}${suffix}")
  selected+=("${pkg}${suffix}")
done

# app requirements pin mkt-schema/mkt-databases to git main for Streamlit
# Cloud; drop those lines so the editable local checkouts are used instead
if [ "$WITH_APP" = "1" ]; then
  app_reqs=$(mktemp)
  trap 'rm -f "$app_reqs"' EXIT
  grep -v "subdirectory=missense_kinase_toolkit/" "$APP_REQUIREMENTS" > "$app_reqs"
  install_args+=(-r "$app_reqs")
  selected+=(app)
fi

echo "installing: ${selected[*]}"
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
    echo "# Load project environment variables"
    cat .env
  } >> "$VENV_DIR/bin/activate"
else
  echo ".env not found, skipping append to $VENV_DIR/bin/activate"
fi

echo ""
# absolute path, since the script cd'd away from the caller's cwd
echo "done. activate with: source $(cd "$VENV_DIR" && pwd)/bin/activate"
