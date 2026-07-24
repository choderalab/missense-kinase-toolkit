---
name: docs
description: >-
  missense-kinase-toolkit Sphinx docs; extends my central `docs` skill. Use when
  adding/removing public API, fixing docstrings, or previewing the docs locally
  (docs/generated stubs are gitignored and regenerated at build time, so RTD picks
  up new API automatically — no committed docs change needed).
---

# missense-kinase-toolkit documentation

## Baseline — fetch first

Apply my canonical `docs` conventions (MolSSI cookiecutter-cms Sphinx layout,
autosummary + autodoc + napoleon for NumPy docstrings, recursive custom
templates, `make clean && make html`, the never-hand-edit-`generated/` rule,
Read the Docs) before the repo-specific notes below. WebFetch and follow:

https://raw.githubusercontent.com/jessicaw9910/skills/main/.claude/skills/docs/SKILL.md

If the fetch fails (no network / non-200), **tell me the central `docs` skill
could not be retrieved** and confirm how to proceed — do not silently skip the
baseline.

## Repo-specific additions

- Docs live in `docs/` at the repo root. `docs/api.rst` drives autosummary and
  lists the two top-level packages: **`mkt.schema`** and **`mkt.databases`**
  (recursion covers the submodules/classes). Add any new top-level package
  there.
- **Build it with the two scripts in `docs/` (not the conda `docs` env):**
  - `docs/create_docs_venv.sh [py_version]` — creates an isolated venv at
    `docs/.venv` (gitignored), installs `sphinx` + `sphinx-rtd-theme`, then
    editable-installs `mkt-schema` then `mkt-databases` (databases depends on
    schema). Uses `uv` if present, else `python -m venv` + pip. Default Python
    3.12 (mkt requires `>=3.9,<3.13`).
  - `docs/build_docs.sh` — activates `docs/.venv`, removes `docs/_build` and the
    untracked `docs/generated` stubs, sets `SPHINX_BUILD=1`, then runs
    `python -m sphinx -b html . _build/html` and opens the result.
- Why a venv and not the conda `docs` env: the conda `docs` env is **py3.13**,
  outside mkt's supported range, so `mkt.databases` can't be installed there.
  Both `mkt.schema` and `mkt.databases` are imported in `docs/conf.py`, so
  autodoc needs them importable.
- **OneDrive gotcha:** this repo lives under OneDrive, which silently rewrites a
  venv's interpreter **symlinks into plain text files** — the old root-level
  `docs_env/` died this way (`bin/python` became a 10-byte text file, so it ran
  but produced no output). If `docs/.venv` ever goes quiet/empty, just rerun
  `docs/create_docs_venv.sh` to recreate it.
- `docs/generated/*.rst` are untracked autosummary output; `build_docs.sh`
  deletes and regenerates them, so removed/renamed API drops out automatically.
  Never hand-edit them.
- **Adding/removing public API needs no committed docs change.** `docs/generated/`
  is gitignored and `docs/conf.py` sets `autosummary_generate = True`, so RTD
  regenerates every stub from source on each build. `autodoc_default_options` uses
  `members: True` (and `undoc-members: True`), so a new method on an
  already-documented class (e.g. a new `KinaseInfo` method) renders automatically
  once merged — only its docstring matters. Building/regenerating stubs locally is
  for **preview only**, never a prerequisite for the rendered RTD output. Reach for
  autosummary edits (`docs/api.rst`) only when adding a whole new top-level package.
- Custom templates: `docs/_templates/custom-{module,class}-template.rst`.
- Read the Docs config is `readthedocs.yml` at the **repo root** (not
  `.readthedocs.yaml`); RTD installs doc deps from `docs/requirements.yaml`.
