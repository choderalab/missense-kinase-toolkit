---
name: tests
description: >-
  missense-kinase-toolkit-specific testing conventions; extends my central
  `tests` skill. Use when adding or running tests, or debugging test failures.
---

# Testing in missense-kinase-toolkit

## Baseline — fetch first

Apply my canonical `tests` conventions (pytest layout, the `network` marker,
token-gated skips, session/module fixtures, xdist safety, graceful API-failure
skips) before the repo-specific notes below. WebFetch and follow:

https://raw.githubusercontent.com/jessicaw9910/skills/main/.claude/skills/tests/SKILL.md

If the fetch fails (no network / non-200), **tell me the central `tests` skill
could not be retrieved** and confirm how to proceed — do not silently skip the
baseline.

## Repo-specific additions

### Locations

- `missense_kinase_toolkit/schema/mkt/schema/tests/` — `test_<concern>.py`
  (imports, deserialize, adjudicate, serde, utils) with shared fixtures in
  `conftest.py`: session-scoped read-only `dict_kinase` and a single-object
  `mutable_kinase(hgnc_name)` factory (deep-copies one entry, not the full dict).
- `missense_kinase_toolkit/databases/mkt/databases/tests/` — one
  `test_<module>.py` per source module, shared fixtures in `conftest.py`.
- No tests for `ml/`, `experiments/`, or `app/`.

### Running

Use the project venv: `missense_kinase_toolkit/VE/bin/python -m pytest ...`.

```bash
pytest -v --cov-report=xml --color=yes --cov=mkt.schema \
    missense_kinase_toolkit/schema/mkt/schema/tests/
pytest -v --cov-report=xml --color=yes --cov=mkt.databases \
    missense_kinase_toolkit/databases/mkt/databases/tests/
```

The databases and schema suites run under xdist in CI (`-n 2 --dist loadfile`);
keep them parallel-safe locally too (guard shared on-disk work with
`filelock.FileLock` — see `kincore_harmonized_dict` — or write to per-test
`tmp_path`, as `schema/.../test_serde.py` does).

### Local venv (`VE/`)

Created by `bin/create_venv.sh` (`--help` for usage;
`--python X.Y` or a 3.9–3.12 prompt):

- Sub-package flags `--[no-]schema`, `--[no-]databases`, `--[no-]app` (default
  on) and `--[no-]ml` (default off). Selected sub-packages are installed
  **editable** in one `uv pip install` (pip fallback) with the positional
  `EXTRA` args as extras (default: every extra each defines, i.e. `[dev,test]`);
  `--app` adds `app/requirements.txt` minus its git-pinned mkt lines. Schema is
  required by every other selection (it is not on PyPI).
- `uv venv --seed`, so `VE/bin/python -m pip` works.
- Appends `.env` vars to `VE/bin/activate`; installs the pre-commit hook if
  `pre-commit` is on PATH.
- `--overrides-only` skips creation and re-applies the selected sub-packages'
  editable installs into the existing `VE/` (after anything that replaced them
  with non-editable copies).

Gotchas:

- Both `[test]` extras now carry `pytest-xdist`, so `-n` works in a fresh `VE/`.
- `pre-commit` is usually not on PATH but runs as a git commit hook, so
  black/isort/flake8/pyupgrade fire on `git commit`. To run manually:
  `VE/bin/black`, `VE/bin/flake8 --max-line-length=88 --extend-ignore=E203,E501`,
  `uvx isort --profile black` (repo isort has no `known_first_party`, so `mkt.*`
  imports group with third-party).

### Gating and config

- Token gating example (OncoKB):

  ```python
  pytestmark = pytest.mark.skipif(
      maybe_get_oncokb_token() is None,
      reason="OncoKB API token not set (ONCOKB_TOKEN); skipping live tests",
  )
  ```
- Set config via the `mkt.databases.config` setters in fixtures
  (`set_output_dir(".")`, `set_cbioportal_instance(...)`), not `os.environ`.
