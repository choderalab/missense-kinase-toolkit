---
name: ci
description: >-
  missense-kinase-toolkit-specific CI/CD layout; extends my central `ci` skill.
  Use when editing .github/workflows, adding a sub-package, changing the test
  env, or debugging CI failures.
---

# missense-kinase-toolkit CI/CD

## Baseline — fetch first

Apply my canonical `ci` conventions (path-filtered per-package workflows,
uv test envs, OS/Python matrix, Codecov flags, pre-commit hook set) before the
repo-specific notes below. WebFetch and follow:

https://raw.githubusercontent.com/jessicaw9910/skills/main/.claude/skills/ci/SKILL.md

If the fetch fails (no network / non-200), **tell me the central `ci` skill
could not be retrieved** and confirm how to proceed — do not silently skip the
baseline.

## Repo-specific additions

### Workflows (`.github/workflows/`)

One workflow per setuptools sub-package, path-filtered (plus a weekly
`cron: "0 0 * * 0"`):

- `schema-ci.yaml` — paths `missense_kinase_toolkit/schema/**`, coverage flag
  `schema`, installs `schema[test]` only.
- `databases-ci.yaml` — paths `missense_kinase_toolkit/databases/**`, coverage
  flag `databases`, installs `schema` + `databases[test,pymol]` in one
  `uv pip install` (the local schema path satisfies databases' `mkt.schema`
  dep). The `pymol` extra is `pymol-open-source-whl` with a
  `sys_platform != 'win32'` marker, because that wheel can't start a session on
  Windows; there the pymol SASA test importorskips. A separate `pymol-windows`
  job (micromamba env of `python=3.11` + conda-forge `pymol-open-source`, deps
  via `uv pip`) runs only `test_sasa.py` to cover the documented Windows route,
  and fails outright if `pymol2` can't start.

Both: matrix `os: [macOS, ubuntu, windows] × python: ["3.10", "3.11"]`,
**`astral-sh/setup-uv`** (cached, `activate-environment: true`), pytest with
`-n 2 --dist loadfile --durations=20`, then Codecov upload with the per-package
`flags`. `app-ci.yaml` is a Linux/Py3.12 Streamlit smoke test mirroring
Streamlit Cloud (also uv). There is **no** workflow for `ml/` or `experiments/`.

The old micromamba step is left commented out in each workflow for reference;
`devtools/conda-envs/test_env.yaml` is kept for local conda users only and is
**not** what CI installs. Every `uv pip install` is wrapped in a `retry` shell
function (3 attempts, 20s/40s backoff) so a transient network failure doesn't
trip the matrix's `fail-fast`.

Test-speed notes:
- schema-ci runs `-n 0` on macOS: the full `KinaseInfo` dict is ~7 GB per xdist
  worker and macOS runners have 7 GB RAM.
- `test_chembl.py` probes ChEMBL's status endpoint once and skips the module if
  it's down; `kincore_harmonized_dict` stubs `MMCIF2Dict` (CIF parsing is
  covered by the gemmi-based `TestKinCoReCIFIntegrity`); databases-ci caches
  `data/AF2_Active_Models_v2.zip` via `actions/cache`.

### Adding / changing

- New runtime dependency → add it to the sub-package's `pyproject.toml`
  `dependencies`; new test-only dependency → its `[test]` extra. CI installs
  from these, so nothing else needs updating (optionally mirror it in
  `test_env.yaml` for local conda users).
- Every dependency must have PyPI wheels (or be pure Python) on all three OSes.
  Check before adding with
  `uv pip compile --python-platform x86_64-pc-windows-msvc --only-binary :all: ...`.
- The workflows' `paths` filters don't include `.github/workflows/**`, so a
  workflow-only change doesn't trigger CI on its own — touch the sub-package to
  exercise it. On a PR, though, the filter is applied to the whole PR diff, so
  every push reruns every workflow the PR touches.

### Pre-commit

`.pre-commit-config.yaml` is scoped to `^missense_kinase_toolkit` (excluding
`KinaseInfo/`): `black`, `isort` (profile black), `flake8` (max-line-length 88,
ignore E203/E501), `pyupgrade --py39-plus`, plus whitespace/yaml hooks.
