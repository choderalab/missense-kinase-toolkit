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
micromamba env, OS/Python matrix, Codecov flags, pre-commit hook set) before the
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
  `schema`, installs only the schema package.
- `databases-ci.yaml` — paths `missense_kinase_toolkit/databases/**`, coverage
  flag `databases`, installs schema **then** databases (`--no-deps`, because
  databases depends on schema). Runs pytest with `-n 2 --dist loadfile`.

Both: matrix `os: [macOS, ubuntu, windows] × python: ["3.10", "3.11"]`,
`mamba-org/setup-micromamba@v1` with
`environment-file: devtools/conda-envs/test_env.yaml`, then Codecov upload with
the per-package `flags`. There is **no** workflow for `ml/`, `experiments/`, or
`app/`.

### Adding / changing

- New runtime/test dependency → add it to `devtools/conda-envs/test_env.yaml`
  (CI installs the package with `--no-deps`, so deps must be in the env file).

### Pre-commit

`.pre-commit-config.yaml` is scoped to `^missense_kinase_toolkit` (excluding
`KinaseInfo/`): `black`, `isort` (profile black), `flake8` (max-line-length 88,
ignore E203/E501), `pyupgrade --py39-plus`, plus whitespace/yaml hooks.
