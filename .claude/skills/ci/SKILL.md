---
name: ci
description: >-
  GitHub Actions CI layout for the missense-kinase-toolkit mono repo
  (per-sub-package path-filtered workflows, micromamba test env, OS/Python
  matrix, Codecov flags) and pre-commit. Use when editing .github/workflows,
  adding a sub-package, changing the test env, or debugging CI failures.
---

# missense-kinase-toolkit CI/CD

## Workflows (`.github/workflows/`)

One workflow per setuptools sub-package, **path-filtered** so each runs only
when its sub-package changes (plus a weekly `cron: "0 0 * * 0"`):

- `schema-ci.yaml` — paths `missense_kinase_toolkit/schema/**`, coverage flag
  `schema`, installs only the schema package.
- `databases-ci.yaml` — paths `missense_kinase_toolkit/databases/**`, coverage
  flag `databases`, installs schema **then** databases (`--no-deps`, because
  databases depends on schema). Runs pytest with `-n 2 --dist loadfile`
  (xdist) — keep network fixtures session/module-scoped and xdist-safe (see
  the `tests` skill).

Both: matrix `os: [macOS, ubuntu, windows] × python: ["3.10", "3.11"]`,
`mamba-org/setup-micromamba@v1` with
`environment-file: devtools/conda-envs/test_env.yaml`, then Codecov upload with
the per-package `flags`.

There is **no** workflow for `ml/`, `experiments/`, or `app/`.

## Adding / changing a workflow

- New pip-installable sub-package → copy an existing workflow, set its `paths`
  filter, install order (schema first if it depends on schema, `--no-deps`),
  `--cov=mkt.<pkg>`, the test dir, and a unique Codecov `flags`.
- New runtime/test dependency → add it to `devtools/conda-envs/test_env.yaml`
  (CI installs the package with `--no-deps`, so deps must be in the env file).
- Tests needing a token (e.g. OncoKB) are skipped unless the env var is set;
  to enable in CI add a GitHub Actions secret and export it before pytest.

## Pre-commit

`.pre-commit-config.yaml` is scoped to `^missense_kinase_toolkit` (excluding
`KinaseInfo/`): `black`, `isort` (profile black), `flake8` (max-line-length 88,
ignore E203/E501), `pyupgrade --py39-plus`, plus whitespace/yaml hooks. Run
`pre-commit run --all-files` before pushing.
