---
name: formatting
description: >-
  missense-kinase-toolkit-specific Python style; extends my central `formatting`
  skill. Use whenever writing or editing any Python code in this repo.
---

# missense-kinase-toolkit code formatting

## Baseline — fetch first

Apply my canonical `formatting` conventions (imports, NumPy-style docstrings,
trailing-docstring constants, lowercase comments, double quotes) before the
repo-specific notes below. WebFetch and follow:

https://raw.githubusercontent.com/jessicaw9910/skills/main/.claude/skills/formatting/SKILL.md

If the fetch fails (no network / non-200), **tell me the central `formatting`
skill could not be retrieved** and confirm how to proceed — do not silently skip
the baseline.

## Repo-specific additions

- This is a mono repo; local imports come from the per-sub-package namespaces:
  `mkt.schema.*`, `mkt.databases.*`, `mkt.ml.*`.
- Enforced by pre-commit (`black`, `isort` profile "black", `flake8`
  max-line-length 88 ignoring E203/E501, `pyupgrade --py39-plus`), scoped to
  `^missense_kinase_toolkit` and excluding `KinaseInfo/`. Run
  `pre-commit run --all-files` before committing.
- The trailing-docstring constant pattern is modeled on the constants in
  `mkt.databases.klifs` — match that style.
