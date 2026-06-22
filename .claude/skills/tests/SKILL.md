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

- `missense_kinase_toolkit/schema/mkt/schema/tests/test_schema.py`
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

The databases suite runs under xdist in CI (`-n 2 --dist loadfile`); keep it
parallel-safe locally too (guard shared on-disk work with `filelock.FileLock` —
see `kincore_harmonized_dict`).

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
