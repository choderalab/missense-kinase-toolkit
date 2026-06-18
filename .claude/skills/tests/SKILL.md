---
name: tests
description: >-
  How to write and run tests in missense-kinase-toolkit (pytest layout per
  sub-package, the network marker, token-gated skips, session/module fixtures
  and xdist-safe patterns in conftest.py). Use when adding or running tests, or
  debugging test failures.
---

# Testing in missense-kinase-toolkit

## Locations

- `missense_kinase_toolkit/schema/mkt/schema/tests/test_schema.py`
- `missense_kinase_toolkit/databases/mkt/databases/tests/` — one
  `test_<module>.py` per source module, shared fixtures in `conftest.py`.
- No tests for `ml/`, `experiments/`, or `app/`.

## Running

```bash
pytest -v --cov-report=xml --color=yes --cov=mkt.schema \
    missense_kinase_toolkit/schema/mkt/schema/tests/
pytest -v --cov-report=xml --color=yes --cov=mkt.databases \
    missense_kinase_toolkit/databases/mkt/databases/tests/
```

Use the project venv: `missense_kinase_toolkit/VE/bin/python -m pytest ...`.
The databases suite runs under xdist in CI (`-n 2 --dist loadfile`); keep it
parallel-safe locally too.

## Markers and gating

- Mark any test that hits the network with `@pytest.mark.network` (registered
  in `conftest.py`). Apply to the class or function.
- Gate tests needing a token behind a module-level skip, e.g. OncoKB:

  ```python
  pytestmark = pytest.mark.skipif(
      maybe_get_oncokb_token() is None,
      reason="OncoKB API token not set (ONCOKB_TOKEN); skipping live tests",
  )
  ```

## Fixture conventions (`conftest.py`)

- Query each external API **once** — use `scope="session"` or `scope="module"`
  fixtures (e.g. `egfr_uniprot`, `egfr_klifs_info`), then assert against the
  shared object. Imports go inside the fixture to keep collection cheap.
- For xdist, guard shared on-disk work with a `filelock.FileLock` (see
  `kincore_harmonized_dict`) so parallel workers don't clobber each other.
- Skip gracefully on transient API failures: `pytest.skip(...)` when a live
  endpoint returns non-200 rather than asserting hard.
- Set config via the `mkt.databases.config` setters in fixtures
  (`set_output_dir(".")`, `set_cbioportal_instance(...)`), not `os.environ`.
