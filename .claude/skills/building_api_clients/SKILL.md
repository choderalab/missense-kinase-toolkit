---
name: building_api_clients
description: >-
  How to add or modify a database/API client in mkt.databases (the APIClient
  hierarchy in api_schema.py, query datetime + cache provenance stamping,
  requests caching, token handling). Use when integrating a new data source
  (REST/Swagger/GraphQL) or editing an existing client like cbioportal, oncokb,
  uniprot, chembl, open_targets, hgnc, pfam, ncbi, protvar, cancer_hotspots.
---

# Building API clients in mkt.databases

All clients live in `missense_kinase_toolkit/databases/mkt/databases/` and build
on the hierarchy in `api_schema.py`.

## Pick the right base class

```
APIClient (base, @dataclass)
  ├── SwaggerAPIClient (abstract)
  │     └── APIKeySwaggerClient → cBioPortal (cbioportal.py)
  ├── RESTAPIClient (abstract)
  │     └── APIKeyRESTAPIClient → OncoKB (oncokb.py)
  └── GraphQLClient → OpenTargets (open_targets.py)
```

- **Swagger/OpenAPI** spec → subclass `SwaggerAPIClient` (or
  `APIKeySwaggerClient` if it needs a token). Implement `query_api()` returning
  a `bravado` `SwaggerClient`.
- **Plain REST** → subclass `RESTAPIClient` (or `APIKeyRESTAPIClient`).
- **GraphQL** → use/extend `GraphQLClient`.
- Token-bearing clients implement `maybe_get_token()` and call `set_api_key()`;
  read tokens via `mkt.databases.config` `maybe_get_*_token()`, never
  `os.environ` directly.

Clients are `@dataclass`es that do their work in `__post_init__` (call
`super().__post_init__()` first when subclassing a concrete client). Mark
derived/post-init fields `field(init=False)`.

## HTTP + caching

Route every HTTP call through the shared cached session:

```python
from mkt.databases.requests_wrapper import get_cached_session

res = get_cached_session().get(url)
APIClient.check_response(res)  # logs HTTPError, does not raise
```

bravado `SwaggerClient` bypasses requests-cache.

## Stamp query datetime + cache provenance

`APIClient` carries `query_datetime` and `from_cache`. Stamp **once per init
flow**, in the parent's `query_api`, and let inheritance carry it to subclasses
(do not re-stamp per request):

- requests-cache responses → `self._stamp_from_response(res)` (reads
  `created_at` / `from_cache`).
- clients that bypass the cache (e.g. bravado) → `self._stamp_now()`
  (`from_cache` left `None`).

## Conventions

- Store the raw payload in `_data`/`_json` and the harmonized result in `_df`
  (`field(init=False)`); expose `get_*()` accessors.
- DataFrame helpers live in `mkt.databases.io_utils`
  (`parse_iterabc2dataframe`, `save_dataframe_to_csv`, `return_kinase_dict`).
- Log with a module-level `logger = logging.getLogger(__name__)`; warn (don't
  crash) when an entity/token is missing so callers can fall back to CSV.
- Add a matching test module under `tests/` marked `@pytest.mark.network`
  (see the `tests` skill).
