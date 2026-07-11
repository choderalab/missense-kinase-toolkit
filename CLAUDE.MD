# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Skills

This repo ships project skills under `.claude/skills/`. **Invoke the relevant
skill before doing that kind of work** — they hold the detailed
conventions/recipes that used to live in this file.

The generic skills (`formatting`, `plotting`, `tests`, `ci`, `git`, `docs`) are thin:
they `WebFetch` my canonical baseline conventions from the central skills repo
(`github.com/jessicaw9910/skills`) and layer on only the repo-specific
additions. If a fetch fails, the skill will say so rather than silently skipping
the baseline.

- **`formatting`** — before writing or editing any Python (import grouping,
  NumPy-style docstrings, lowercase comments, double-quoted strings).
- **`plotting`** — before writing or editing any figure/plot code (Arial,
  `LIST_NPG_COLORS`, PowerPoint/PDF-safe transparency, spine removal).
- **`building_api_clients`** — before adding or editing a database/API client in
  `mkt.databases` (the `APIClient` hierarchy, caching, query/cache stamping).
- **`tests`** — before adding or running tests (pytest layout, `network` marker,
  token-gated skips, xdist-safe fixtures).
- **`ci`** — before editing `.github/workflows`, adding a sub-package, or
  changing the test env (path-filtered per-sub-package workflows, micromamba).
- **`pymol_viz`** — before generating/editing PyMOL files, structure highlight
  configs, or SASA (licensed-PyMOL scripts vs in-process `pymol2`).
- **`app_integration`** — before editing the Streamlit app or
  `mkt.databases.app` (logic in the package, UI in Streamlit).
- **`git`** — before creating commits, branches, or pull requests.
- **`docs`** — before building/regenerating the Sphinx API docs or writing
  docstrings for the rendered docs (autosummary + napoleon; `make clean &&
  make html`; never hand-edit `docs/generated/`).

## Project Overview

`missense-kinase-toolkit` (mkt) is a Python mono repo for integrating kinase data from multiple sources (KLIFS, KinCore, UniProt, cBioPortal, OncoKB, ChEMBL, OpenTargets, HGNC, Pfam, NCBI, ProtVar), modeling kinase schemas with Pydantic, and building ML models for kinase activity prediction. It also includes a Streamlit web application for interactive kinase exploration.

## Mono Repo Structure

All sub-packages reside under `missense_kinase_toolkit/`:

| Sub-package | Pip name | Import namespace | Description |
|---|---|---|---|
| `schema/` | `mkt-schema` | `mkt.schema` | Core Pydantic models (`KinaseInfo`), serialization (json/yaml/toml), constants (KLIFS regions, kinase groups) |
| `databases/` | `mkt-databases` | `mkt.databases` | API clients, web scrapers, data harmonization, plotting, CLI tools |
| `ml/` | `mkt-ml` | `mkt.ml` | PyTorch models (cross-attention, factor), HuggingFace encoders (ChemBERTa, ESM2), W&B tracking |
| `experiments/` | `experiments` | `missense_kinase_toolkit.experiments` | Experimental analysis tools (minimal, poetry-based) |
| `app/` | N/A | N/A | Streamlit web app (not pip-installable, deployed at https://mkt-app.streamlit.app) |

### Dependency Chain

```
mkt-schema          (standalone - no internal deps)
    ↑                   ↑
    |                   |
mkt-databases       mkt-ml
(depends on         (depends on
 mkt-schema)         mkt-schema)
    ↑
    |
app (Streamlit)
(depends on mkt-schema + mkt-databases)
```

## Environment Setup

### Virtual Environment
Always use the project venv for running Python commands:
```bash
missense_kinase_toolkit/VE/bin/python
```

### Python Support
Supported Python versions: 3.9-3.12

### Installing Sub-packages

Order matters when installing from source since sub-packages depend on each other:

```bash
# install schema first (no internal deps)
pip install -e ./missense_kinase_toolkit/schema

# then databases or ml (both depend on schema)
pip install -e ./missense_kinase_toolkit/databases
pip install -e ./missense_kinase_toolkit/ml
```

### Conda/Micromamba Environment (for testing)

```bash
micromamba create -f devtools/conda-envs/test_env.yaml
micromamba activate missense-kinase-toolkit-test
```

### Environment Variables

Managed via `mkt.databases.config` module-level `get_*()`/`set_*()` functions that read from `os.environ`:

| Variable | Required | Description |
|---|---|---|
| `OUTPUT_DIR` | Yes | Path to output directory (calls `sys.exit(1)` if missing) |
| `CBIOPORTAL_INSTANCE` | Yes | cBioPortal API URL (calls `sys.exit(1)` if missing) |
| `CBIOPORTAL_TOKEN` | No | cBioPortal data access token |
| `ONCOKB_TOKEN` | No | OncoKB API token |
| `REQUESTS_CACHE` | No | Path to requests-cache SQLite database |

## Key Commands

### CLI Entry Points (`mkt-databases`)
```bash
generate_kinaseinfo_objects     # build KinaseInfo objects from APIs and serialize to tar.gz
extract_cbioportal_missense_kinases  # extract missense kinase mutations from cBioPortal
generate_dataset_csv_files      # generate CSV dataset files
generate_pymol_files            # generate PyMOL visualization files (see pymol_viz skill)
```

### CLI Entry Points (`mkt-ml`)
```bash
generate_datasets   # generate ML dataset files
run_trainer         # run PyTorch training loop
```

### Development Commands
```bash
# run pre-commit hooks (see ci skill)
pre-commit run --all-files

# run tests per sub-package (see tests skill)
pytest -v --cov=mkt.schema missense_kinase_toolkit/schema/mkt/schema/tests/
pytest -v --cov=mkt.databases missense_kinase_toolkit/databases/mkt/databases/tests/
```

## Code Architecture

### Core Pydantic Models (`mkt.schema.kinase_schema`)

`KinaseInfo` is the central data model representing a kinase at the kinase domain level:

```
KinaseInfo
  ├── hgnc_name: str
  ├── uniprot_id: SwissProtIDSuffix | TrEMBLIDSuffix
  ├── uniprot: UniProt              # sequence, phosphosites
  ├── kinhub: KinHub | None         # KinHub classification (group, family)
  ├── klifs: KLIFS | None           # KLIFS pocket info
  ├── pfam: Pfam | None             # Pfam domain boundaries
  ├── kincore: KinCore | None       # KinCore structure data (FASTA + CIF)
  ├── KLIFS2UniProtIdx: dict | None # KLIFS pocket → UniProt index mapping
  └── KLIFS2UniProtSeq: dict | None # KLIFS region → UniProt sequence mapping
```

Key methods on `KinaseInfo`:
- `adjudicate_kd_sequence()` - prioritizes KinCore CIF > KinCore FASTA > Pfam-sliced UniProt
- `adjudicate_group()` - priority: `kincore.fasta.group` > `kinhub.group` > `klifs.group`
- `is_lipid_kinase()`, `is_pseudogene()` - classification helpers

566 pre-built `KinaseInfo` objects are shipped as `KinaseInfo.tar.gz` (package data in `mkt.schema`), loaded in-memory via `deserialize_kinase_dict()`.

### API Client Hierarchy (`mkt.databases.api_schema`)

```
APIClient (base)
  ├── SwaggerAPIClient (abstract)
  │     └── APIKeySwaggerClient → cBioPortal
  ├── RESTAPIClient (abstract)
  │     └── APIKeyRESTAPIClient → OncoKB
  └── GraphQLClient → OpenTargetsDrugMoA
```

Additional database clients: `UniProtFASTA`, `Pfam`, `HGNC`, `ProteinNCBI`, `ProtvarScore`, `ChEMBLMoleculeSearch`. For how to add or modify a client, see the **`building_api_clients`** skill.

### Serialization (`mkt.schema.io_utils`)

- `serialize_kinase_dict(dict, suffix)` - writes json/yaml/toml per-kinase files
- `deserialize_kinase_dict(suffix, str_path)` - reads from directory or in-memory from `.tar.gz`
- Module-level `_deserialization_cache` prevents redundant re-reads
- TOML serialization skipped on Windows (`os.name == "nt"`)

### Utility Patterns

- **Recursive attribute access** (`mkt.schema.utils`): `rgetattr(obj, "kincore.fasta.group")` / `rsetattr()` for traversing nested Pydantic models
- **Request caching** (`mkt.databases.requests_wrapper`): `get_cached_session()` wraps `requests-cache` for all HTTP calls
- **Config management** (`mkt.databases.config`): module-level `get_*()`/`set_*()` functions reading/writing `os.environ`
- **Logging** (`mkt.databases.log_config`): `configure_logging()` and `add_logging_flags()`

### Visualization & App

Kinase structure visualization (`mkt.databases.app`, `mkt.databases.pymol`) and the Streamlit front-end (`missense_kinase_toolkit/app/`) — see the **`pymol_viz`** and **`app_integration`** skills.

### Versioning

All setuptools sub-packages use `versioningit` for git-tag-based dynamic versioning (default tag: `"0.0.0"`).

## Testing

Tests live per sub-package and include live API tests requiring network access. See the **`tests`** skill for layout, the `network` marker, token-gated skips, and xdist-safe fixtures.

## CI/CD

Path-filtered GitHub Actions workflows run one per setuptools sub-package via micromamba, with Codecov per-package flags; pre-commit enforces formatting/linting. See the **`ci`** skill.

## Documentation

Built with Sphinx (NumPy-style docstrings via Napoleon) and hosted on ReadTheDocs. Config in `docs/conf.py`, build dependencies in `docs/requirements.yaml` and `docs/requirements.txt`.

## Logging

Configure logging via `mkt.databases.log_config`:
```python
from mkt.databases.log_config import configure_logging

configure_logging()
```
