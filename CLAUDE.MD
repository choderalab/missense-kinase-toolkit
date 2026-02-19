# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
generate_pymol_files            # generate PyMOL visualization files
```

### CLI Entry Points (`mkt-ml`)
```bash
generate_datasets   # generate ML dataset files
run_trainer         # run PyTorch training loop
```

### Development Commands
```bash
# run pre-commit hooks
pre-commit run --all-files

# run tests per sub-package
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

Additional database clients: `UniProtFASTA`, `Pfam`, `HGNC`, `ProteinNCBI`, `ProtvarScore`, `ChEMBLMoleculeSearch`

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

### Versioning

All setuptools sub-packages use `versioningit` for git-tag-based dynamic versioning (default tag: `"0.0.0"`).

## Testing

### Test Locations
- `missense_kinase_toolkit/schema/mkt/schema/tests/test_schema.py`
- `missense_kinase_toolkit/databases/mkt/databases/tests/test_databases.py`
- No tests exist for `ml`, `experiments`, or `app`

### Running Tests
```bash
pytest -v --cov-report=xml --color=yes --cov=mkt.schema missense_kinase_toolkit/schema/mkt/schema/tests/
pytest -v --cov-report=xml --color=yes --cov=mkt.databases missense_kinase_toolkit/databases/mkt/databases/tests/
```

Note: `test_databases.py` includes live API tests (cBioPortal, HGNC, UniProt, ChEMBL, etc.) that require network access.

## CI/CD

### GitHub Actions Workflows

- **`schema-ci.yaml`**: triggers on push/PR to `main` affecting `missense_kinase_toolkit/schema/**`; weekly cron
- **`databases-ci.yaml`**: triggers on push/PR to `main` affecting `missense_kinase_toolkit/databases/**`; weekly cron
- Both use: macOS/ubuntu/windows × Python 3.10, 3.11 matrix via `micromamba`
- Coverage uploaded to Codecov with per-package flags

### Pre-commit Hooks (`.pre-commit-config.yaml`)

All hooks scoped to `^missense_kinase_toolkit` (excluding `KinaseInfo/`):
- `black` - code formatting
- `isort` (profile: "black") - import sorting
- `flake8` (max-line-length: 88, ignores E203/E501) - linting
- `pyupgrade` (--py39-plus) - syntax modernization
- `check-yaml`, `debug-statements`, `end-of-file-fixer`, `trailing-whitespace`

## Code Style Guidelines

### Import Statements
- All import statements MUST be placed at the top of the script/module
- Group imports in the following order:
  1. Standard library imports
  2. Third-party library imports (numpy, pandas, matplotlib, etc.)
  3. Local application imports (mkt modules)
- Separate each group with a blank line
- Example:
```python
import numpy as np
from collections import Counter
import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from mkt.schema.kinase_schema import KinaseInfo
from mkt.databases.klifs import KLIFSPocket
```

### Docstrings
- Use NumPy-style docstrings with ":" after section headers
- Required sections: Parameters, Returns
- Example:
```python
def my_function(param1: str, param2: int) -> bool:
    """Brief description.

    Parameters:
    -----------
    param1 : str
        Description of param1.
    param2 : int
        Description of param2.

    Returns:
    --------
    bool
        Description of return value.
    """
```

### Comments
- Use lowercase letters for non-proper noun starting comments
- Example: `# create bins for the count column` (not "Create bins...")

### String Quotes
- Use double quotation marks (`"`) for strings where possible
- Exceptions:
  - Inside f-strings, use single quotes for nested strings: `f"Value: {dict['key']}"`
  - In docstrings, continue using triple double quotes: `"""`
- Example:
```python
# Preferred
filename = "my_file.csv"
color = "#FFD700"
result = f"The value is {my_dict['key']}"

# Avoid
filename = 'my_file.csv'
color = '#FFD700'
```

### Plotting
- Use Nature journal style color palette: `LIST_NPG_COLORS`
- Set plot style consistently:
```python
plt.rcParams["font.family"] = "Arial"
sns.set_palette(sns.color_palette(LIST_NPG_COLORS))
sns.set_style("white")
```
- **PowerPoint/PDF Transparency Best Practices**: To prevent greyish hue on transparent objects when SVGs are used in PowerPoint and converted to PDF:
  - **Option 1 (Recommended)**: Use RGBA tuples for colors with separate alpha values instead of global alpha parameter:
  ```python
  import matplotlib.colors as mcolors

  # For ellipses/patches: transparent fill, opaque edge
  facecolor_rgba = mcolors.to_rgba("#FFD700", alpha=0.5)
  ellipse = Ellipse((x, y), width, height, facecolor=facecolor_rgba, edgecolor="#FFD700")

  # For legends: fully opaque background
  ax.legend(..., framealpha=1.0, facecolor='white', edgecolor='black')
  ```
  This method avoids the PowerPoint PDF rendering issue by baking transparency into color values.

  - **Option 2**: Save as high-DPI PNG instead of SVG for PowerPoint presentations:
  ```python
  fig.savefig("plot.png", dpi=300, bbox_inches='tight', facecolor='white')
  ```
  - **Option 3**: If using SVG, configure matplotlib for PowerPoint compatibility and convert SVG to PowerPoint shapes:
  ```python
  import matplotlib as mpl
  mpl.rcParams["svg.fonttype"] = "none"  # Prevents characters as separate text boxes
  mpl.rcParams["svg.hashsalt"] = "42"    # Makes SVG generation deterministic
  fig.savefig("plot.svg", bbox_inches='tight', facecolor='white')
  ```
  Then in PowerPoint: drag SVG file → right-click → Convert to Shape
- Remove plot spines while keeping tick marks for a cleaner look:
```python
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
```

## Pull Request Template

When creating PRs, use `.github/PULL_REQUEST_TEMPLATE.md`:
- Description section for PR purpose
- Todos checklist for accomplishments
- Questions section for open items
- Status checkbox for readiness

## Documentation

Built with Sphinx (NumPy-style docstrings via Napoleon) and hosted on ReadTheDocs. Config in `docs/conf.py`, build dependencies in `docs/requirements.yaml` and `docs/requirements.txt`.

## Logging

Configure logging via `mkt.databases.log_config`:
```python
from mkt.databases.log_config import configure_logging

configure_logging()
```
