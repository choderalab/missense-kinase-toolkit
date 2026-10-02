Getting Started
===============

This page details how to get started with missense-kinase-toolkit. We have used a monorepo structure where all sub-packages are contained within the :code:`missense_kinase_toolkit` sub-directory:

- :code:`mkt-schema` (:code:`mkt.schema`): Pydantic models for kinase data (:code:`KinaseInfo`), serialization helpers, and a pre-built set of :code:`KinaseInfo` objects shipped as package data.
- :code:`mkt-databases` (:code:`mkt.databases`): API clients, data harmonization, plotting, and the command-line tools that build the :code:`mkt.schema` data.
- :code:`mkt-ml` (:code:`mkt.ml`): models and training utilities for kinase activity prediction.

If you only want to browse kinases, the Streamlit app at https://mkt-app.streamlit.app needs no installation.

Installation
++++++++++++

All sub-packages support Python 3.9 to 3.12. They are installed from GitHub, not PyPI. :code:`mkt-databases` and :code:`mkt-ml` both depend on :code:`mkt-schema`, so install :code:`mkt-schema` first:

.. code-block:: bash

    pip install "mkt-schema @ git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/schema"

Then add whichever of the other sub-packages you need:

.. code-block:: bash

    pip install "mkt-databases @ git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/databases"
    pip install "mkt-ml @ git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/ml"

Optional: PyMOL
---------------

:code:`mkt.databases.sasa` can compute per-residue solvent-accessible surface area with an in-process PyMOL session (:code:`pymol2`) in addition to the default Biopython (Shrake-Rupley) backend. PyMOL is not a required dependency. If :code:`pymol2` cannot be imported, the PyMOL backend is dropped with a warning and SASA falls back to Biopython.

On Linux and macOS, install :code:`mkt-databases` with the :code:`pymol` extra, which pulls the :code:`pymol-open-source-whl` wheel from PyPI:

.. code-block:: bash

    pip install "mkt-databases[pymol] @ git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/databases"

On Windows, the PyPI wheel installs but cannot start a PyMOL session, so the :code:`pymol` extra installs nothing there. Install PyMOL from conda-forge instead, then install the sub-packages into the same environment:

.. code-block:: bash

    conda create -n mkt -c conda-forge python=3.11 pymol-open-source
    conda activate mkt
    pip install "mkt-schema @ git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/schema"
    pip install "mkt-databases @ git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/databases"

From source (development)
-------------------------

To work on the code, clone the repository and run :code:`bin/create_venv.sh`. It creates a virtual environment at :code:`missense_kinase_toolkit/VE/` with editable installs of the selected sub-packages, and uses `uv <https://docs.astral.sh/uv/>`_ if it is installed (otherwise :code:`venv` + :code:`pip`):

.. code-block:: bash

    git clone https://github.com/choderalab/missense-kinase-toolkit.git
    cd missense-kinase-toolkit
    ./bin/create_venv.sh --python 3.11
    source missense_kinase_toolkit/VE/bin/activate

By default the script installs :code:`mkt-schema`, :code:`mkt-databases`, and the Streamlit app's requirements, each sub-package with all of its extras (:code:`dev`, :code:`test`, and :code:`pymol` for :code:`mkt-databases`). Useful options:

- :code:`--ml` / :code:`--no-databases` / :code:`--no-app` (and the other :code:`--[no-]<sub-package>` flags) choose which sub-packages to install. :code:`mkt-schema` is required by all of the others.
- Positional arguments pick the extras instead of installing all of them, e.g. :code:`./bin/create_venv.sh --python 3.11 test`.
- :code:`--overrides-only` re-applies the editable installs to an existing :code:`VE/` without rebuilding it.
- If :code:`missense_kinase_toolkit/.env` exists, its variables are added to :code:`VE/bin/activate` (see `Environment variables`_).

Run :code:`./bin/create_venv.sh --help` for the full usage.

Quick start
+++++++++++

:code:`mkt-schema` ships a pre-built :code:`KinaseInfo` object for each kinase, so you can explore kinase data without network access or API tokens. Pass :code:`list_ids` to load only the kinases you need. Loading all of them takes much longer and uses a lot of memory.

.. code-block:: python

    from mkt.schema.io_utils import deserialize_kinase_dict

    dict_kinase = deserialize_kinase_dict(list_ids=["ABL1", "EGFR"])
    abl1 = dict_kinase["ABL1"]

    abl1.uniprot_id                    # 'P00519'
    abl1.adjudicate_group()            # 'TK'
    abl1.adjudicate_kd_sequence()      # kinase domain sequence
    abl1.KLIFS2UniProtIdx["I:1"]       # UniProt residue at KLIFS pocket position I:1

Each :code:`KinaseInfo` combines data from HGNC, UniProt, KinHub, KLIFS, Pfam, and KinCore. See the :doc:`api` for all fields and methods.

Environment variables
+++++++++++++++++++++

Loading the bundled :code:`KinaseInfo` objects needs no configuration. The command-line tools and API clients that query external services or write results read these environment variables:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variable
     - Purpose
   * - :code:`OUTPUT_DIR`
     - Directory for generated files. Tools that write output exit if it is not set.
   * - :code:`CBIOPORTAL_INSTANCE`
     - cBioPortal API host (e.g. :code:`www.cbioportal.org`). The cBioPortal client exits if it is not set.
   * - :code:`CBIOPORTAL_TOKEN`
     - Optional data-access token for private cBioPortal instances.
   * - :code:`ONCOKB_TOKEN`
     - OncoKB API token, required for the annotation clients (:code:`OncoKBProteinChange`, :code:`OncoKBGenomicChange`, :code:`OncoKBStructuralVariant`). :code:`OncoKBInfo` (:code:`/info`) and :code:`OncoKBCancerGeneList` (:code:`/utils/cancerGeneList`) use public endpoints and work without it.
   * - :code:`REQUESTS_CACHE`
     - Optional path to a :code:`requests-cache` SQLite database, to cache API responses between runs.

Command-line tools
++++++++++++++++++

Installing :code:`mkt-databases` adds these commands. Each one takes :code:`--help`.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Command
     - Purpose
   * - :code:`generate_kinaseinfo_objects`
     - Build :code:`KinaseInfo` objects from the source databases and serialize them.
   * - :code:`extract_cbioportal_missense_kinases`
     - Extract missense mutations in kinases from a cBioPortal study.
   * - :code:`generate_dataset_csv_files`
     - Build the processed dataset CSVs (e.g. Davis, PKIS2) and render their figures.
   * - :code:`generate_conservation_data`
     - Generate the KLIFS conservation-data artifact and its figures.
   * - :code:`generate_pymol_files`
     - Generate PyMOL visualization files for kinase structures.

:code:`mkt-ml` adds :code:`generate_datasets` (build machine-learning datasets) and :code:`run_trainer` (run model training).
