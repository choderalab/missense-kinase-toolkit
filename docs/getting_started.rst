Getting Started
===============

This page details how to get started with missense-kinase-toolkit. We have used a monorepo structure where all sub-packages are contained within the :code:`missense-kinase-toolkit` sub-directory.

Installation
++++++++++++

To install the :code:`mkt` sub-packages from Github, you can use the following command:

.. code-block:: bash

    pip install git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/<sub-package directory>

Optional: PyMOL
---------------

:code:`mkt.databases.sasa` can compute per-residue solvent-accessible surface area with an in-process PyMOL session (:code:`pymol2`) in addition to the default Biopython (Shrake-Rupley) backend. PyMOL is not a required dependency. If :code:`pymol2` cannot be imported, the PyMOL backend is dropped with a warning and SASA falls back to Biopython.

On Linux and macOS, install the :code:`pymol` extra, which pulls the :code:`pymol-open-source-whl` wheel from PyPI:

.. code-block:: bash

    pip install "mkt-schema @ git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/schema"
    pip install "mkt-databases[pymol] @ git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/databases"

On Windows, the PyPI wheel installs but cannot start a PyMOL session, so the :code:`pymol` extra installs nothing there. Install PyMOL from conda-forge instead, then install the sub-packages into the same environment:

.. code-block:: bash

    conda create -n mkt -c conda-forge python=3.11 pymol-open-source
    conda activate mkt
    pip install "mkt-schema @ git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/schema"
    pip install "mkt-databases @ git+https://github.com/choderalab/missense-kinase-toolkit.git#subdirectory=missense_kinase_toolkit/databases"
