Getting Started
===============

This page details how to get started with missense-kinase-toolkit.

Installation
++++++++++++

.. code-block:: bash
    git clone https://github.com/choderalab/missense-kinase-toolkit.git
    cd missense-kinase-toolkit
    python3 -m venv VE
    source VE/bin/activate
    pip install -e '.[test,extras]'

Alternatively, since we have used `poetry` as our default package manager for this project. Once you have cloned the repository and have either installed `poetry` locally or in your environment of interest, you can install the package by running the following command in the root directory of the repository using the provided `poetry.lock` file and the following command

.. code-block:: bash

    poetry install
