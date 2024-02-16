"""
Unit and regression test for the missense_kinase_toolkit package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import missense_kinase_toolkit


def test_missense_kinase_toolkit_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "missense_kinase_toolkit" in sys.modules
