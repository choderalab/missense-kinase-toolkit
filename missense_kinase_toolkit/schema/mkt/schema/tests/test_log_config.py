"""Tests for the shared mkt logging configuration."""

import argparse
import logging

import pytest


@pytest.fixture
def mkt_logger():
    """Yield the ``mkt`` namespace logger, restoring its handlers and level afterwards."""
    from mkt.schema.log_config import STR_LOGGER_NAMESPACE

    logger = logging.getLogger(STR_LOGGER_NAMESPACE)
    list_handlers, int_level = list(logger.handlers), logger.level
    yield logger
    logger.handlers[:] = list_handlers
    logger.setLevel(int_level)


def test_configure_logging_is_idempotent(mkt_logger):
    """Repeat calls (e.g. Streamlit reruns) replace the handler instead of stacking."""
    from mkt.schema.log_config import STR_HANDLER_NAME, configure_logging

    for _ in range(3):
        configure_logging()

    assert [h.get_name() for h in mkt_logger.handlers].count(STR_HANDLER_NAME) == 1


def test_configure_logging_levels(mkt_logger):
    """``verbose`` toggles DEBUG/INFO; an explicit ``level`` overrides it."""
    from mkt.schema.log_config import configure_logging

    configure_logging()
    assert mkt_logger.level == logging.INFO

    configure_logging(verbose=True)
    assert mkt_logger.level == logging.DEBUG

    configure_logging(verbose=True, level="warning")
    assert mkt_logger.level == logging.WARNING

    # sub-package loggers inherit from the namespace logger
    for name in ("mkt.schema.io_utils", "mkt.databases.app", "mkt.ml.trainer"):
        assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING


def test_add_logging_flags():
    """``--verbose`` parses a level name that configure_logging accepts."""
    from mkt.schema.log_config import add_logging_flags

    parser = add_logging_flags(argparse.ArgumentParser())
    assert parser.parse_args([]).verbose == "INFO"
    assert parser.parse_args(["--verbose", "DEBUG"]).verbose == "DEBUG"
