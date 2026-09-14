"""Logging configuration shared by all mkt sub-packages.

Provides :func:`configure_logging`, which attaches a single stream handler to the ``mkt``
namespace logger (inherited by ``mkt.schema``, ``mkt.databases``, and ``mkt.ml``), and
:func:`add_logging_flags` to attach a ``--verbose`` level flag to an argparse parser.
"""

import logging

STR_LOGGER_NAMESPACE = "mkt"
"""str: Parent logger of every mkt sub-package (mkt.schema, mkt.databases, mkt.ml)."""

STR_HANDLER_NAME = "mkt"
"""str: Name of the handler attached by configure_logging, so repeat calls replace it."""

STR_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""str: Log record format used by the mkt stream handler."""

LIST_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
"""list[str]: Level names accepted by the ``--verbose`` flag."""


def add_logging_flags(parser):
    """Add a ``--verbose`` logging-level flag to an argparse parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.

    Returns
    -------
    argparse.ArgumentParser
        The same parser with ``--verbose`` added (pass ``args.verbose`` as ``level``).
    """
    parser.add_argument(
        "--verbose",
        type=str,
        default="INFO",
        choices=LIST_LOG_LEVELS,
        help="Set the logging level.",
    )
    return parser


def configure_logging(verbose: bool = False, level: str | int | None = None) -> None:
    """Attach one stream handler to the ``mkt`` namespace logger.

    Idempotent: a handler from a previous call is replaced, so repeated calls (e.g. on
    every Streamlit rerun) do not duplicate output. Call from entry points only, not at
    import time.

    Parameters
    ----------
    verbose : bool, optional
        If True, log at DEBUG, else INFO, by default False; ignored when ``level`` is set.
    level : str | int | None, optional
        Explicit level name (e.g. ``"WARNING"``) or number, by default None.

    Returns
    -------
    None
        The ``mkt`` logger is configured in place.
    """
    if level is None:
        level = logging.DEBUG if verbose else logging.INFO
    elif isinstance(level, str):
        level = level.upper()

    logger = logging.getLogger(STR_LOGGER_NAMESPACE)
    for handler in [h for h in logger.handlers if h.get_name() == STR_HANDLER_NAME]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.set_name(STR_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(STR_LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
