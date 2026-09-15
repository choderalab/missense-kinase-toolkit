"""Repo-wide pytest hooks shared by every sub-package test suite.

Live API tests carry the ``network`` marker. When one of them fails because the
network or the upstream service was unavailable -- rather than because the code
under test is wrong -- :func:`pytest_runtest_makereport` rewrites the failure as
a skip and logs a warning, so CI reports an honest signal instead of going red
for someone else's outage. Failures that are genuinely ours (assertions, bad
request construction, 4xx responses) are left alone.
"""

import logging

import pytest
from requests import RequestException
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    ContentDecodingError,
    HTTPError,
    RetryError,
    Timeout,
    TooManyRedirects,
)

logger = logging.getLogger(__name__)

TUPLE_TRANSIENT_EXC = (
    ConnectionError,
    Timeout,
    TooManyRedirects,
    ChunkedEncodingError,
    ContentDecodingError,
    RetryError,
)
"""tuple[type, ...]: requests exceptions that mean "we never got a usable answer".
Deliberately excludes the malformed-request errors (``MissingSchema``,
``InvalidURL``, ``URLRequired``), which are bugs on our side."""

TUPLE_TRANSIENT_STATUS = (408, 425, 429, 500, 502, 503, 504)
"""tuple[int, ...]: response codes attributable to the upstream service. A 4xx
outside this set means we sent a bad request, so it must still fail."""


def _is_transient(exc: RequestException) -> bool:
    """Whether a requests exception reflects an upstream problem, not our code.

    Parameters
    ----------
    exc : RequestException
        Exception raised by requests (directly or via a wrapper).

    Returns
    -------
    bool
        True if the failure is attributable to the network or the upstream
        service, False if it indicates a bad request on our side
    """
    if isinstance(exc, HTTPError):
        response = exc.response
        return response is not None and response.status_code in TUPLE_TRANSIENT_STATUS
    return isinstance(exc, TUPLE_TRANSIENT_EXC)


def _find_transient_error(exc: BaseException | None) -> RequestException | None:
    """Walk an exception chain for a transient network error.

    Clients frequently re-raise a requests error wrapped in their own exception,
    so the whole ``__cause__`` / ``__context__`` chain is searched.

    Parameters
    ----------
    exc : BaseException | None
        Exception at the head of the chain.

    Returns
    -------
    RequestException | None
        First transient network error found, or None
    """
    set_seen: set[int] = set()
    while exc is not None and id(exc) not in set_seen:
        set_seen.add(id(exc))
        if isinstance(exc, RequestException) and _is_transient(exc):
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Convert transient network failures in ``network`` tests into skips."""
    report = yield

    if (
        report.outcome != "failed"
        or call.excinfo is None
        or item.get_closest_marker("network") is None
    ):
        return report

    exc = _find_transient_error(call.excinfo.value)
    if exc is None:
        return report

    logger.warning(
        "%s (%s): transient network error, reporting as skipped rather than "
        "failed -- %s: %s",
        item.nodeid,
        report.when,
        type(exc).__name__,
        exc,
    )
    report.outcome = "skipped"
    report.longrepr = (
        str(item.path),
        item.location[1] or 0,
        f"Skipped: transient network error ({type(exc).__name__}: {exc})",
    )
    return report
