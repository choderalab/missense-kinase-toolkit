import os
from functools import cache

from requests.adapters import HTTPAdapter, Retry
from requests_cache import CachedSession

# this script was written by Jeff Quinn (MSKCC, Tansey lab)
REQUEST_CACHE_VAR = "REQUEST_CACHE"


def add_retry_to_session(
    session,
    retries=5,
    backoff_factor=0.3,
    status_forcelist=(429, 500, 501, 502, 503, 504),
):
    """Add retry logic to a session.

    Parameters
    ----------
    session : requests.Session
        Session object
    retries : int
        Number of retries
    backoff_factor : float
        Backoff factor
    status_forcelist : tuple[int]
        Tuple of status codes to force a retry

    Returns
    -------
    requests.Session
        Session object with retry logic

    """
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@cache
def get_cached_session():
    """Get a cached session.

    Returns
    -------
    requests.Session
        Cached session object

    """
    if REQUEST_CACHE_VAR in os.environ:
        cache_location = os.environ[REQUEST_CACHE_VAR]

        session = CachedSession(
            cache_location, allowable_codes=(200, 404, 400), backend="sqlite"
        )
    else:
        session = CachedSession(backend="memory")

    return add_retry_to_session(session)
