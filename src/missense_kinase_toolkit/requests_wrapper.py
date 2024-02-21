from requests_cache import CachedSession
import os
from functools import cache
from requests.adapters import HTTPAdapter, Retry

# this script was written by Jeff Quinn (MSKCC, Tansey lab)

ETL_REQUEST_CACHE_VAR = "ETL_REQUEST_CACHE"


def add_retry_to_session(
    session,
    retries=5,
    backoff_factor=0.3,
    status_forcelist=(429, 500, 501, 502, 503, 504),
):
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
    if "ETL_REQUEST_CACHE" in os.environ:
        cache_location = os.environ["ETL_REQUEST_CACHE"]

        session = CachedSession(
            cache_location, allowable_codes=(200, 404, 400), backend="sqlite"
        )
    else:
        session = CachedSession(backend="memory")

    return add_retry_to_session(session)
