"""Base API client hierarchy (Swagger, REST, GraphQL) with query and cache provenance stamping.

Defines :class:`APIClient` and its abstract subclasses (:class:`SwaggerAPIClient`,
:class:`RESTAPIClient`, :class:`GraphQLClient`, and their API-key variants), which
centralize request execution, query-datetime recording, and requests-cache provenance
for all concrete database clients.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests
from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
from mkt.databases.requests_wrapper import get_cached_session

logger = logging.getLogger(__name__)


@dataclass
class APIClient:
    """Base class for API clients."""

    query_datetime: datetime | None = field(default=None, init=False)
    """UTC datetime the underlying network query was made (cache creation time if served from cache)."""
    from_cache: bool | None = field(default=None, init=False)
    """Whether the most recent response was served from requests-cache."""

    @staticmethod
    def check_response(res: requests.Response) -> None:
        """Check the response status code for errors."""
        try:
            res.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("Error at %s", "division", exc_info=e)

    def _stamp_from_response(self, res: requests.Response) -> None:
        """Record query datetime + cache provenance from a requests-cache response.

        Parameters:
        -----------
        res : requests.Response
            Response object from a requests-cache CachedSession; falls back to current
            UTC time if the response lacks requests-cache metadata.
        """
        created_at = getattr(res, "created_at", None)
        if created_at is not None:
            # normalize to tz-aware UTC (older requests-cache versions return tz-naive)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            self.query_datetime = created_at
            self.from_cache = bool(getattr(res, "from_cache", False))
        else:
            self.query_datetime = datetime.now(timezone.utc)
            self.from_cache = False

    def _stamp_now(self) -> None:
        """Record current UTC datetime; cache provenance unknown.

        For clients that bypass requests-cache (e.g., bravado SwaggerClient), this
        records when the query ran but cannot determine if a cached layer was hit.
        """
        self.query_datetime = datetime.now(timezone.utc)
        self.from_cache = None


class SwaggerAPIClient(APIClient, ABC):
    """Base class for Swagger API clients."""

    @abstractmethod
    def query_api(self) -> SwaggerClient:
        """Query a Swagger API and return result.

        Parameters
        ----------
        url : str
            API URL

        Returns
        -------
        SwaggerClient
            API response
        """
        ...


class APIKeySwaggerClient(SwaggerAPIClient, ABC):
    """Base class for Swagger API clients with API key support."""

    @abstractmethod
    def maybe_get_token(self) -> str | None:
        """Get API token, if available.

        Returns
        -------
        str | None
            API token if available; None otherwise

        """
        ...

    def set_api_key(self) -> RequestsClient:
        """Set API key for cBioPortal API.

        Returns
        -------
        RequestsClient
            RequestsClient object with API key set

        """
        token = self.maybe_get_token()
        http_client = RequestsClient()
        if token is not None:
            http_client.set_api_key(
                self.instance,
                f"Bearer {token}",
                param_name="Authorization",
                param_in="header",
            )
        else:
            print("No API token provided")
        return http_client


class RESTAPIClient(APIClient, ABC):
    """Base class for REST API clients."""

    @abstractmethod
    def query_api(self): ...


class APIKeyRESTAPIClient(RESTAPIClient, ABC):
    """Base class for REST API clients with API key support."""

    @abstractmethod
    def maybe_get_token(self) -> str | None:
        """Get API token, if available.

        Returns
        -------
        str | None
            API token if available; None otherwise

        """
        ...

    def set_api_key(self) -> dict:
        """Set API token for REST API.

        Returns
        -------
        dict
            Dictionary with API token set

        """
        token = self.maybe_get_token()
        return {"Authorization": f"Bearer {token}"} if token else {}


# currently only using for OpenTargets API - may want to generalize later
@dataclass
class GraphQLClient(APIClient):
    """Base class for GraphQL API clients."""

    url: str | None = None
    """GraphQL API URL."""
    query_string: str | None = None
    """GraphQL query string to be executed."""
    variables: dict = field(default_factory=dict)
    """Variables for the GraphQL query, if any."""
    response: dict | None = None
    """Response from the GraphQL API query, if any."""

    def __post_init__(self):
        """Initialize the GraphQL API client."""
        if self.url is None:
            raise ValueError("API URL must be provided.")
        if self.query_string is None:
            raise ValueError("Query string must be provided.")
        try:
            self.response = self.query_api()
        except requests.exceptions.RequestException as e:
            logger.error("Error querying GraphQL API: %s", e)
            raise

    def query_api(self) -> dict:
        """Query a GraphQL API and return result.

        Returns
        -------
        dict
            API response
        """
        res = get_cached_session().post(
            self.url,
            json={
                "query": self.query_string,
                "variables": self.variables,
            },
        )
        res.raise_for_status()
        self._stamp_from_response(res)
        return res.json()
