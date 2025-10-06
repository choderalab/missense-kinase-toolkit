import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import requests
from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
from mkt.databases.requests_wrapper import get_cached_session

logger = logging.getLogger(__name__)


@dataclass
class APIClient:
    """Base class for API clients."""

    @staticmethod
    def check_response(res: requests.Response) -> None:
        """Check the response status code for errors."""
        try:
            res.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("Error at %s", "division", exc_info=e)


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
        return res.json()
