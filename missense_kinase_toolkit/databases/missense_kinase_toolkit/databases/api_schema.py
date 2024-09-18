from abc import ABC, abstractmethod

from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient

# TODO: Define Pydantic data models for the following data sources:
# REST API and Swagger API clients
# https://pypi.org/project/abstract-http-client/
# cBioPortal mutations
# cBioPortal clinical annotations
# Pfam annotations
# UniProt annotations (cannonical sequence)
# Kinase lists
# KLIFs annotations


class SwaggerAPIClient(ABC):
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


class RESTAPIClient(ABC):
    @abstractmethod
    def query_api(self): ...
