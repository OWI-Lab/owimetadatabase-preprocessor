"""
Custom exceptions for the API client. These exceptions encapsulate
common errors that can occur during API calls and data post-processing.
"""

from typing import Optional

import requests


class APIException(Exception):
    """Base exception for all errors raised by API."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class APIConnectionError(APIException):
    """Exception raised when the API cannot be reached or returns a failure."""

    def __init__(
        self, message: str, response: Optional[requests.Response] = None
    ) -> None:
        self.response = response
        super().__init__(message)

    def __str__(self) -> str:
        status = f" (Status: {self.response.status_code})" if self.response else ""
        return f"{self.message}{status}"


class DataNotFoundError(APIException):
    """Exception raised when no data is found for the given query parameters."""

    def __init__(
        self, message: str = "No data found for the given search criteria."
    ) -> None:
        super().__init__(message)


class DataProcessingError(APIException):
    """Exception raised when there is a problem while processing the data."""

    def __init__(self, message: str = "Error during data processing.") -> None:
        super().__init__(message)


class InvalidParameterError(APIException):
    """Exception raised when query parameters are invalid or missing."""

    def __init__(
        self, message: str = "Invalid or missing parameters for the request."
    ) -> None:
        super().__init__(message)
