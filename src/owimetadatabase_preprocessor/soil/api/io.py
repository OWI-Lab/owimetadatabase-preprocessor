"""
API client Module for the soil data in the OWIMetadatabase.
"""
import requests
import pandas as pd
from typing import Dict, Any, Union

class SoilAPIClient:
    """
    API client to handle HTTP communication for soil data.
    """
    def __init__(
        self,
        api_root: str = "https://owimetadatabase.azurewebsites.net/api/v1",
        api_subdir: str = "/soildata/",
        token: Union[str, None] = None,
        uname: Union[str, None] = None,
        password: Union[str, None] = None,
    ) -> None:
        """
        Constructor for the SoilAPI.

        :param api_root: Base URL for the API
        :param api_subdir: Sub-directory for soil data endpoints
        :param token: API token (if required)
        :param uname: Username for authentication
        :param password: Password for authentication
        """
        self.api_root = api_root + api_subdir
        self.token = token
        self.uname = uname
        self.password = password

    def send_request(self, data_type: str, params: Dict[str, Any]) -> requests.Response:
        """
        Sends an HTTP GET request to the specified API endpoint with given parameters.

        :param data_type: The API endpoint (e.g. "soilprofilesummary")
        :param params: Dictionary of URL parameters to include in the request
        :return: Response object from the API request
        """
        url = self.api_root + data_type
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = requests.get(url, params=params, headers=headers, timeout=10)
        return response

    def check_request_health(self, response: requests.Response) -> None:
        """
        Checks if the API request succeeded.

        :param response: Response object from the API request
        :raises Exception: If the response indicates a failure.
        """
        if not response.ok:
            raise Exception(
                f"API request failed with status code: {response.status_code}. Response: {response.text}"
            )

    def output_to_df(self, response: requests.Response) -> pd.DataFrame:
        """
        Converts the API response (JSON) to a pandas DataFrame.

        :param response: Response object from the API request
        :return: A pandas DataFrame constructed from the JSON data.
        :raises Exception: If JSON decoding fails.
        """
        try:
            data = response.json()
        except Exception as err:
            raise Exception("Failed to decode JSON from API response") from err
        return pd.DataFrame(data)

    def process_data(self, data_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process output data according to specified request parameters.

        :param data_type: Type of the data we want to request (according to 
            database model).
        :param params: Parameters to send with the request to the database.
        :return: Dictionary containing the DataFrame and the raw response.
        """
        response = self.send_request(data_type, params)
        self.check_request_health(response)
        df = self.output_to_df(response)
        return {"dataframe": df, "response": response}