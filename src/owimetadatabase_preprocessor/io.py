"""Module for the base class handling the access to the Database API."""

import json
import pandas as pd
import requests


class API(object):
    """Base API class handling user access information to the Database API."""

    def __init__(
        self,
        api_root: str,
        header: dict[str, str] | None = None,
        uname: str | None = None,
        password: str | None = None,
    ) -> None:
        self.api_root = api_root
        self.header = header
        self.uname = uname
        self.password = password
        self.auth = None
        if self.uname is not None and self.password is not None:
            self.auth = requests.auth.HTTPBasicAuth(self.uname, self.password)

    def send_request(
        self,
        url_data_type: str,
        url_params: dict[str, str]
    ) -> requests.Response:
        """Handle sending appropriate request according to the type of authentication.
        
        :param url_data_type: Type of the data we want to request (according to database architecture specifics).
        :param url_params: Parameters to send with the request to the database.
        :return:  An instance of the response object.
        """
        if self.header is not None:
            response = requests.get(
                url=self.api_root + url_data_type,
                headers=self.header,
                params=url_params,
            )
        else:
            if self.uname is None or self.password is None:
                e = "Either self.header or self.uname and self.password must be defined."
                raise ValueError(e)
            else:
                response = requests.get(
                    url=self.api_root + url_data_type,
                    auth=self.auth,
                    params=url_params,
                )
        return response
    
    @staticmethod
    def output_to_df(response: requests.Response) -> pd.DataFrame:
        """Transform output to Pandas dataframe.

        :param response: Raw output of the sent request. 
        :return: Pandas dataframe of the data from the output.
        """
        df = pd.DataFrame(json.loads(response.text))
        return df
    
    @staticmethod
    def postprocess_data(df: pd.DataFrame, output_type: str) -> dict[str, bool | int]:
        """Process dataframe information to extarct the necessary additional data.

        :param df: Dataframe of the output data.
        :param output_type: Expected type (amount) of the data extracted.
        :return: Dictionary of the additional data extracted.
        """
        if output_type == "single":
            if df.__len__() == 0:
                exists = False
                project_id = None
            elif df.__len__() == 1:
                exists = True
                project_id = df["id"].iloc[0]
            else:
                raise ValueError(
                    "More than one project site was returned, check search criteria."
                )
            data_add = {"existance": exists, "id": project_id}
        elif output_type == "list":
            if df.__len__() == 0:
                exists = False
            else:
                exists = True
            data_add = {"existance": exists}
        return data_add
