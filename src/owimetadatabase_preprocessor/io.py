"""Module for the base class handling the access to the Database API."""

import json
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import requests

from owimetadatabase_preprocessor.utils import deepcompare


class API(object):
    """Base API class handling user access information to the Database API."""

    def __init__(
        self,
        api_root: str = "https://owimetadatabase.owilab.be/api/v1",
        token: Union[str, None] = None,
        uname: Union[str, None] = None,
        password: Union[str, None] = None,
        **kwargs,
    ) -> None:
        self.api_root = api_root
        self.uname = uname
        self.password = password
        self.auth = None
        self.header = None
        if "header" in kwargs.keys():
            self.header = kwargs["header"]
        elif token:
            self.header = {"Authorization": f"Token {token}"}
        elif self.uname and self.password:
            self.auth = requests.auth.HTTPBasicAuth(self.uname, self.password)
        else:
            raise ValueError(
                "Either header, token or user name and password must be defined."
            )

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return deepcompare(self, other)
        elif isinstance(other, dict):
            return deepcompare(self.__dict__, other)
        else:
            return False

    def send_request(
        self, url_data_type: str, url_params: Dict[str, str]
    ) -> requests.Response:
        """Handle sending appropriate request according to the type of authentication.

        :param url_data_type: Type of the data we want to request (according to database model).
        :param url_params: Parameters to send with the request to the database.
        :return:  An instance of the Response object.
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
    def check_request_health(resp: requests.Response) -> None:
        """Check status code of the response to request and provide detials if unexpected.

        :param resp: Instance of the Response object.
        :return:
        """
        if resp.status_code != 200:
            e = "Error " + str(resp.status_code) + ".\n" + resp.reason
            raise Exception(e)

    @staticmethod
    def output_to_df(response: requests.Response) -> pd.DataFrame:
        """Transform output to Pandas dataframe.

        :param response: Raw output of the sent request.
        :return: Pandas dataframe of the data from the output.
        """
        df = pd.DataFrame(json.loads(response.text))
        return df

    @staticmethod
    def postprocess_data(
        df: pd.DataFrame, output_type: str
    ) -> Dict[str, Union[bool, np.int64, None]]:
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
        else:
            raise ValueError(
                "Output type must be either 'single' or 'list', not "
                + output_type
                + "."
            )
        return data_add

    def process_data(
        self, url_data_type: str, url_params: Dict[str, str], output_type: str
    ) -> Tuple[pd.DataFrame, Dict[str, Union[bool, np.int64, None]]]:
        """Process output data according to specified request parameters.

        :param url_data_type: Type of the data we want to request (according to database model).
        :param url_params: Parameters to send with the request to the database.
        :param output_type: Expected type (amount) of the data extracted.
        :return: A tuple of dataframe with the requested data and additional data from postprocessing.
        """
        resp = self.send_request(url_data_type, url_params)
        self.check_request_health(resp)
        df = self.output_to_df(resp)
        df_add = self.postprocess_data(df, output_type)
        return df, df_add
