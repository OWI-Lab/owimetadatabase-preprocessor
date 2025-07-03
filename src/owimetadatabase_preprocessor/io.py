"""Module for the base class handling the access to the Database API."""

import json
import warnings
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import requests

from owimetadatabase_preprocessor.utility.exceptions import (
    APIConnectionError,
    DataProcessingError,
    InvalidParameterError,
)
from owimetadatabase_preprocessor.utility.utils import deepcompare


class API:
    """Base API class handling user access information to the Database API."""

    def __init__(
        self,
        api_root: str = "https://owimetadatabase.azurewebsites.net/api/v1",
        token: Union[str, None] = None,
        uname: Union[str, None] = None,
        password: Union[str, None] = None,
        **kwargs,
    ) -> None:
        """Create an instance of the API class with the required parameters.

        :param api_root: Optional: root URL of the API endpoint, the default
            working database url is provided.
        :param token: Optional: token to access the API.
        :param uname: Optional: username to access the API.
        :param password: Optional: password to access the API.
        :param kwargs: Additional parameters to pass to the API.
        :return: None
        """
        self.api_root = api_root
        self.uname = uname
        self.password = password
        self.auth = None
        self.header = None
        if "header" in kwargs.keys():
            self.header = kwargs["header"]
            if "Authorization" in self.header:
                if not self.header["Authorization"].startswith("Token "):
                    if self.header["Authorization"].startswith("token "):
                        self.header = {
                            "Authorization": f"Token {self.header['Authorization'][6:]}"
                        }
                    elif self.header["Authorization"].startswith(
                        "token"
                    ) or self.header["Authorization"].startswith("Token"):
                        self.header = {
                            "Authorization": f"Token {self.header['Authorization'][5:]}"
                        }
                    else:
                        self.header = {
                            "Authorization": f"Token {self.header['Authorization']}"
                        }
            else:
                raise InvalidParameterError(
                    "If you provide a header directly, \
                    the header must contain the 'Authorization' \
                    key with the value starting with 'Token'."
                )
        elif token:
            self.header = {"Authorization": f"Token {token}"}
        elif self.uname and self.password:
            self.auth = requests.auth.HTTPBasicAuth(self.uname, self.password)
        else:
            raise InvalidParameterError(
                "Either header, token or user name and password must be defined."
            )

    def __eq__(self, other: Union["API", dict]) -> bool:
        """
        Compare two instances of the API class.

        :param other: Another instance of the API class or a dictionary.
        :return: True if the instances are equal, False otherwise.
        """
        if isinstance(other, type(self)):
            comp = deepcompare(self, other)
            assert comp[0], comp[1]
        elif isinstance(other, dict):
            comp = deepcompare(self.__dict__, other)
            assert comp[0], comp[1]
        else:
            assert False, "Comparison is not possible due to incompatible types!"
        return comp[0]

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
                raise InvalidParameterError(e)
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
        :return: None
        """
        if resp.status_code != 200:
            raise APIConnectionError(
                message=f"Error {resp.status_code}.\n{resp.reason}", response=resp
            )

    @staticmethod
    def output_to_df(response: requests.Response) -> pd.DataFrame:
        """Transform output to Pandas dataframe.

        :param response: Raw output of the sent request.
        :return: Pandas dataframe of the data from the output.
        """
        try:
            data = json.loads(response.text)
        except Exception as err:
            raise DataProcessingError(
                "Failed to decode JSON from API response"
            ) from err
        return pd.DataFrame(data)

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
                raise InvalidParameterError(
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
            raise InvalidParameterError(
                "Output type must be either 'single' or 'list', not "
                + output_type
                + "."
            )
        return data_add

    @staticmethod
    def validate_data(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Validate the data extracted from the database.

        :param df: Dataframe of the output data.
        :param data_type: Type of the data we want to request (according to
            database model).
        :return: Dataframe with corrected data.
        """
        z_sa_mp = {"min": -100000, "max": -10000}
        z_sa_tp = {"min": -20000, "max": -1000}
        z_sa_tw = {"min": 1000, "max": 100000}
        sa_type = ["TW", "TP", "MP"]
        z = [z_sa_tw, z_sa_tp, z_sa_mp]
        if data_type == "subassemblies":
            if df.__len__() == 0:
                return df
            for i, sat in enumerate(sa_type):
                cond_small_units = (df["subassembly_type"] == sat) & (
                    df["z_position"] < z[i]["min"]
                )
                cond_big_units = (df["subassembly_type"] == sat) & (
                    df["z_position"] > z[i]["max"]
                )
                if df[cond_small_units].__len__() > 0:
                    df.loc[cond_small_units, "z_position"] = (
                        df.loc[cond_small_units, "z_position"] / 1e3
                    )
                    warnings.warn(
                        f"The value of z location for {df.loc[cond_small_units | cond_big_units, 'title'].values} \
                        might be wrong or in wrong units! There will be an attempt to correct the units."
                    )
                if df[cond_big_units].__len__() > 0:
                    df.loc[cond_big_units, "z_position"] = (
                        df.loc[cond_big_units, "z_position"] * 1e3
                    )
                    warnings.warn(
                        f"The value of z location for {df.loc[cond_small_units | cond_big_units, 'title'].values} \
                        might be wrong or in wrong units! There will be an attempt to correct the units."
                    )
        return df

    def process_data(
        self, url_data_type: str, url_params: Dict[str, str], output_type: str
    ) -> Tuple[pd.DataFrame, Dict[str, Union[bool, np.int64, None]]]:
        """Process output data according to specified request parameters.

        :param url_data_type: Type of the data we want to request (according to
            database model).
        :param url_params: Parameters to send with the request to the database.
        :param output_type: Expected type (amount) of the data extracted.
        :return: A tuple of dataframe with the requested data and additional
            data from postprocessing.
        """
        resp = self.send_request(url_data_type, url_params)
        self.check_request_health(resp)
        df = self.output_to_df(resp)
        df = self.validate_data(df, url_data_type)
        df_add = self.postprocess_data(df, output_type)
        # Add the response object to the returned dictionary so tests can inspect it
        df_add["response"] = resp
        return df, df_add
