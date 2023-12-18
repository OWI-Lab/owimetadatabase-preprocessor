from typing import Dict, Union
from unittest import mock

import pandas as pd
import pytest
import requests

from owimetadatabase_preprocessor.io import API


@pytest.fixture
def mock_requests_get(mocker: mock.Mock) -> mock.Mock:
    mock = mocker.patch("requests.get")
    mock.return_value = requests.Response()
    return mock


class TestAPIAuth:
    """Tests of authentication setup."""

    def test_API_header(self, api_root: str) -> None:
        """Test parent API class with header that it initializes everything correctly."""
        header = {"Authorization": "Token 12345"}
        api_test = API(api_root, header=header)
        assert api_test.api_root == api_root
        assert api_test.header == header
        assert api_test.uname is None
        assert api_test.password is None
        assert api_test.auth is None

    def test_API_user(self, api_root: str) -> None:
        """Test parent API class with user credentials that it initializes everything correctly."""
        name = "test"
        pswd = "12345"
        api_test = API(api_root, uname=name, password=pswd)
        assert api_test.api_root == api_root
        assert api_test.header is None
        assert api_test.uname == name
        assert api_test.password == pswd
        assert api_test.auth == requests.auth.HTTPBasicAuth(name, pswd)


def test_send_request_with_token(mock_requests_get: mock.Mock, api_root: str) -> None:
    header = {"Authorization": "Token 12345"}
    url_data_type = "/test/"
    url_params = {"test": "test"}
    api_test = API(api_root, header=header)
    response = api_test.send_request(url_data_type, url_params)
    assert isinstance(response, requests.models.Response)


def test_send_request_with_name_pass(
    mock_requests_get: mock.Mock, api_root: str
) -> None:
    name = "test"
    pswd = "12345"
    url_data_type = "/test/"
    url_params = {"test": "test"}
    api_test = API(api_root, uname=name, password=pswd)
    response = api_test.send_request(url_data_type, url_params)
    assert isinstance(response, requests.models.Response)


def test_check_request_health() -> None:
    response = requests.Response()
    response.status_code = 200
    API.check_request_health(response)
    response.status_code = 404
    with pytest.raises(Exception):
        API.check_request_health(response)


def test_output_to_df() -> None:
    response = requests.Response()
    response._content = (
        b'[{"col_1": 11, "col_2": 12, "col_3": 13}, '
        b'{"col_1": 21, "col_2": 22, "col_3": 23}]'
    )
    df = API.output_to_df(response)
    assert isinstance(df, pd.DataFrame)
    assert df["col_1"][0] == 11
    assert df["col_1"][1] == 21
    assert df["col_2"][0] == 12
    assert df["col_2"][1] == 22
    assert df["col_3"][0] == 13
    assert df["col_3"][1] == 23


@pytest.mark.parametrize(
    "df, output_type, expected_result, expected_exception",
    [
        (pd.DataFrame([]), "single", {"existance": False, "id": None}, None),
        (
            pd.DataFrame([{"id": 239, "col_test": "text test"}]),
            "single",
            {"existance": True, "id": 239},
            None,
        ),
        (
            pd.DataFrame(
                [{"id": 1, "col_test": "text 1"}, {"id": 2, "col_test": "text 2"}]
            ),
            "single",
            None,
            ValueError,
        ),
        (pd.DataFrame([]), "list", {"existance": False}, None),
        (
            pd.DataFrame([{"id": 239, "col_test": "text test"}]),
            "list",
            {"existance": True},
            None,
        ),
        (
            pd.DataFrame(
                [{"id": 1, "col_test": "text 1"}, {"id": 2, "col_test": "text 2"}]
            ),
            "list",
            {"existance": True},
            None,
        ),
        (
            pd.DataFrame([{"id": 239, "col_test": "text test"}]),
            "multiple",
            None,
            ValueError,
        ),
    ],
)
def test_postprocess_data(
    df: pd.DataFrame,
    output_type: str,
    expected_result: Union[None, Dict[str, Union[bool, int]]],
    expected_exception: None,
) -> None:
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            result = API.postprocess_data(df, output_type)
    else:
        result = API.postprocess_data(df, output_type)
        assert result == expected_result


@pytest.fixture
def mock_requests_get_advanced(mocker: mock.Mock) -> mock.Mock:
    mock = mocker.patch("requests.get")

    def response() -> requests.Response:
        resp = requests.Response()
        resp.status_code = 200
        resp._content = (
            b'[{"col_1": 11, "col_2": 12, "col_3": 13}, '
            b'{"col_1": 21, "col_2": 22, "col_3": 23}]'
        )
        return resp

    mock.return_value = response()
    return mock


def test_process_data(mock_requests_get_advanced: mock.Mock, api_root: str) -> None:
    header = {"Authorization": "Token 12345"}
    url_data_type = "/test/"
    url_params = {"test": "test"}
    output_type = "list"
    api_test = API(api_root, header=header)
    df, df_add = api_test.process_data(url_data_type, url_params, output_type)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_add, dict)
    assert isinstance(df_add["existance"], bool)
