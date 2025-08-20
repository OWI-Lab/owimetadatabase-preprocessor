from typing import Union
from unittest import mock

import pytest
import requests


@pytest.fixture
def mock_requests_get(mocker: mock.Mock) -> mock.Mock:
    mock = mocker.patch("requests.get")
    mock.return_value = requests.Response()
    return mock


@pytest.fixture
def mock_requests_get_advanced(mocker: mock.Mock) -> mock.Mock:
    mock = mocker.patch("requests.get")

    def response() -> requests.Response:
        resp = requests.Response()
        resp.status_code = 200
        resp._content = b'[{"col_1": 11, "col_2": 12, "col_3": 13}, {"col_1": 21, "col_2": 22, "col_3": 23}]'
        return resp

    mock.return_value = response()
    return mock


@pytest.fixture(scope="module")
def api_root() -> str:
    return "https://test.api/test"


@pytest.fixture(scope="module")
def header() -> dict[str, str]:
    return {"Authorization": "Token 12345"}


@pytest.fixture(scope="function")
def dict_gen_dict_in() -> dict[str, str]:
    return {
        "key_1": "value_1",
        "key_2": "value_2",
        "key_3": "value_3",
        "key_4": "value_4",
        "key_5": "value_5",
    }


@pytest.fixture(scope="function", params=[1, 2, 3, 4, 5, 6, 7, 8])
def dict_gen_dict_out(request) -> dict[str, Union[dict, dict[str, Union[str, list[str]]]]]:
    param = request.param
    if param == 1:
        method_keys = {"method_": "exclude", "keys_": ["key_1"]}
        dict_ = {
            "key_2": "value_2",
            "key_3": "value_3",
            "key_4": "value_4",
            "key_5": "value_5",
        }
    elif param == 2:
        method_keys = {"method_": "exclude", "keys_": ["key_1", "key_2"]}
        dict_ = {"key_3": "value_3", "key_4": "value_4", "key_5": "value_5"}
    elif param == 3:
        method_keys = {
            "method_": "exclude",
            "keys_": ["key_1", "key_2", "key_3", "key_4", "key_5"],
        }
        dict_ = {}
    elif param == 4:
        method_keys = {"method_": "exclude", "keys_": []}
        dict_ = {
            "key_1": "value_1",
            "key_2": "value_2",
            "key_3": "value_3",
            "key_4": "value_4",
            "key_5": "value_5",
        }
    elif param == 5:
        method_keys = {"method_": "include", "keys_": ["key_1"]}
        dict_ = {"key_1": "value_1"}
    elif param == 6:
        method_keys = {"method_": "include", "keys_": ["key_1", "key_2"]}
        dict_ = {"key_1": "value_1", "key_2": "value_2"}
    elif param == 7:
        method_keys = {
            "method_": "include",
            "keys_": ["key_1", "key_2", "key_3", "key_4", "key_5"],
        }
        dict_ = {
            "key_1": "value_1",
            "key_2": "value_2",
            "key_3": "value_3",
            "key_4": "value_4",
            "key_5": "value_5",
        }
    elif param == 8:
        method_keys = {"method_": "include", "keys_": []}
        dict_ = {}
    return {"method_keys": method_keys, "data_out": dict_}
