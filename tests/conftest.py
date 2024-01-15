from typing import Dict

import pytest


@pytest.fixture(scope="module")
def api_root() -> str:
    return "https://test.api/test"


@pytest.fixture(scope="module")
def header() -> Dict[str, str]:
    return {"Authorization": "Token 12345"}


@pytest.fixture(scope="function")
def dict_in() -> Dict[str, str]:
    return {
        "key_1": "value_1",
        "key_2": "value_2",
        "key_3": "value_3",
        "key_4": "value_4",
        "key_5": "value_5",
    }


@pytest.fixture(scope="function", params=[1, 2, 3, 4, 5, 6, 7, 8])
def dict_out(request) -> Dict[str, str]:
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
