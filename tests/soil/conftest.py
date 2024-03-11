import json
from unittest import mock

import pytest
import requests


@pytest.fixture(scope="function")
def soil_init(header):
    return {
        "api_root": "https://owimetadatabase.owilab.be/api/v1/soildata/",
        "header": header,
        "auth": None,
        "uname": None,
        "password": None,
    }


@pytest.fixture(scope="function")
def mock_requests_get_proximity_entities_2d(mocker: mock.Mock) -> mock.Mock:

    def custom_side_effect(*args, **kwargs) -> requests.Response:
        resp = requests.Response()
        if kwargs.get("params") == {
            "latitude": 50.1,
            "longitude": 2.22,
            "offset": 0.75,
        }:
            data = [
                {
                    "col_1": 1,
                    "col_2": 49.5,
                    "col_3": True,
                    "col_4": "test_1",
                    "col_5": {"test_1": 1.0},
                    "col_6": None,
                },
                {
                    "col_1": 2,
                    "col_2": 51.1,
                    "col_3": False,
                    "col_4": "test_2",
                    "col_5": {"test_2": 2.0},
                    "col_6": None,
                },
            ]
            resp.status_code = 200
        else:
            data = []
            resp.status_code = 404
            resp.reason = (
                "The object corresponding to the specified parameters is not found"
            )
        resp._content = json.dumps(data).encode("utf-8")
        return resp

    mock = mocker.patch("requests.get", side_effect=custom_side_effect)
    return mock
