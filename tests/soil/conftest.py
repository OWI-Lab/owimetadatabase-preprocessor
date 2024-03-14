import json
from typing import Dict, List, Union
from unittest import mock

import pandas as pd
import pytest
import requests

from owimetadatabase_preprocessor.soil.io import SoilAPI


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


@pytest.fixture(scope="function")
def api_soil(api_root, header):
    return SoilAPI(api_root, header=header)


@pytest.fixture(scope="function")
def data_search() -> List[Dict[str, Union[str, float]]]:
    return [
        {
            "id": 1,
            "easting": 2.1,
            "northing": 50.1,
            "title": "CPT_1",
        },
        {
            "id": 2,
            "easting": 2.15,
            "northing": 51.1,
            "title": "CPT_2",
        },
        {
            "id": 3,
            "easting": 1.95,
            "northing": 49.75,
            "title": "CPT_3",
        },
    ]


@pytest.fixture(scope="function")
def data_gather() -> List[Dict[str, Union[str, float]]]:
    return [
        {
            "id": 1,
            "easting": 2.1,
            "northing": 50.1,
            "title": "CPT_1",
            "easting [m]": 435633.98547013174,
            "northing [m]": 5550137.09550342,
            "offset [m]": 13224.871758864,
        },
        {
            "id": 2,
            "easting": 2.15,
            "northing": 51.1,
            "title": "CPT_2",
            "easting [m]": 440484.60093828203,
            "northing [m]": 5661288.939505116,
            "offset [m]": 122781.865111092,
        },
        {
            "id": 3,
            "easting": 1.95,
            "northing": 49.75,
            "title": "CPT_3",
            "easting [m]": 424360.77288160764,
            "northing [m]": 5511364.170916873,
            "offset [m]": 28028.623819221,
        },
    ]


@pytest.fixture(scope="function")
def data_gather_sorted() -> List[Dict[str, Union[str, float]]]:
    return [
        {
            "id": 1,
            "easting": 2.1,
            "northing": 50.1,
            "title": "CPT_1",
            "easting [m]": 435633.98547013174,
            "northing [m]": 5550137.09550342,
            "offset [m]": 13224.871758864,
        },
        {
            "id": 3,
            "easting": 1.95,
            "northing": 49.75,
            "title": "CPT_3",
            "easting [m]": 424360.77288160764,
            "northing [m]": 5511364.170916873,
            "offset [m]": 28028.623819221,
        },
        {
            "id": 2,
            "easting": 2.15,
            "northing": 51.1,
            "title": "CPT_2",
            "easting [m]": 440484.60093828203,
            "northing [m]": 5661288.939505116,
            "offset [m]": 122781.865111092,
        },
    ]


@pytest.fixture(scope="function")
def mock_requests_search_any_entity(data_search, mocker: mock.Mock) -> mock.Mock:

    def custom_side_effect(*args, **kwargs) -> requests.Response:
        resp = requests.Response()
        if (
            kwargs.get("params")["latitude"] == 50.0
            and kwargs.get("params")["longitude"] == 2.0
            and kwargs.get("params")["offset"] >= 10.0
        ):
            data = data_search
            resp.status_code = 200
        elif (
            kwargs.get("params")["latitude"] == 50.0
            and kwargs.get("params")["longitude"] == 2.0
            and kwargs.get("params")["offset"] >= 1.0
            and kwargs.get("params")["offset"] <= 10.0
        ):
            data = [data_search[i] for i in [0, 2]]
            resp.status_code = 200
        elif (
            kwargs.get("params")["latitude"] == 50.0
            and kwargs.get("params")["longitude"] == 2.0
            and kwargs.get("params")["offset"] <= 1.0
        ):
            data = []
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


@pytest.fixture(scope="function")
def close_entity_true(request, data_search):
    if request.param is not None:
        idx = request.param
    if idx == 0:
        return pd.DataFrame([data_search[i] for i in [0, 2]])
    elif idx == 1:
        return pd.DataFrame(data_search)
    elif idx == 2:
        return pd.DataFrame([data_search[i] for i in [0, 2]])


@pytest.fixture(scope="function")
def df_gathered_inp(request, data_gather):
    if request.param is not None:
        idx = request.param
    if idx == "regular":
        return pd.DataFrame(data_gather, index=[0, 1, 2])
    elif idx == "single":
        return pd.DataFrame(data_gather[0], index=[0])


@pytest.fixture(scope="function")
def dict_gathered_true(request, data_gather, data_gather_sorted):
    if request.param is not None:
        idx = request.param
    if idx == "regular":
        return {
            "data": pd.DataFrame(data_gather_sorted, index=[0, 2, 1]),
            "id": 1,
            "title": "CPT_1",
            "offset [m]": 13224.871758864,
        }
    elif idx == "single":
        return {
            "data": pd.DataFrame(data_gather[0], index=[0]),
            "id": 1,
            "title": "CPT_1",
            "offset [m]": 13224.871758864,
        }


@pytest.fixture(scope="function")
def dict_gathered_final_true(data_gather_sorted):
    return {
        "data": pd.DataFrame(data_gather_sorted, index=[0, 2, 1]),
        "id": 1,
        "title": "CPT_1",
        "offset [m]": 13224.871758862186,
    }
