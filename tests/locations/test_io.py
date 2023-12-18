from typing import Dict
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import requests

from owimetadatabase_preprocessor.locations.io import LocationsAPI


@pytest.fixture
def mock_requests_get_projectsites(mocker: mock.Mock) -> mock.Mock:
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


def test_get_projectsites(
    api_root: str, header: Dict[str, str], mock_requests_get_projectsites: mock.Mock
) -> None:
    api_test = LocationsAPI(api_root, header)
    data = api_test.get_projectsites()
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["exists"]


@pytest.fixture
def mock_requests_get_projectsite_detail(mocker: mock.Mock) -> mock.Mock:
    mock = mocker.patch("requests.get")

    def response() -> requests.Response:
        resp = requests.Response()
        resp.status_code = 200
        resp._content = ('[{"id": 239, "col_2": "text 1", "col_3": "text 2"}]').encode(
            "utf-8"
        )
        return resp

    mock.return_value = response()
    return mock


def test_get_projectsite_detail(
    api_root: str,
    header: Dict[str, str],
    mock_requests_get_projectsite_detail: mock.Mock,
) -> None:
    api_test = LocationsAPI(api_root, header)
    data = api_test.get_projectsite_detail(projectsite="Nobelwind")
    assert isinstance(data["id"], np.int64)
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["id"] == 239
    assert data["exists"]


@pytest.fixture
def mock_requests_get_assetlocations(mocker: mock.Mock) -> mock.Mock:
    def custom_side_effect(*args, **kwargs) -> requests.Response:
        if kwargs.get("params") == {"projectsite__title": "Nobelwind"}:
            resp = requests.Response()
            resp.status_code = 200
            resp._content = (
                b'[{"id": 11, "project": "Nobelwind", "col_3": 13}, '
                b'{"id": 21, "project": "Nobelwind", "col_3": 23}]'
            )
        else:
            resp = requests.Response()
            resp.status_code = 200
            resp._content = (
                b'[{"id": 11, "project": "Nobelwind", "col_3": 13}, '
                b'{"id": 21, "project": "Nobelwind", "col_3": 23}, '
                b'{"id": 31, "project": "Another", "col_3": 33}]'
            )
        return resp

    mock = mocker.patch("requests.get", side_effect=custom_side_effect)
    return mock


def test_get_assetlocations_single(
    api_root: str, header: Dict[str, str], mock_requests_get_assetlocations: mock.Mock
) -> None:
    api_test = LocationsAPI(api_root, header)
    data = api_test.get_assetlocations(projectsite="Nobelwind")
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["exists"]
    assert data["data"].__len__() == 2
    assert data["data"]["project"][0] == "Nobelwind"
    assert data["data"]["project"][1] == "Nobelwind"


def test_get_assetlocations_all(
    api_root: str,
    header: Dict[str, str],
    mock_requests_get_assetlocations: mock.Mock,
) -> None:
    api_test = LocationsAPI(api_root, header)
    data = api_test.get_assetlocations()
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["exists"]
    assert data["data"].__len__() == 3
    assert data["data"]["project"][0] == "Nobelwind"
    assert data["data"]["project"][1] == "Nobelwind"
    assert data["data"]["project"][2] == "Another"


def test_get_assetlocation_detail(
    api_root: str,
    header: Dict[str, str],
    mock_requests_get_projectsite_detail: mock.Mock,
) -> None:
    api_test = LocationsAPI(api_root, header)
    data = api_test.get_assetlocation_detail(
        projectsite="Nobelwind", assetlocation="BBK01"
    )
    assert isinstance(data["id"], np.int64)
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["id"] == 239
    assert data["exists"]
