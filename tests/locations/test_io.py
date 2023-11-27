import pandas as pd
import pytest
import requests

from owimetadatabase_preprocessor.locations.io import LocationsAPI


@pytest.fixture
def api_root():
    return "https://test.api/test"


@pytest.fixture
def header():
    header = {"Authorization": "Token 12345"}


@pytest.fixture
def mock_requests_get_projectsites(mocker):
    mock = mocker.patch("requests.get")
    def response():
        resp = requests.Response()
        resp.status_code = 200
        resp._content = (
            b'[{"col_1": "11", "col_2": "12", "col_3": "13"}, '
            b'{"col_1": "21", "col_2": "22", "col_3": "23"}]'
        )
        return resp
    mock.return_value = response()
    return mock


def test_get_projectsites(api_root, header, mock_requests_get_projectsites) -> None:
    api_test = LocationsAPI(api_root, header)
    data = api_test.get_projectsites()
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["exists"] == True


@pytest.fixture
def mock_requests_get_projectsite_detail(mocker):
    mock = mocker.patch("requests.get")
    def response():
        resp = requests.Response()
        resp.status_code = 200
        resp._content = (
            b'[{"id": "239", "col_2": "text 1", "col_3": "text 2"}]'
        )
        return resp
    mock.return_value = response()
    return mock


def test_get_projectsite_detail(api_root, header, mock_requests_get_projectsite_detail) -> None:
    api_test = LocationsAPI(api_root, header)
    data = api_test.get_projectsite_detail(projectsite="Nobelwind")
    assert isinstance(data["id"], int)
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["id"][0] == 239
    assert data["exists"] == True


@pytest.fixture
def mock_requests_get_assetlocations(mocker):
    def custom_side_effect(*args, **kwargs):
        if kwargs.get('params') == {'projectsite__title': 'Nobelwind'}:
            resp = requests.Response()
            resp.status_code = 200
            resp._content = (
                b'[{"id": "11", "project": "Nobelwind", "col_3": "13"}, '
                b'{"id": "21", "project": "Nobelwind", "col_3": "23"}]'
            )
            return resp
        else:
            resp = requests.Response()
            resp.status_code = 200
            resp._content = (
                b'[{"id": "11", "project": "Nobelwind", "col_3": "13"}, '
                b'{"id": "21", "project": "Nobelwind", "col_3": "23"}, '
                b'{"id": "31", "project": "Another", "col_3": "33"}]'
            )
            return resp
    mock = mocker.patch("requests.get", side_effect=custom_side_effect)
    return mock


def test_get_assetlocations_single(api_root, header, mock_requests_get_assetlocations) -> None:
    api_test = LocationsAPI(api_root, header)
    data = api_test.get_assetlocations(projectsite="Nobelwind")
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["exists"] == True
    assert data["data"].__len__ == 2
    assert data["data"]["project"][0] == "Nobelwind"
    assert data["data"]["project"][1] == "Nobelwind"


def test_get_assetlocations_single(api_root, header, mock_requests_get_assetlocations) -> None:
    api_test = LocationsAPI(api_root, header)
    data = api_test.get_assetlocations()
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["exists"] == True
    assert data["data"].__len__ == 1
    assert data["data"]["project"][0] == "Another"


def test_get_assetlocation_detail(api_root, header, mock_requests_get_projectsite_detail) -> None:
    api_test = LocationsAPI(api_root, header)
    data = api_test.get_assetlocation_detail(projectsite="Nobelwind", assetlocation="BBK01")
    assert isinstance(data["id"], int)
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["id"][0] == 239
    assert data["exists"] == True
