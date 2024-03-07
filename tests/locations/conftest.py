from unittest import mock

import pytest
import requests


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
