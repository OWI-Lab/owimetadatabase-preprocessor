from unittest import mock

import numpy as np
import pandas as pd

from owimetadatabase_preprocessor.locations.io import LocationsAPI


def test_get_projectsites(api_root: str, header: dict[str, str], mock_requests_get_advanced: mock.Mock) -> None:
    api_test = LocationsAPI(api_root, header=header)
    data = api_test.get_projectsites()
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["exists"]


def test_get_projectsite_detail(
    api_root: str,
    header: dict[str, str],
    mock_requests_get_projectsite_detail: mock.Mock,
) -> None:
    api_test = LocationsAPI(api_root, header=header)
    data = api_test.get_projectsite_detail(projectsite="Nobelwind")
    assert isinstance(data["id"], np.int64)
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["id"] == 239
    assert data["exists"]


def test_get_assetlocations_single(api_root: str, header: dict[str, str], mock_requests_get_assetlocations: mock.Mock) -> None:
    api_test = LocationsAPI(api_root, header=header)
    data = api_test.get_assetlocations(projectsite="Nobelwind")
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["exists"]
    assert data["data"].__len__() == 2
    assert data["data"]["project"][0] == "Nobelwind"
    assert data["data"]["project"][1] == "Nobelwind"


def test_get_assetlocations_all(
    api_root: str,
    header: dict[str, str],
    mock_requests_get_assetlocations: mock.Mock,
) -> None:
    api_test = LocationsAPI(api_root, header=header)
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
    header: dict[str, str],
    mock_requests_get_projectsite_detail: mock.Mock,
) -> None:
    api_test = LocationsAPI(api_root, header=header)
    data = api_test.get_assetlocation_detail(projectsite="Nobelwind", assetlocation="BBK01")
    assert isinstance(data["id"], np.int64)
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["id"] == 239
    assert data["exists"]
