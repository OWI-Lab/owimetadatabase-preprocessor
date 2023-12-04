import json
from typing import Any, Callable, Dict, List, Union
from unittest import mock

import pandas as pd
import pandas.testing as pd_testing
import pytest
import requests

from owimetadatabase_preprocessor.geometry.io import GeometryAPI


@pytest.fixture
def api_root() -> str:
    return "https://test.api/test"


@pytest.fixture
def header() -> Dict[str, str]:
    return {"Authorization": "Token 12345"}


@pytest.fixture
def data_subassemblies(request) -> List[Dict[str, object]]:
    if request.param is not None:
        params = request.param
    data_original = [
        {"id": 11, "project": "Nobelwind", "type": "TW", "turbine": "BBK01"},
        {"id": 11, "project": "Nobelwind", "type": "TP", "turbine": "BBK01"},
        {"id": 11, "project": "Nobelwind", "type": "MP", "turbine": "BBK01"},
        {"id": 21, "project": "Nobelwind", "type": "TW", "turbine": "BBK02"},
        {"id": 21, "project": "Nobelwind", "type": "TP", "turbine": "BBK02"},
        {"id": 21, "project": "Nobelwind", "type": "MP", "turbine": "BBK02"},
        {"id": 31, "project": "Another", "type": "TW", "turbine": "GGP02"},
    ]
    if params == "empty":
        data = data_original
    elif params == "project":
        data = data_original[:6]
    elif params == "asset" or params == "project_asset":
        data = data_original[:3]
    elif params == "subassembly":
        data = [data_original[i] for i in [0, 3, 6]]
    elif params == "project_subassembly":
        data = [data_original[i] for i in [0, 3]]
    elif params == "asset_subassembly" or params == "project_asset_subassembly":
        data = [data_original[0]]
    return data


@pytest.fixture
def data_subassemblies_call() -> Callable[[str], Any]:
    def data_gen(params: str):
        data_original = [
            {"id": 11, "project": "Nobelwind", "type": "TW", "turbine": "BBK01"},
            {"id": 11, "project": "Nobelwind", "type": "TP", "turbine": "BBK01"},
            {"id": 11, "project": "Nobelwind", "type": "MP", "turbine": "BBK01"},
            {"id": 21, "project": "Nobelwind", "type": "TW", "turbine": "BBK02"},
            {"id": 21, "project": "Nobelwind", "type": "TP", "turbine": "BBK02"},
            {"id": 21, "project": "Nobelwind", "type": "MP", "turbine": "BBK02"},
            {"id": 31, "project": "Another", "type": "TW", "turbine": "GGP02"},
        ]
        if params == "empty":
            data = data_original
        elif params == "project":
            data = data_original[:6]
        elif params == "asset" or params == "project_asset":
            data = data_original[:3]
        elif params == "subassembly":
            data = [data_original[i] for i in [0, 3, 6]]
        elif params == "project_subassembly":
            data = [data_original[i] for i in [0, 3]]
        elif params == "asset_subassembly" or params == "project_asset_subassembly":
            data = [data_original[0]]
        return data

    return data_gen


@pytest.fixture
def data_buildingblocks(request) -> List[Dict[str, object]]:
    if request.param is not None:
        params = request.param
    data_original = [
        {"id": 1, "project": "Nobelwind", "subassembly_id": 235, "title": "BBK01_TW_1"},
        {"id": 2, "project": "Nobelwind", "subassembly_id": 235, "title": "BBK01_TW_2"},
        {"id": 3, "project": "Nobelwind", "subassembly_id": 235, "title": "BBK01_TW_3"},
        {"id": 4, "project": "Nobelwind", "subassembly_id": 236, "title": "BBK01_TP_1"},
        {"id": 5, "project": "Nobelwind", "subassembly_id": 237, "title": "BBK01_MP_1"},
        {"id": 6, "project": "Nobelwind", "subassembly_id": 238, "title": "BBK02_TW_1"},
        {"id": 7, "project": "Another", "subassembly_id": 355, "title": "GGP05_TW_1"},
    ]
    if params == "empty":
        data = data_original
    elif params == "project":
        data = data_original[:6]
    elif params == "asset" or params == "project_asset":
        data = data_original[:5]
    elif params == "subassembly":
        data = [data_original[i] for i in [0, 1, 2, 5, 6]]
    elif (
        params == "id"
        or params == "project_id"
        or params == "asset_id"
        or params == "subassembly_id"
        or params == "project_subassembly_id"
        or params == "asset_subassembly_id"
        or params == "project_asset_subassembly_id"
        or params == "project_asset_id"
        or params == "project_subassembly"
        or params == "asset_subassembly"
        or params == "project_asset_subassembly"
    ):
        data = data_original[:3]
    return data


@pytest.fixture
def data_buildingblocks_call() -> Callable[[str], Any]:
    def data_gen(params: str):
        data_original = [
            {
                "id": 1,
                "project": "Nobelwind",
                "subassembly_id": 235,
                "title": "BBK01_TW_1",
            },
            {
                "id": 2,
                "project": "Nobelwind",
                "subassembly_id": 235,
                "title": "BBK01_TW_2",
            },
            {
                "id": 3,
                "project": "Nobelwind",
                "subassembly_id": 235,
                "title": "BBK01_TW_3",
            },
            {
                "id": 4,
                "project": "Nobelwind",
                "subassembly_id": 236,
                "title": "BBK01_TP_1",
            },
            {
                "id": 5,
                "project": "Nobelwind",
                "subassembly_id": 237,
                "title": "BBK01_MP_1",
            },
            {
                "id": 6,
                "project": "Nobelwind",
                "subassembly_id": 238,
                "title": "BBK02_TW_1",
            },
            {
                "id": 7,
                "project": "Another",
                "subassembly_id": 355,
                "title": "GGP05_TW_1",
            },
        ]
        if params == "empty":
            data = data_original
        elif params == "project":
            data = data_original[:6]
        elif params == "asset" or params == "project_asset":
            data = data_original[:5]
        elif params == "subassembly":
            data = [data_original[i] for i in [0, 1, 2, 5, 6]]
        elif (
            params == "id"
            or params == "project_id"
            or params == "asset_id"
            or params == "subassembly_id"
            or params == "project_subassembly_id"
            or params == "asset_subassembly_id"
            or params == "project_asset_subassembly_id"
            or params == "project_asset_id"
            or params == "project_subassembly"
            or params == "asset_subassembly"
            or params == "project_asset_subassembly"
        ):
            data = data_original[:3]
        return data

    return data_gen


@pytest.fixture
def mock_requests_get_materials(mocker: mock.Mock) -> mock.Mock:
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


@pytest.fixture
def mock_requests_get_subassemblies(
    mocker: mock.Mock, data_subassemblies_call: Callable[[str], Any]
) -> mock.Mock:
    def custom_side_effect(*args, **kwargs) -> requests.Response:
        resp = requests.Response()
        resp.status_code = 200
        if not kwargs.get("params"):
            data = data_subassemblies_call("empty")
        else:
            if kwargs.get("params") == {"asset__projectsite__title": "Nobelwind"}:
                data = data_subassemblies_call("project")
            elif kwargs.get("params") == {"asset__title": "BBK01"}:
                data = data_subassemblies_call("asset")
            elif kwargs.get("params") == {"subassembly_type": "TW"}:
                data = data_subassemblies_call("subassembly")
            elif kwargs.get("params") == {
                "asset__projectsite__title": "Nobelwind",
                "asset__title": "BBK01",
            }:
                data = data_subassemblies_call("project_asset")
            elif kwargs.get("params") == {
                "asset__projectsite__title": "Nobelwind",
                "subassembly_type": "TW",
            }:
                data = data_subassemblies_call("project_subassembly")
            elif kwargs.get("params") == {
                "asset__title": "BBK01",
                "subassembly_type": "TW",
            }:
                data = data_subassemblies_call("asset_subassembly")
            elif kwargs.get("params") == {
                "asset__projectsite__title": "Nobelwind",
                "asset__title": "BBK01",
                "subassembly_type": "TW",
            }:
                data = data_subassemblies_call("project_asset_subassembly")
            else:
                data = data_subassemblies_call("empty")
        resp._content = json.dumps(data).encode("utf-8")
        return resp

    mock = mocker.patch("requests.get", side_effect=custom_side_effect)
    return mock


@pytest.mark.parametrize(
    "params, data_subassemblies",
    [
        ({}, "empty"),
        ({"projectsite": "Nobelwind"}, "project"),
        ({"assetlocation": "BBK01"}, "asset"),
        ({"subassembly_type": "TW"}, "subassembly"),
        (
            {"projectsite": "Nobelwind", "assetlocation": "BBK01"},
            "project_asset",
        ),
        (
            {"projectsite": "Nobelwind", "subassembly_type": "TW"},
            "project_subassembly",
        ),
        (
            {"assetlocation": "BBK01", "subassembly_type": "TW"},
            "asset_subassembly",
        ),
        (
            {
                "projectsite": "Nobelwind",
                "assetlocation": "BBK01",
                "subassembly_type": "TW",
            },
            "project_asset_subassembly",
        ),
    ],
    indirect=["data_subassemblies"],
)
def test_get_subassemblies(
    api_root: str,
    header: Dict[str, str],
    mock_requests_get_subassemblies: mock.Mock,
    params: Union[Dict[str, Union[str, int]], None],
    data_subassemblies: Union[Dict[str, str], None],
) -> None:
    api_test = GeometryAPI(api_root, header)
    data_ = api_test.get_subassemblies(**params)  # type: ignore
    expected_data = pd.DataFrame(data_subassemblies)
    assert isinstance(data_["data"], pd.DataFrame)
    assert isinstance(data_["exists"], bool)
    assert data_["exists"]
    pd_testing.assert_frame_equal(data_["data"], expected_data)


@pytest.fixture
def mock_requests_get_buildingblocks(
    mocker: mock.Mock, data_buildingblocks_call: Callable[[str], Any]
) -> mock.Mock:
    def custom_side_effect(*args, **kwargs) -> requests.Response:
        resp = requests.Response()
        resp.status_code = 200
        if not kwargs.get("params"):
            data = data_buildingblocks_call("empty")
        else:
            if kwargs.get("params") == {
                "sub_assembly__asset__projectsite__title": "Nobelwind"
            }:
                data = data_buildingblocks_call("project")
            elif kwargs.get("params") == {"sub_assembly__asset__title": "BBK01"}:
                data = data_buildingblocks_call("asset")
            elif kwargs.get("params") == {"sub_assembly__subassembly_type": "TW"}:
                data = data_buildingblocks_call("subassembly")
            elif kwargs.get("params") == {"sub_assembly__id": 235}:
                data = data_buildingblocks_call("id")
            elif kwargs.get("params") == {
                "sub_assembly__asset__projectsite__title": "Nobelwind",
                "sub_assembly__asset__title": "BBK01",
            }:
                data = data_buildingblocks_call("project_asset")
            elif kwargs.get("params") == {
                "sub_assembly__asset__projectsite__title": "Nobelwind",
                "sub_assembly__subassembly_type": "TW",
            }:
                data = data_buildingblocks_call("project_subassembly")
            elif kwargs.get("params") == {
                "sub_assembly__asset__title": "BBK01",
                "sub_assembly__subassembly_type": "TW",
            }:
                data = data_buildingblocks_call("asset_subassembly")
            elif kwargs.get("params") == {
                "sub_assembly__asset__projectsite__title": "Nobelwind",
                "sub_assembly__asset__title": "BBK01",
                "sub_assembly__subassembly_type": "TW",
            }:
                data = data_buildingblocks_call("project_asset_subassembly")
            elif kwargs.get("params") == {
                "sub_assembly__asset__projectsite__title": "Nobelwind",
                "sub_assembly__asset__title": "BBK01",
                "sub_assembly__subassembly_type": "TW",
                "sub_assembly__id": 235,
            }:
                data = data_buildingblocks_call("project_asset_subassembly_id")
            elif kwargs.get("params") == {
                "sub_assembly__asset__projectsite__title": "Nobelwind",
                "sub_assembly__id": 235,
            }:
                data = data_buildingblocks_call("project_id")
            elif kwargs.get("params") == {
                "sub_assembly__asset__title": "BBK01",
                "sub_assembly__id": 235,
            }:
                data = data_buildingblocks_call("asset_id")
            elif kwargs.get("params") == {
                "sub_assembly__subassembly_type": "TW",
                "sub_assembly__id": 235,
            }:
                data = data_buildingblocks_call("subassembly_id")
            elif kwargs.get("params") == {
                "sub_assembly__asset__projectsite__title": "Nobelwind",
                "sub_assembly__subassembly_type": "TW",
                "sub_assembly__id": 235,
            }:
                data = data_buildingblocks_call("project_subassembly_id")
            elif kwargs.get("params") == {
                "sub_assembly__asset__title": "BBK01",
                "sub_assembly__subassembly_type": "TW",
                "sub_assembly__id": 235,
            }:
                data = data_buildingblocks_call("asset_subassembly_id")
            elif kwargs.get("params") == {
                "sub_assembly__asset__projectsite__title": "Nobelwind",
                "sub_assembly__asset__title": "BBK01",
                "sub_assembly__id": 235,
            }:
                data = data_buildingblocks_call("project_asset_id")
            else:
                data = data_buildingblocks_call("empty")
        resp._content = json.dumps(data).encode("utf-8")
        return resp

    mock = mocker.patch("requests.get", side_effect=custom_side_effect)
    return mock


@pytest.mark.parametrize(
    "params, data_buildingblocks",
    [
        ({}, "empty"),
        ({"projectsite": "Nobelwind"}, "project"),
        ({"assetlocation": "BBK01"}, "asset"),
        ({"subassembly_type": "TW"}, "subassembly"),
        ({"subassembly_id": 235}, "id"),
        (
            {"projectsite": "Nobelwind", "assetlocation": "BBK01"},
            "project_asset",
        ),
        (
            {"projectsite": "Nobelwind", "subassembly_type": "TW"},
            "project_subassembly",
        ),
        (
            {"assetlocation": "BBK01", "subassembly_type": "TW"},
            "asset_subassembly",
        ),
        (
            {
                "projectsite": "Nobelwind",
                "assetlocation": "BBK01",
                "subassembly_type": "TW",
            },
            "project_asset_subassembly",
        ),
        (
            {"projectsite": "Nobelwind", "subassembly_id": 235},
            "project_id",
        ),
        (
            {"assetlocation": "BBK01", "subassembly_id": 235},
            "asset_id",
        ),
        (
            {"subassembly_type": "TW", "subassembly_id": 235},
            "subassembly_id",
        ),
        (
            {
                "projectsite": "Nobelwind",
                "subassembly_type": "TW",
                "subassembly_id": 235,
            },
            "project_subassembly_id",
        ),
        (
            {"assetlocation": "BBK01", "subassembly_type": "TW", "subassembly_id": 235},
            "asset_subassembly_id",
        ),
        (
            {
                "projectsite": "Nobelwind",
                "assetlocation": "BBK01",
                "subassembly_type": "TW",
                "subassembly_id": 235,
            },
            "project_asset_subassembly_id",
        ),
        (
            {
                "projectsite": "Nobelwind",
                "assetlocation": "BBK01",
                "subassembly_id": 235,
            },
            "project_asset_id",
        ),
    ],
    indirect=["data_buildingblocks"],
)
def test_get_buildingblocks(
    api_root: str,
    header: Dict[str, str],
    mock_requests_get_buildingblocks: mock.Mock,
    params: Union[Dict[str, Union[str, int]], None],
    data_buildingblocks: Union[Dict[str, str], None],
) -> None:
    api_test = GeometryAPI(api_root, header)
    data_ = api_test.get_buildingblocks(**params)  # type: ignore
    expected_data = pd.DataFrame(data_buildingblocks)
    assert isinstance(data_["data"], pd.DataFrame)
    assert isinstance(data_["exists"], bool)
    assert data_["exists"]
    pd_testing.assert_frame_equal(data_["data"], expected_data)


def test_get_materials(
    api_root: str, header: Dict[str, str], mock_requests_get_materials: mock.Mock
) -> None:
    api_test = GeometryAPI(api_root, header)
    data = api_test.get_materials()
    assert isinstance(data["data"], pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["exists"]
