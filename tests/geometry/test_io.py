from typing import Dict, Union
from unittest import mock

import pandas as pd
import pandas.testing as pd_testing
import pytest

from owimetadatabase_preprocessor.geometry.io import GeometryAPI


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


@pytest.mark.parametrize(
    "params, data_buildingblocks",
    [
        ({}, "empty"),
        ({"projectsite": "Nobelwind"}, "project"),
        ({"assetlocation": "BBK01"}, "asset"),
        ({"subassembly_type": "TW"}, "subassembly"),
        ({"subassembly_id": str(235)}, "id"),
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


def test_get_owt_geometry_processor(
    api_root: str,
    header: Dict[str, str],
    mat: pd.DataFrame,
    sa: pd.DataFrame,
    OWT_mock: mock.Mock,
    mock_requests_get_materials: mock.Mock,
    mock_requests_get_subassemblies: mock.Mock,
) -> None:
    with mock.patch(
        "owimetadatabase_preprocessor.geometry.io.OWT", return_value=OWT_mock
    ):
        api_test = GeometryAPI(api_root, header)
        processor = api_test.get_owt_geometry_processor(
            turbine="BBK01", tower_base=10.0, monopile_head=5.0
        )
        assert isinstance(processor, type(OWT_mock))
        assert isinstance(processor.api, GeometryAPI)
        assert isinstance(processor.materials, pd.DataFrame)
        assert isinstance(processor.sub_assemblies, pd.DataFrame)
        assert isinstance(processor.tower_base, float)
        assert isinstance(processor.monopile_head, float)
        #  assert isinstance(processor.sub_assemblies, Dict[str, List[SubAssembly]])
        pd_testing.assert_frame_equal(processor.materials, mat)
        pd_testing.assert_frame_equal(processor.sub_assemblies, sa)
