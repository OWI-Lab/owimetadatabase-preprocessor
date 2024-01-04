import json

from typing import  Any, Callable, Dict, List, Tuple, Union
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import requests

from owimetadatabase_preprocessor.geometry.io import GeometryAPI
from owimetadatabase_preprocessor.geometry.processing import OWT
from owimetadatabase_preprocessor.geometry.structures import Material, Position, BuildingBlock, SubAssembly


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
            elif kwargs.get("params") == {"sub_assembly__id": "235"}:
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
                "sub_assembly__id": "235",
            }:
                data = data_buildingblocks_call("project_asset_subassembly_id")
            elif kwargs.get("params") == {
                "sub_assembly__asset__projectsite__title": "Nobelwind",
                "sub_assembly__id": "235",
            }:
                data = data_buildingblocks_call("project_id")
            elif kwargs.get("params") == {
                "sub_assembly__asset__title": "BBK01",
                "sub_assembly__id": "235",
            }:
                data = data_buildingblocks_call("asset_id")
            elif kwargs.get("params") == {
                "sub_assembly__subassembly_type": "TW",
                "sub_assembly__id": "235",
            }:
                data = data_buildingblocks_call("subassembly_id")
            elif kwargs.get("params") == {
                "sub_assembly__asset__projectsite__title": "Nobelwind",
                "sub_assembly__subassembly_type": "TW",
                "sub_assembly__id": "235",
            }:
                data = data_buildingblocks_call("project_subassembly_id")
            elif kwargs.get("params") == {
                "sub_assembly__asset__title": "BBK01",
                "sub_assembly__subassembly_type": "TW",
                "sub_assembly__id": "235",
            }:
                data = data_buildingblocks_call("asset_subassembly_id")
            elif kwargs.get("params") == {
                "sub_assembly__asset__projectsite__title": "Nobelwind",
                "sub_assembly__asset__title": "BBK01",
                "sub_assembly__id": "235",
            }:
                data = data_buildingblocks_call("project_asset_id")
            else:
                data = data_buildingblocks_call("empty")
        resp._content = json.dumps(data).encode("utf-8")
        return resp

    mock = mocker.patch("requests.get", side_effect=custom_side_effect)
    return mock


@pytest.fixture
def mat() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"col_1": 11, "col_2": 12, "col_3": 13},
            {"col_1": 21, "col_2": 22, "col_3": 23}
        ]
    )


@pytest.fixture
def sa() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": 11, "project": "Nobelwind", "type": "TW", "turbine": "BBK01"},
            {"id": 11, "project": "Nobelwind", "type": "TP", "turbine": "BBK01"},
            {"id": 11, "project": "Nobelwind", "type": "MP", "turbine": "BBK01"},
        ]
    )


@pytest.fixture
def OWT_mock(
    api_root: str,
    header: Dict[str, str],
    mat: pd.DataFrame,
    sa: pd.DataFrame,
) -> mock.Mock:
    
    def OWT_mock_init(self, *args, **kwargs):
        self.api = GeometryAPI(args[0], args[1])
        self.materials = args[2]
        self.sub_assemblies = args[3]
        self.tower_base = 10.0
        self.monopile_head = 5.0

    #  mocked_OWT = mock.create_autospec(OWT, instance=True)
    mocked_OWT = mock.Mock(spec=OWT)
    OWT_mock_init(mocked_OWT, api_root, header, mat, sa)
    #  mocked_OWT.side_effect = lambda *args, **kwargs: OWT_mock_init(mocked_OWT, api_root, header, mat, sa)
    #  mocked_OWT.__init__ = mock.Mock(return_value=None)
    return mocked_OWT


@pytest.fixture(scope="module")
def data_mat_dict() -> Dict[str, Union[str, np.float64]]:
    return {
        "title": "steel",
        "description": "Structural steel",
        "young_modulus": np.float64(210.0),
        "poisson_ratio": np.float64(0.3),
    }


@pytest.fixture(scope="module")
def data_mat(data_mat_dict) -> Dict[str, Union[str, np.int64, np.float64]]:    
    data_mat_dict = dict(data_mat_dict)
    data_mat_dict["id"] = np.int64(1)
    data_mat_dict["density"] = np.float64(7952.0)
    return data_mat_dict


@pytest.fixture(scope="module")
def data_pos() -> Dict[str, Union[str, np.float64]]:
    return {
        "x": np.float64(0.0),
        "y": np.float64(0.0),
        "z": np.float64(17000.0),
        "alpha": np.float64(1.0),
        "beta": np.float64(2.0),
        "gamma": np.float64(3.0),
        "reference_system": "LAT",
    }


@pytest.fixture(scope="module")
def data_bb() -> Dict[str, Union[str, np.int64, np.float64]]:
    return {
        "id": np.int64(1),
        "title": "BBG01_TW_FLANGE",
        "description": "Something 1",
        "x_position": np.float64(0.0),
        "y_position": np.float64(0.0),
        "z_position": np.float64(17000.0),
        "alpha": np.float64(1.0),
        "beta": np.float64(2.0),
        "gamma": np.float64(3.0),
        "vertical_position_reference_system": "LAT",
        "material": np.float64(1.0)
    }


@pytest.fixture(scope="module")
def data_bb_init_no_sa(data_bb, data_pos) -> Dict[str, Union[str, np.int64, np.float64]]:
    return {
        "id": np.int64(1),
        "title": "BBG01_TW_FLANGE",
        "description": "Something 1",
        "json": data_bb,
        "position": data_pos,
    }


@pytest.fixture(scope="module")
def data_bb_init_with_sa(data_bb_init_no_sa, Mat, SA) -> Dict[str, Union[str, np.int64, np.float64]]:
    data_bb_init_with_sa_dict = dict(data_bb_init_no_sa)
    data_bb_init_with_sa_dict["subassembly"] = SA
    data_bb_init_with_sa_dict["material"] = Mat
    return data_bb_init_with_sa_dict
    

@pytest.fixture(scope="module")
def Mat(data_mat) -> mock.Mock:
    
    def Mat_mock_init(self, *args, **kwargs):
        materials = args[0]
        self.title = materials["title"]
        self.description = materials["description"]
        self.young_modulus = materials["young_modulus"]
        self.density = materials["density"]
        self.poisson_ratio = materials["poisson_ratio"]
        self.id = materials["id"]                    

    mocked_Mat = mock.Mock()
    Mat_mock_init(mocked_Mat, data_mat)
    return mocked_Mat


@pytest.fixture(scope="module")
def SA(Mat) -> mock.Mock:
    
    def SA_mock_init(self, *args, **kwargs):
        self.materials = [Mat]

    mocked_SA = mock.Mock()
    SA_mock_init(mocked_SA)
    return mocked_SA


@pytest.fixture(scope="module")
def data_bb_flex(request):
    param = request.param
    data_bb = request.getfixturevalue("data_bb")
    height = np.float64(10.0)
    mass = np.float64(1500.0)
    outer_diameter = np.float64(1000.0)
    outer_diameter_alt = np.float64(1250.0)
    wall_t = np.float64(0.2)
    data_bb_ = dict(data_bb)
    if param == "h":
        data_bb_["height"] = height
        return data_bb_
    elif param == "m":
        data_bb_["mass"] = mass
        data_bb_["height"] = height
        data_bb_["moment_of_inertia_x"] = np.float64(0.1)
        data_bb_["moment_of_inertia_y"] = np.float64(0.2)
        data_bb_["moment_of_inertia_z"] = np.float64(0.3)
        data_bb_["marker"] = {
            "x": data_bb_["x_position"],
            "y": data_bb_["y_position"],
            "z": data_bb_["z_position"],
            "radius": np.float64(round(data_bb_["mass"]) / 10),
            "hovertext": "<br>".join(
                [
                    data_bb_["title"],
                    "Mass: " + str(data_bb_["mass"]) + "kg",
                    "x: " + str(data_bb_["x_position"]),
                    "y: " + str(data_bb_["y_position"]),
                    "z: " + str(data_bb_["z_position"]),
                ]
            ),
        }
        data_bb_["dict"] = {
            "title": data_bb_["title"],
            "x": data_bb_["x_position"],
            "y": data_bb_["y_position"],
            "z": data_bb_["z_position"],
            "OD": "",
            "wall_thickness": None,
            "height": data_bb_["height"],
            "volume": None,
            "mass": data_bb_["mass"],
            "moment_of_inertia": {
                k: data_bb_["moment_of_inertia_" + k] for i, k in zip([0, 1, 2], ["x", "y", "z"])
            },
            "description": data_bb_["description"],
        }
        data_bb_["type"] = "lumped_mass"
        data_bb_["str"] = data_bb_["title"] + " (" + data_bb_["type"] + ")"
        return data_bb_
    elif param == "m_distr":
        data_bb_["mass_distribution"] = mass
        return data_bb_
    elif param == "bot_od":
        data_bb_["bottom_outer_diameter"] = outer_diameter
        data_bb_["top_outer_diameter"] = outer_diameter
        data_bb_["wall_thickness"] = wall_t
        return data_bb_
    elif param == "bot_od_alt":
        data_bb_["bottom_outer_diameter"] = outer_diameter_alt
        data_bb_["top_outer_diameter"] = outer_diameter
        data_bb_["wall_thickness"] = wall_t
        return data_bb_
    elif param == "bot_od_top_nan":
        data_bb_["bottom_outer_diameter"] = outer_diameter
        data_bb_["top_outer_diameter"] = np.nan
        data_bb_["wall_thickness"] = wall_t
        return data_bb_
    elif param == "bot_od_bot_nan":
        data_bb_["bottom_outer_diameter"] = np.nan
        data_bb_["top_outer_diameter"] = outer_diameter
        data_bb_["wall_thickness"] = wall_t
        return data_bb_
    elif param == "m_distr_vh":
        data_bb_["volume_distribution"] = mass
        data_bb_["mass_distribution"] = mass
        data_bb_["height"] = height
        data_bb_["mass_calc"] = np.float64(
            round(data_bb_["mass_distribution"]*data_bb_["height"]/1000)
        )
        data_bb_["volume_calc"] = np.float64(
            round(data_bb_["volume_distribution"]*data_bb_["height"]/1000)
        )
        data_bb_["line"] = {
            "x": [data_bb_["x_position"], data_bb_["x_position"]],
            "y": [data_bb_["y_position"], data_bb_["y_position"]],
            "z": [data_bb_["z_position"], data_bb_["z_position"] + data_bb_["height"]],
            "color": "black",
        }
        data_bb_["dict"] = {
            "title": data_bb_["title"],
            "x": data_bb_["x_position"],
            "y": data_bb_["y_position"],
            "z": data_bb_["z_position"],
            "OD": "",
            "wall_thickness": None,
            "height": data_bb_["height"],
            "volume": data_bb_["volume_calc"],
            "mass": data_bb_["mass_calc"],
            "moment_of_inertia": {"x": None, "y": None, "z": None},
            "description": data_bb_["description"],
        }
        data_bb_["type"] = "distributed_mass"
        data_bb_["str"] = data_bb_["title"] + " (" + data_bb_["type"] + ")"
        return data_bb_
    elif param == "bot_od_h":
        data_bb_["bottom_outer_diameter"] = outer_diameter
        data_bb_["top_outer_diameter"] = outer_diameter
        data_bb_["wall_thickness"] = wall_t
        data_bb_["height"] = height
        data_bb_["volume_calc"] = np.float64(6.2819286701185626e-06)
        data_bb_["density_calc"] = np.float64(7952.0)
        data_bb_["mass_calc"] = np.float64(round(data_bb_["volume_calc"]*data_bb_["density_calc"], 1))
        data_bb_["outline"] = (
            [
                data_bb_["bottom_outer_diameter"] / 2,
                -data_bb_["bottom_outer_diameter"] / 2,
                -data_bb_["top_outer_diameter"] / 2,
                data_bb_["top_outer_diameter"] / 2,
                data_bb_["bottom_outer_diameter"] / 2,
            ],
            [
                data_bb_["z_position"],
                data_bb_["z_position"],
                data_bb_["z_position"] + data_bb_["height"],
                data_bb_["z_position"] + data_bb_["height"],
                data_bb_["z_position"]
            ]
        )
        data_bb_["dict"] = {
            "title": data_bb_["title"],
            "x": data_bb_["x_position"],
            "y": data_bb_["y_position"],
            "z": data_bb_["z_position"],
            "OD": str(round(data_bb_["bottom_outer_diameter"])),
            "wall_thickness": data_bb_["wall_thickness"],
            "height": data_bb_["height"],
            "volume": data_bb_["volume_calc"],
            "mass": data_bb_["mass_calc"],
            "moment_of_inertia": {"x": None, "y": None, "z": None},
            "description": data_bb_["description"],
        }
        data_bb_["type"] = "tubular_section"
        data_bb_["str"] = data_bb_["title"] + " (" + data_bb_["type"] + ")"
        return data_bb_
    else:
        return data_bb
    

@pytest.fixture(scope="module")
def data_sa() -> Dict[str, Union[str, np.int64, np.float64]]:
    return {
        "id": np.int64(651),
        "title": "BBG01_TW",
        "description": "Something 2",
        "x_position": np.float64(0.0),
        "y_position": np.float64(0.0),
        "z_position": np.float64(17000.0),
        "vertical_position_reference_system": "LAT",
        "subassembly_type": "TW",
        "source": "vestas_tower_dwg.pdf",
        "asset": np.int64(341),
        "material": np.float64(1.0)
    }


@pytest.fixture(scope="module")
def data_mat_df(data_mat):
    return pd.DataFrame(data_mat, index=[4])


@pytest.fixture(scope="module")
def data_sa_init(api_root, header, data_mat_df) -> Dict[str, Union[str, np.int64, np.float64]]:
    return {
        "api": GeometryAPI(api_root, header),
        "id": np.int64(651),
        "title": "BBG01_TW",
        "description": "Something 2",
        "position": {"x": np.float64(0.0), "y": np.float64(0.0), "z": np.float64(17000.0), "reference_system": "LAT"},
        "type": "TW",
        "source": "vestas_tower_dwg.pdf",
        "asset": np.int64(341),
        "bb": None,
        "materials": [m.to_dict() for _, m in data_mat_df.iterrows()],
    }


@pytest.fixture(scope="module")
def data_sa_flex(request) -> Dict[str, Union[str, np.int64, np.float64]]:
    param = request.param
    data_sa = request.getfixturevalue("data_sa")
    data_sa_ = dict(data_sa)
    if param == "tw":
        data_sa_["subassembly_type"] = "TW"
        data_sa_["color"] = "grey"
        return data_sa_
    elif param == "tp":
        data_sa_["subassembly_type"] = "TP"
        data_sa_["color"] = "goldenrod"
        return data_sa_
    elif param == "mp":
        data_sa_["subassembly_type"] = "MP"
        data_sa_["color"] = "brown"
        return data_sa_


@pytest.fixture(scope="module")
def data_bb_real() -> List[Dict[str, Union[str, np.int64, np.float64]]]:
    return [
        {
            "id": 13432,
            "description": None,
            "slug": "bbg01_tw_20",
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
            "x_position": 0.0,
            "y_position": 0.0,
            "z_position": 10400.0,
            "vertical_position_reference_system": "SUB",
            "title": "bbg01_tw_20",
            "moment_of_inertia_x": None,
            "moment_of_inertia_y": None,
            "moment_of_inertia_z": None,
            "mass": 10000,
            "sub_assembly": 651,
            "projectsite_name": "Nobelwind",
            "asset_name": "BBG01",
            "subassembly_name": "BBG01_TW",
            "material_name": "S355 WTG TOWER",
            "youngs_modulus": 210.0,
            "density": 7952.0,
            "poissons_ratio": 0.2,
            "bottom_outer_diameter": 4489.7,
            "top_outer_diameter": 4489.7,
            "height": 2830.0,
            "wall_thickness": 23.7,
            "material": 1.0
        },
        {
            "id": 13431,
            "description": None,
            "slug": "bbg01_tw_21",
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
            "x_position": 0.0,
            "y_position": 0.0,
            "z_position": 7570.0,
            "vertical_position_reference_system": "SUB",
            "title": "bbg01_tw_21",
            "moment_of_inertia_x": None,
            "moment_of_inertia_y": None,
            "moment_of_inertia_z": None,
            "mass": 10000,
            "sub_assembly": 651,
            "projectsite_name": "Nobelwind",
            "asset_name": "BBG01",
            "subassembly_name": "BBG01_TW",
            "material_name": "S355 WTG TOWER",
            "youngs_modulus": 210.0,
            "density": 7952.0,
            "poissons_ratio": 0.2,
            "bottom_outer_diameter": 4490.9,
            "top_outer_diameter": 4490.9,
            "height": 2830.0,
            "wall_thickness": 24.9,
            "material": 1.0
        },
        {
            "id": 13430,
            "description": None,
            "slug": "bbg01_tw_22",
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
            "x_position": 0.0,
            "y_position": 0.0,
            "z_position": 5170.0,
            "vertical_position_reference_system": "SUB",
            "title": "bbg01_tw_22",
            "moment_of_inertia_x": None,
            "moment_of_inertia_y": None,
            "moment_of_inertia_z": None,
            "mass": 3137.1,
            "sub_assembly": 651,
            "projectsite_name": "Nobelwind",
            "asset_name": "BBG01",
            "subassembly_name": "BBG01_TW",
            "material_name": "S355 WTG TOWER",
            "youngs_modulus": 210.0,
            "density": 7952.0,
            "poissons_ratio": 0.2,
            "bottom_outer_diameter": 4495.1,
            "top_outer_diameter": 4495.1,
            "height": 2400.0,
            "wall_thickness": 29.1,
            "material": 1.0
        }
    ]


@pytest.fixture(scope="module")
def data_bb_alt() -> List[Dict[str, Union[str, np.int64, np.float64]]]:
    return {}


@pytest.fixture
def mock_requests_get_buildingblocks_sa(mocker: mock.Mock, data_bb_real) -> mock.Mock:
    mock = mocker.patch("requests.get")

    def response() -> requests.Response:
        resp = requests.Response()
        resp.status_code = 200
        resp._content = json.dumps(data_bb_real).encode("utf-8")
        return resp

    mock.return_value = response()
    return mock


@pytest.fixture
def mock_requests_get_buildingblocks_sa_alt(mocker: mock.Mock, data_bb_alt) -> mock.Mock:
    mock = mocker.patch("requests.get")

    def response() -> requests.Response:
        resp = requests.Response()
        resp.status_code = 200
        resp._content = json.dumps(data_bb_alt).encode("utf-8")
        return resp

    mock.return_value = response()
    return mock


@pytest.fixture(scope="module")
def data_sa_bb(data_sa) -> Dict[str, Union[str, np.int64, np.float64]]:
    data_sa_ = dict(data_sa)
    data_sa_["bb"] = [
        {
            "id": 13442,
            "title": "BBG01_TW_FLANGE",
            "description": "None",
            "x_position": np.float64(0.0),
            "y_position": np.float64(0.0),
            "z_position": np.float64(17000.0),
            "alpha": np.float64(1.0),
            "beta": np.float64(2.0),
            "gamma": np.float64(3.0),
            "vertical_position_reference_system": "LAT",
            "material": np.float64(1.0)
        }
    ]
    return data_sa_


@pytest.fixture(scope="module")
def outline_data() -> Tuple[List[float], List[float]]:
    x = [
        2247.55, 2247.55,
        2245.45, 2245.45,
        2244.85, 2244.85,
        -2244.85, -2244.85,
        -2245.45, -2245.45,
        -2247.55, -2247.55
    ]
    z = [
        22170.0, 24570.0,
        24570.0, 27400.0,
        27400.0, 30230.0,
        30230.0, 27400.0,
        27400.0, 24570.0,
        24570.0, 22170.0
    ]
    return x, z


@pytest.fixture(scope="module")
def sa_as_df(data_bb_real) -> pd.DataFrame:
    OD = ["4490", "4491", "4495"]
    v = [0.941, 0.9887, 0.9799]
    m = [7483.1, 7862, 7792]
    data = [
        [
            data_bb_real[i]["title"],
            data_bb_real[i]["x_position"],
            data_bb_real[i]["y_position"],
            data_bb_real[i]["z_position"],
            OD[i],
            data_bb_real[i]["wall_thickness"],
            data_bb_real[i]["height"],
            v[i],
            m[i],
            {"x": None, "y": None, "z": None},
            "",            
        ] for i in range(3)
    ]
    df = pd.DataFrame(
        data=data,
        columns=[
            "title", "x", "y", "z",
            "OD", "wall_thickness", "height", "volume", "mass", "moment_of_inertia", "description"
        ]
    )
    df.set_index("title", inplace=True)
    return df


@pytest.fixture(scope="module")
def absolute_bot() -> np.float64:
    return 22.17


@pytest.fixture(scope="module")
def absolute_top() -> np.float64:
    return 30.23
