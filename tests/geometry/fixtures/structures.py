import json
from copy import deepcopy
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import requests

from owimetadatabase_preprocessor.geometry.io import GeometryAPI
from owimetadatabase_preprocessor.geometry.structures import Material, Position
from owimetadatabase_preprocessor.utils import dict_generator


@pytest.fixture(scope="function")
def materials_dicts_init(data):
    materials_ = []
    for mat in data["mat"]:
        materials_.append(dict_generator(mat, keys_=["slug"], method_="exclude"))
    return materials_


@pytest.fixture(scope="function")
def materials_dicts_asdict(data):
    materials_ = []
    for mat in data["mat"]:
        materials_.append(
            dict_generator(mat, keys_=["id", "density", "slug"], method_="exclude")
        )
    return materials_


@pytest.fixture(scope="function")
def position_1(data):
    data_ = dict_generator(
        data["bb"][0],
        keys_=[
            "alpha",
            "beta",
            "gamma",
            "x_position",
            "y_position",
            "z_position",
            "vertical_position_reference_system",
        ],
        method_="include",
    )
    return {
        "x": data_["x_position"],
        "y": data_["y_position"],
        "z": data_["z_position"],
        "alpha": data_["alpha"],
        "beta": data_["beta"],
        "gamma": data_["gamma"],
        "reference_system": data_["vertical_position_reference_system"],
    }


@pytest.fixture(scope="function")
def position_5(data):
    data_ = dict_generator(
        data["bb"][4],
        keys_=[
            "alpha",
            "beta",
            "gamma",
            "x_position",
            "y_position",
            "z_position",
            "vertical_position_reference_system",
        ],
        method_="include",
    )
    return {
        "x": data_["x_position"],
        "y": data_["y_position"],
        "z": data_["z_position"],
        "alpha": data_["alpha"],
        "beta": data_["beta"],
        "gamma": data_["gamma"],
        "reference_system": data_["vertical_position_reference_system"],
    }


@pytest.fixture(scope="function")
def bb_in_no_mat(data):
    return dict_generator(
        data["bb"][0],
        keys_=[
            "slug",
            "area_distribution",
            "c_d",
            "c_m",
            "sub_assembly",
            "projectsite_name",
            "asset_name",
            "subassembly_name",
            "material_name",
            "youngs_modulus",
            "density",
            "poissons_ratio",
            "mass",
            "height",
            "mass_distribution",
            "volume_distribution",
            "bottom_outer_diameter",
            "top_outer_diameter",
            "wall_thickness",
            "moment_of_inertia_x",
            "moment_of_inertia_y",
            "moment_of_inertia_z",
        ],
        method_="exclude",
    )


@pytest.fixture(scope="function")
def bb_out_no_mat(bb_in_no_mat, position_1):
    data_ = deepcopy(bb_in_no_mat)
    data_["position"] = position_1
    data_["material"] = None
    data_["json"] = bb_in_no_mat
    return dict_generator(
        data_,
        keys_=[
            "x_position",
            "y_position",
            "z_position",
            "alpha",
            "beta",
            "gamma",
            "vertical_position_reference_system",
        ],
        method_="exclude",
    )


@pytest.fixture(scope="function")
def bb_in(data):
    return dict_generator(
        data["bb"][4],
        keys_=[
            "slug",
            "area_distribution",
            "c_d",
            "c_m",
            "sub_assembly",
            "projectsite_name",
            "asset_name",
            "subassembly_name",
            "material_name",
            "youngs_modulus",
            "density",
            "poissons_ratio",
            "mass",
            "height",
            "mass_distribution",
            "volume_distribution",
            "bottom_outer_diameter",
            "top_outer_diameter",
            "wall_thickness",
            "moment_of_inertia_x",
            "moment_of_inertia_y",
            "moment_of_inertia_z",
        ],
        method_="exclude",
    )


@pytest.fixture(scope="function")
def bb_out(bb_in, position_5, materials_dicts_init):
    data_ = deepcopy(bb_in)
    data_["position"] = position_5
    data_["json"] = bb_in
    data_["material"] = materials_dicts_init[0]
    data_["description"] = ""
    return dict_generator(
        data_,
        keys_=[
            "x_position",
            "y_position",
            "z_position",
            "alpha",
            "beta",
            "gamma",
            "vertical_position_reference_system",
        ],
        method_="exclude",
    )


@pytest.fixture(scope="function")
def sa_mock(materials_dicts_init) -> mock.Mock:
    def SA_mock_init(self, *args, **kwargs):
        self.materials = args[0]

    mat = [Material(material) for material in materials_dicts_init]
    mocked_SA = mock.Mock()
    SA_mock_init(mocked_SA, mat)
    return mocked_SA


@pytest.fixture(scope="module")
def api_test(api_root, header):
    return GeometryAPI(api_root=api_root, header=header)


@pytest.fixture(scope="function")
def materials_df(materials_dicts_init):
    return pd.DataFrame(materials_dicts_init)


@pytest.fixture(scope="function")
def sa_in(data):
    return dict_generator(
        data["sa"][0], keys_=["slug", "model_definition"], method_="exclude"
    )


@pytest.fixture(scope="function")
def position_sa_1(data):
    data_ = dict_generator(
        data["sa"][0],
        keys_=[
            "x_position",
            "y_position",
            "z_position",
            "vertical_position_reference_system",
        ],
        method_="include",
    )
    return {
        "x": data_["x_position"],
        "y": data_["y_position"],
        "z": data_["z_position"],
        "alpha": np.float64(0),
        "beta": np.float64(0),
        "gamma": np.float64(0),
        "reference_system": data_["vertical_position_reference_system"],
    }


@pytest.fixture(scope="function")
def sa_out(sa_in, position_sa_1, materials_dicts_init, api_root, header):
    data_ = deepcopy(sa_in)
    data_["position"] = position_sa_1
    data_["bb"] = None
    data_["materials"] = materials_dicts_init
    data_["api"] = {
        "api_root": api_root + "/geometry/userroutes/",
        "header": header,
        "uname": None,
        "password": None,
        "auth": None,
        "loc_api": {
            "api_root": api_root + "/locations/",
            "header": header,
            "uname": None,
            "password": None,
            "auth": None,
        },
    }
    data_["type"] = data_["subassembly_type"]
    return dict_generator(
        data_,
        keys_=[
            "x_position",
            "y_position",
            "z_position",
            "vertical_position_reference_system",
            "subassembly_type",
        ],
        method_="exclude",
    )


@pytest.fixture(scope="function")
def bb_in_list(data):
    bb_list = []
    for i in range(len(data["bb"])):
        bb_dict = dict_generator(
            data["bb"][i],
            keys_=[
                "slug",
                "area_distribution",
                "c_d",
                "c_m",
                "sub_assembly",
                "projectsite_name",
                "asset_name",
                "subassembly_name",
                "material_name",
                "youngs_modulus",
                "density",
                "poissons_ratio",
                "mass",
                "height",
                "mass_distribution",
                "volume_distribution",
                "bottom_outer_diameter",
                "top_outer_diameter",
                "wall_thickness",
                "moment_of_inertia_x",
                "moment_of_inertia_y",
                "moment_of_inertia_z",
            ],
            method_="exclude",
        )
        bb_list.append(bb_dict)
    return bb_list


@pytest.fixture(scope="function")
def bb_out_list(bb_in_list, materials_dicts_init):
    bb_list = deepcopy(bb_in_list)
    for i in range(len(bb_list)):
        bb_list[i]["json"] = bb_list[i].copy()
        bb_list[i]["position"] = Position(
            bb_list[i]["x_position"],
            bb_list[i]["y_position"],
            bb_list[i]["z_position"],
            bb_list[i]["alpha"],
            bb_list[i]["beta"],
            bb_list[i]["gamma"],
            bb_list[i]["vertical_position_reference_system"],
        )
        if bb_list[i]["material"] is not None and not np.isnan(bb_list[i]["material"]):
            bb_list[i]["material"] = materials_dicts_init[
                np.int64(bb_list[i]["material"]) - 1
            ]
        else:
            bb_list[i]["material"] = None
        if bb_in_list[i]["description"] is None:
            bb_list[i]["description"] = ""
        bb_list[i] = dict_generator(
            bb_list[i],
            keys_=[
                "x_position",
                "y_position",
                "z_position",
                "alpha",
                "beta",
                "gamma",
                "vertical_position_reference_system",
            ],
            method_="exclude",
        )
    return [bb_list[:5], bb_list[5:8], bb_list[8:12]]


@pytest.fixture
def mock_requests_sa_get_bb_bb(mocker: mock.Mock, bb_in_list) -> mock.Mock:
    def custom_side_effect(*args, **kwargs) -> requests.Response:
        resp = requests.Response()
        resp.status_code = 200
        if kwargs.get("params") == {"sub_assembly__id": "1"}:
            data = [bb_in_list[i] for i in range(5)]
        elif kwargs.get("params") == {"sub_assembly__id": "2"}:
            data = [bb_in_list[i] for i in range(5, 8)]
        elif kwargs.get("params") == {"sub_assembly__id": "3"}:
            data = [bb_in_list[i] for i in range(8, 12)]
        resp._content = json.dumps(data).encode("utf-8")
        return resp

    mock = mocker.patch("requests.get", side_effect=custom_side_effect)
    return mock


@pytest.fixture(scope="function")
def bb_in_list_prop(data):
    bb_list = []
    for i in range(len(data["bb"])):
        bb_dict = dict_generator(
            data["bb"][i],
            keys_=[
                "slug",
                "area_distribution",
                "c_d",
                "c_m",
                "material_name",
                "youngs_modulus",
                "density",
                "poissons_ratio",
            ],
            method_="exclude",
        )
        bb_list.append(bb_dict)
    return bb_list


@pytest.fixture
def mock_requests_sa_get_bb(mocker: mock.Mock, bb_in_list_prop, data) -> mock.Mock:
    def custom_side_effect(*args, **kwargs) -> requests.Response:
        resp = requests.Response()
        resp.status_code = 200
        if kwargs.get("params") == {"sub_assembly__id": "1"}:
            data = [bb_in_list_prop[i] for i in range(5)]
        elif kwargs.get("params") == {"sub_assembly__id": "2"}:
            data = [bb_in_list_prop[i] for i in range(5, 8)]
        elif kwargs.get("params") == {"sub_assembly__id": "3"}:
            data = [bb_in_list_prop[i] for i in range(8, 12)]
        else:
            data = []
        resp._content = json.dumps(data).encode("utf-8")
        return resp

    mock = mocker.patch("requests.get", side_effect=custom_side_effect)
    return mock
