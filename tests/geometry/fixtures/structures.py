import json
from copy import deepcopy
from pathlib import Path

from typing import  Any, Callable, Dict, List, Tuple, Union
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import requests

from owimetadatabase_preprocessor.geometry.io import GeometryAPI
from owimetadatabase_preprocessor.geometry.processing import OWT
from owimetadatabase_preprocessor.geometry.structures import Material, Position, BuildingBlock, SubAssembly

from owimetadatabase_preprocessor.utils import dict_generator, fix_nan


@pytest.fixture(scope="module")
def data():
    file_dir = Path(__file__).parent.parent
    data_path = file_dir / "data"
    data_type = {
        "mat": "materials",
        "sa": "subassemblies",
        "bb": "building_blocks",
        "bb_prop": "properties_bb",
        "sa_prop": "properties_sa"
    }
    data = {}
    for d in data_type.keys():
        with open(data_path / (data_type[d] + ".json")) as f:
            data_ = json.load(f)
            data[d] = fix_nan(data_)
    return data


@pytest.fixture(scope="function")
def material_main(data):
    return dict_generator(data["mat"][0], keys_=["slug"], method_="exclude")


@pytest.fixture(scope="function")
def material_main_dict(data):
    return dict_generator(data["mat"][0], keys_=["id", "density", "slug"], method_="exclude")


@pytest.fixture(scope="function")
def position_1(data):
    data_ = dict_generator(
        data["bb"][0],
        keys_=["alpha", "beta", "gamma", "x_position", "y_position", "z_position", "vertical_position_reference_system"],
        method_="include"
    )
    return {
        "x": data_["x_position"],
        "y": data_["y_position"],
        "z": data_["z_position"],
        "alpha": data_["alpha"],
        "beta": data_["beta"],
        "gamma": data_["gamma"],
        "reference_system": data_["vertical_position_reference_system"]
    }


@pytest.fixture(scope="function")
def position_5(data):
    data_ = dict_generator(
        data["bb"][4],
        keys_=["alpha", "beta", "gamma", "x_position", "y_position", "z_position", "vertical_position_reference_system"],
        method_="include"
    )
    return {
        "x": data_["x_position"],
        "y": data_["y_position"],
        "z": data_["z_position"],
        "alpha": data_["alpha"],
        "beta": data_["beta"],
        "gamma": data_["gamma"],
        "reference_system": data_["vertical_position_reference_system"]
    }


@pytest.fixture(scope="function")
def bb_in_no_mat(data):
    return dict_generator(
        data["bb"][0],
        keys_= [
            "slug", "area_distribution", "c_d", "c_m", "sub_assembly",
            "projectsite_name", "asset_name", "subassembly_name", 
            "material_name", "youngs_modulus", "density", "poissons_ratio",
            "mass", "height", "mass_distribution", "volume_distribution",
            "bottom_outer_diameter", "top_outer_diameter", "wall_thickness",
            "moment_of_inertia_x", "moment_of_inertia_y", "moment_of_inertia_z"
        ],
        method_="exclude"
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
            "x_position", "y_position", "z_position", 
            "alpha", "beta", "gamma", "vertical_position_reference_system"
        ],
        method_="exclude"
    )


@pytest.fixture(scope="function")
def bb_in(data):
    return dict_generator(
        data["bb"][4],
        keys_= [
            "slug", "area_distribution", "c_d", "c_m", "sub_assembly",
            "projectsite_name", "asset_name", "subassembly_name", 
            "material_name", "youngs_modulus", "density", "poissons_ratio",
            "mass", "height", "mass_distribution", "volume_distribution",
            "bottom_outer_diameter", "top_outer_diameter", "wall_thickness",
            "moment_of_inertia_x", "moment_of_inertia_y", "moment_of_inertia_z"
        ],
        method_="exclude"
    )


@pytest.fixture(scope="function")
def bb_out(bb_in, position_5, material_main):
    data_ = deepcopy(bb_in)
    data_["position"] = position_5
    data_["json"] = bb_in
    data_["material"] = material_main
    data_["description"] = ""
    return dict_generator(
        data_,
        keys_=[
            "x_position", "y_position", "z_position", 
            "alpha", "beta", "gamma", "vertical_position_reference_system"
        ],
        method_="exclude"
    )


@pytest.fixture(scope="function")
def sa_mock(material_main) -> mock.Mock:
    
    def SA_mock_init(self, *args, **kwargs):
        self.materials = [args[0]]
            
    mat = Material(material_main)
    mocked_SA = mock.Mock()
    SA_mock_init(mocked_SA, mat)
    return mocked_SA