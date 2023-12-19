from typing import Dict, Union
from unittest import mock

import numpy as np
import pytest


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
    data_mat_dict["density"] = np.float64(7850.0)
    return data_mat_dict


@pytest.fixture(scope="module")
def data_pos() -> Dict[str, Union[str, np.float64]]:
    return {
        "x": np.float64(1.0),
        "y": np.float64(2.0),
        "z": np.float64(3.0),
        "alpha": np.float64(4.0),
        "beta": np.float64(5.0),
        "gamma": np.float64(6.0),
        "reference_system": "LAT",
    }


@pytest.fixture(scope="module")
def data_bb() -> Dict[str, Union[str, np.int64, np.float64]]:
    return {
        "id": np.int64(1),
        "title": "BBG01_TW_FLANGE",
        "description": "Something 1",
        "x_position": np.float64(1.0),
        "y_position": np.float64(2.0),
        "z_position": np.float64(3.0),
        "alpha": np.float64(4.0),
        "beta": np.float64(5.0),
        "gamma": np.float64(6.0),
        "vertical_position_reference_system": "LAT",
        "material": np.float64(1.0),
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
    if param == "h":
        data_bb_ = dict(data_bb)
        data_bb_["height"] = height
        return data_bb_
    elif param == "m":
        data_bb_ = dict(data_bb)
        data_bb_["mass"] = mass
        return data_bb_
    elif param == "m_distr":
        data_bb_ = dict(data_bb)
        data_bb_["mass_distribution"] = mass
        return data_bb_
    elif param == "bot_od":
        data_bb_ = dict(data_bb)
        data_bb_["bottom_outer_diameter"] = outer_diameter
        data_bb_["top_outer_diameter"] = outer_diameter
        data_bb_["wall_thickness"] = wall_t
        return data_bb_
    elif param == "bot_od_alt":
        data_bb_ = dict(data_bb)
        data_bb_["bottom_outer_diameter"] = outer_diameter_alt
        data_bb_["top_outer_diameter"] = outer_diameter
        data_bb_["wall_thickness"] = wall_t
        return data_bb_
    elif param == "bot_od_top_nan":
        data_bb_ = dict(data_bb)
        data_bb_["bottom_outer_diameter"] = outer_diameter
        data_bb_["top_outer_diameter"] = np.nan
        data_bb_["wall_thickness"] = wall_t
        return data_bb_
    elif param == "bot_od_bot_nan":
        data_bb_ = dict(data_bb)
        data_bb_["bottom_outer_diameter"] = np.nan
        data_bb_["top_outer_diameter"] = outer_diameter
        data_bb_["wall_thickness"] = wall_t
        return data_bb_
    elif param == "m_distr_h":
        data_bb_ = dict(data_bb)
        data_bb_["volume_distribution"] = mass
        data_bb_["height"] = height
        return data_bb_
    else:
        return data_bb