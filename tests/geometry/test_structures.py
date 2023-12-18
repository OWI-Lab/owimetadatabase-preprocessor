import json
from typing import Any, Callable, Dict, List, Union
from unittest import mock

import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import pytest
import requests

from owimetadatabase_preprocessor.geometry.io import GeometryAPI
from owimetadatabase_preprocessor.geometry.processing import OWT
from owimetadatabase_preprocessor.geometry.structures import Material, Position, BuildingBlock, SubAssembly


@pytest.fixture(scope="module")
def mass() -> np.float64:
    return 1500.0


@pytest.fixture(scope="module")
def outer_diameter() -> np.float64:
    return 1000.0


@pytest.fixture(scope="module")
def outer_diameter_alt() -> np.float64:
    return 1250.0


@pytest.fixture(scope="module")
def wall_t() -> np.float64:
    return 0.2


@pytest.fixture(scope="module")
def height() -> np.float64:
    return 10.0


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
def data_bb_mass(data_bb, mass) -> Dict[str, Union[str, np.int64, np.float64]]:
    dict_bb_new = dict(data_bb)
    dict_bb_new["mass"] = mass
    return dict_bb_new


@pytest.fixture(scope="module")
def data_bb_mass_distr(data_bb, mass) -> Dict[str, Union[str, np.int64, np.float64]]:
    dict_bb_new = dict(data_bb)
    dict_bb_new["mass_distribution"] = mass
    return dict_bb_new


@pytest.fixture(scope="module")
def data_bb_mass_distr_h(data_bb_mass_distr, height) -> Dict[str, Union[str, np.int64, np.float64]]:
    dict_bb_new = dict(data_bb_mass_distr)
    dict_bb_new["height"] = height
    dict_bb_new["volume_distribution"] = np.float64(100.0)
    return dict_bb_new


@pytest.fixture(scope="module")
def data_bb_bottom_out_d(data_bb, outer_diameter, wall_t) -> Dict[str, Union[str, np.int64, np.float64]]:
    dict_bb_new = dict(data_bb)
    dict_bb_new["bottom_outer_diameter"] = outer_diameter
    dict_bb_new["top_outer_diameter"] = outer_diameter
    dict_bb_new["wall_thickness"] = wall_t
    return dict_bb_new


@pytest.fixture(scope="module")
def data_bb_bottom_out_d_top_nan(
    data_bb,
    outer_diameter,
    wall_t
) -> Dict[str, Union[str, np.int64, np.float64]]:
    dict_bb_new = dict(data_bb)
    dict_bb_new["bottom_outer_diameter"] = outer_diameter
    dict_bb_new["top_outer_diameter"] = np.nan
    dict_bb_new["wall_thickness"] = wall_t
    return dict_bb_new


@pytest.fixture(scope="module")
def data_bb_bottom_out_d_bot_nan(
    data_bb,
    outer_diameter,
    wall_t
) -> Dict[str, Union[str, np.int64, np.float64]]:
    dict_bb_new = dict(data_bb)
    dict_bb_new["bottom_outer_diameter"] = np.nan
    dict_bb_new["top_outer_diameter"] = outer_diameter
    dict_bb_new["wall_thickness"] = wall_t
    return dict_bb_new


@pytest.fixture(scope="module")
def data_bb_bottom_out_d_alt(
    data_bb,
    outer_diameter,
    outer_diameter_alt,
    wall_t
) -> Dict[str, Union[str, np.int64, np.float64]]:
    dict_bb_new = dict(data_bb)
    dict_bb_new["bottom_outer_diameter"] = outer_diameter_alt
    dict_bb_new["top_outer_diameter"] = outer_diameter
    dict_bb_new["wall_thickness"] = wall_t
    return dict_bb_new


@pytest.fixture(scope="module")
def data_bb_h() -> Dict[str, Union[str, np.int64, np.float64]]:
    dict_bb_new = dict(data_bb)
    dict_bb_new["height"] = np.float64(10.0)
    return dict_bb_new


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


def _assert_attributes(class_, dict_, exclude=None) -> None:
    for key, value in dict_.items():
        if exclude is not None:
            if key in exclude:
                continue
        assert getattr(class_, key) == value
        assert type(getattr(class_, key)) == type(value)


class  TestMaterial:

    def test_init(self, data_mat) -> None:
        mat = Material(data_mat)
        _assert_attributes(mat, data_mat)

    def test_as_dict(self, data_mat, data_mat_dict) -> None:
        mat = Material(data_mat)
        mat_dict = mat.as_dict()
        assert mat_dict == data_mat_dict


class TestPosition:

    def test_init(self, data_pos) -> None:
        pos = Position(**data_pos)
        _assert_attributes(pos, data_pos)


class TestBuildingBlock:

    def test_init_no_sa(self, data_bb, data_bb_init_no_sa, data_pos) -> None:
        bb = BuildingBlock(json=data_bb)
        _assert_attributes(bb, data_bb_init_no_sa, exclude=["position"])
        assert bb.subassembly is None
        assert isinstance(bb.position, Position)
        _assert_attributes(bb.position, data_pos)

    def test_init_with_sa(self, data_bb, data_bb_init_with_sa, data_pos, SA) -> None:
        bb = BuildingBlock(json=data_bb, subassembly=SA)
        _assert_attributes(bb, data_bb_init_with_sa, exclude=["position"])
        assert isinstance(bb.position, Position)
        _assert_attributes(bb.position, data_pos)
    

    @pytest.mark.parametrize(
        "data_bb_var, expected_type", 
        [
            ("data_bb_mass", "lumped_mass"),
            ("data_bb_mass_distr", "distributed_mass"),
            ("data_bb_bottom_out_d", "tubular_section"),
            ("data_bb", ValueError)
        ]
    )
    def test_type(self, request, data_bb_var, expected_type) -> None:
        data_bb_ = request.getfixturevalue(data_bb_var)
        if expected_type == ValueError:
            with pytest.raises(ValueError):
                bb = BuildingBlock(json=data_bb_)
                bb.type
        else:
            bb = BuildingBlock(json=data_bb_)
            assert bb.type == expected_type


    @pytest.mark.parametrize(
        "data_bb_var, expected_wt", 
        [
            ("data_bb_mass", None),
            ("data_bb_mass_distr", None),
            ("data_bb_bottom_out_d", "wall_thickness"),
        ]
    )
    def test_wall_thickness(self, request, data_bb_var, expected_wt) -> None:
        data_bb_ = request.getfixturevalue(data_bb_var)
        bb = BuildingBlock(json=data_bb_)
        expected_wt_ = data_bb_[expected_wt] if expected_wt is not None else None
        assert bb.wall_thickness == expected_wt_


    @pytest.mark.parametrize(
        "data_bb_var, expected_bod", 
        [
            ("data_bb_mass", None),
            ("data_bb_mass_distr", None),
            ("data_bb_bottom_out_d", "bottom_outer_diameter"),
        ]
    )
    def test_bottom_outer_diameter(self, request, data_bb_var, expected_bod) -> None:
        data_bb_ = request.getfixturevalue(data_bb_var)
        bb = BuildingBlock(json=data_bb_)
        expected_bod_ = data_bb_[expected_bod] if expected_bod is not None else None
        assert bb.bottom_outer_diameter == expected_bod_


    @pytest.mark.parametrize(
        "data_bb_var, expected_tod", 
        [
            ("data_bb_mass", None),
            ("data_bb_mass_distr", None),
            ("data_bb_bottom_out_d", "top_outer_diameter"),
        ]
    )
    def test_top_outer_diameter(self, request, data_bb_var, expected_tod) -> None:
        data_bb_ = request.getfixturevalue(data_bb_var)
        bb = BuildingBlock(json=data_bb_)
        expected_tod_ = data_bb_[expected_tod] if expected_tod is not None else None
        assert bb.top_outer_diameter == expected_tod_


    @pytest.mark.parametrize(
        "data_bb_var, expected_od", 
        [
            ("data_bb_mass", None),
            ("data_bb_bottom_out_d_top_nan", None),
            ("data_bb_bottom_out_d_bot_nan", None),
            ("data_bb_bottom_out_d", lambda _, od: str(round(od))),
            (
                "data_bb_bottom_out_d_alt",
                lambda oda, od: str(round(oda)) + "/" + str(round(od))
            ),
        ],
    )
    def test_diameter_str(self, request, data_bb_var, expected_od, outer_diameter, outer_diameter_alt) -> None:
        data_bb_ = request.getfixturevalue(data_bb_var)
        bb = BuildingBlock(json=data_bb_)
        expected_od_ = expected_od(outer_diameter_alt, outer_diameter) if expected_od is not None else ""
        assert bb.diameter_str == expected_od_

    @pytest.mark.parametrize(
        "data_bb_var, expected_h", 
        [
            ("data_bb", None),
            ("data_bb_h", "height"),
        ]
    )
    def height(self, request, data_bb_var, expected_h) -> None:
        data_bb_ = request.getfixturevalue(data_bb_var)
        bb = BuildingBlock(json=data_bb_)
        expected_h_ = data_bb_[expected_h] if expected_h is not None else None
        assert bb.height == expected_h_

    # @pytest.mark.parametrize(
    #     "data_bb_var, expected_v", 
    #     [
    #         ("data_bb_mass", None),
    #         ("data_bb_mass_distr_h", None),
    #         ("data_bb_mass_distr", ValueError),
    #         ("data_bb_bottom_out_d", None),
    #     ],
    # )
    # def volume(self, request, data_bb_var, outer_diameter, wall_t) -> None:
    #     data_bb_ = request.getfixturevalue(data_bb_var)
    #     bb = BuildingBlock(json=data_bb_)
    #     expected_vol = np.pi * (outer_diameter**2 - (outer_diameter - 2*wall_t)**2) / 4
    #     assert bb.volume == expected_vol
