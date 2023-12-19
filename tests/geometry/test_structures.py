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
        "data_bb_flex, expected_type", 
        [
            ("m", "lumped_mass"),
            ("m_distr", "distributed_mass"),
            ("bot_od", "tubular_section"),
            ("", ValueError)
        ],
        indirect=["data_bb_flex"]
    )
    def test_type(self, data_bb_flex, expected_type) -> None:
        data_bb_ = data_bb_flex
        if expected_type == ValueError:
            with pytest.raises(ValueError):
                bb = BuildingBlock(json=data_bb_)
                bb.type
        else:
            bb = BuildingBlock(json=data_bb_)
            assert bb.type == expected_type


    @pytest.mark.parametrize(
        "data_bb_flex, expected_wt", 
        [
            ("m", None),
            ("m_distr", None),
            ("bot_od", "wall_thickness"),
        ],
        indirect=["data_bb_flex"]
    )
    def test_wall_thickness(self, data_bb_flex, expected_wt) -> None:
        data_bb_ = data_bb_flex
        bb = BuildingBlock(json=data_bb_)
        expected_wt_ = data_bb_[expected_wt] if expected_wt is not None else None
        assert bb.wall_thickness == expected_wt_


    @pytest.mark.parametrize(
        "data_bb_flex, expected_bod", 
        [
            ("m", None),
            ("m_distr", None),
            ("bot_od", "bottom_outer_diameter"),
        ],
        indirect=["data_bb_flex"]
    )
    def test_bottom_outer_diameter(self, data_bb_flex, expected_bod) -> None:
        data_bb_ = data_bb_flex
        bb = BuildingBlock(json=data_bb_)
        expected_bod_ = data_bb_[expected_bod] if expected_bod is not None else None
        assert bb.bottom_outer_diameter == expected_bod_


    @pytest.mark.parametrize(
        "data_bb_flex, expected_tod", 
        [
            ("m", None),
            ("m_distr", None),
            ("bot_od", "top_outer_diameter"),
        ],
        indirect=["data_bb_flex"]
    )
    def test_top_outer_diameter(self, data_bb_flex, expected_tod) -> None:
        data_bb_ = data_bb_flex
        bb = BuildingBlock(json=data_bb_)
        expected_tod_ = data_bb_[expected_tod] if expected_tod is not None else None
        assert bb.top_outer_diameter == expected_tod_


    @pytest.mark.parametrize(
        "data_bb_flex, expected_od", 
        [
            ("m", None),
            ("bot_od_top_nan", None),
            ("bot_od_bot_nan", None),
            ("bot_od", lambda _, od: str(round(od))),
            (
                "bot_od_alt",
                lambda oda, od: str(round(oda)) + "/" + str(round(od))
            ),
        ],
        indirect=["data_bb_flex"]
    )
    def test_diameter_str(self, data_bb_flex, expected_od) -> None:
        data_bb_ = data_bb_flex
        bb = BuildingBlock(json=data_bb_)
        expected_od_ = expected_od(data_bb_["bottom_outer_diameter"], data_bb_["top_outer_diameter"]) if expected_od is not None else ""
        assert bb.diameter_str == expected_od_


    @pytest.mark.parametrize(
        "data_bb_flex, expected_h", 
        [
            ("", None),
            ("h", "height"),
        ],
        indirect=["data_bb_flex"]
    )
    def height(self, data_bb_flex, expected_h) -> None:
        data_bb_ = data_bb_flex
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
