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
            ("m_distr_vh", "distributed_mass"),
            ("bot_od", "tubular_section"),
            ("", ValueError)
        ],
        indirect=["data_bb_flex"]
    )
    def test_type(self, data_bb_flex, expected_type) -> None:
        if expected_type == ValueError:
            with pytest.raises(ValueError):
                bb = BuildingBlock(json=data_bb_flex)
                bb.type
        else:
            bb = BuildingBlock(json=data_bb_flex)
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
        bb = BuildingBlock(json=data_bb_flex)
        expected_wt_ = data_bb_flex[expected_wt] if expected_wt is not None else None
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
        bb = BuildingBlock(json=data_bb_flex)
        expected_bod_ = data_bb_flex[expected_bod] if expected_bod is not None else None
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
        bb = BuildingBlock(json=data_bb_flex)
        expected_tod_ = data_bb_flex[expected_tod] if expected_tod is not None else None
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
        bb = BuildingBlock(json=data_bb_flex)
        expected_od_ = expected_od(data_bb_flex["bottom_outer_diameter"], data_bb_flex["top_outer_diameter"]) if expected_od is not None else ""
        assert bb.diameter_str == expected_od_


    @pytest.mark.parametrize(
        "data_bb_flex, expected_h", 
        [
            ("", None),
            ("h", "height"),
        ],
        indirect=["data_bb_flex"]
    )
    def test_height(self, data_bb_flex, expected_h) -> None:
        bb = BuildingBlock(json=data_bb_flex)
        expected_h_ = data_bb_flex[expected_h] if expected_h is not None else None
        assert bb.height == expected_h_


    @pytest.mark.parametrize(
        "data_bb_flex, expected_v", 
        [
            ("m", None),
            ("m_distr_vh", ["volume_distribution", "height"]),
            ("m_distr", ValueError),
        ],
        indirect=["data_bb_flex"]
    )
    def test_volume_no_tube(self, data_bb_flex, expected_v) -> None:
        if expected_v == ValueError:
            with pytest.raises(ValueError):
                bb = BuildingBlock(json=data_bb_flex)
                bb.volume
        else:
            bb = BuildingBlock(json=data_bb_flex)
            expected_vol = (
                np.float64(
                    round(data_bb_flex[expected_v[0]] * data_bb_flex[expected_v[1]]/1000)
                ) if expected_v is not None else None
            )
            assert bb.volume == expected_vol


    @pytest.mark.parametrize(
        "data_bb_flex, expected_v", 
        [
            ("bot_od_h", ["bottom_outer_diameter", "top_outer_diameter", "wall_thickness", "height"]),
        ],
        indirect=["data_bb_flex"]
    )
    def test_volume_tube(self, data_bb_flex, expected_v) -> None:
        bot_od = data_bb_flex[expected_v[0]]
        top_od = data_bb_flex[expected_v[1]]
        wt = data_bb_flex[expected_v[2]]
        h = data_bb_flex[expected_v[3]]
        rbo = bot_od/2
        rto = top_od/2
        rbi = rbo - wt
        rti = rto - wt
        bb = BuildingBlock(json=data_bb_flex)
        expected_vol = np.float64(((np.pi*h/3*(rbo**2+rbo*rto+rto**2)) - (np.pi*h/3*(rbi**2+rbi*rti+rti**2)))/1e9)
        assert bb.volume == expected_vol


    @pytest.mark.parametrize(
        "data_bb_flex, expected_m", 
        [
            ("m", "mass"),
            ("m_distr", ValueError),
            ("m_distr_vh", "mass_calc"),
            ("bot_od_h", ValueError),
        ],
        indirect=["data_bb_flex"]
    )
    def test_mass_1(self, data_bb_flex, expected_m):
        if expected_m == ValueError:
            with pytest.raises(ValueError):
                bb = BuildingBlock(json=data_bb_flex)
                bb.mass
        else:
            bb = BuildingBlock(json=data_bb_flex)
            assert bb.mass == data_bb_flex[expected_m]


    @pytest.mark.parametrize(
        "data_bb_flex, expected_m", 
        [
            ("bot_od_h", ValueError),
        ],
        indirect=["data_bb_flex"]
    )
    def test_mass_2(self, data_bb_flex, expected_m):
        if expected_m == ValueError:
            with pytest.raises(ValueError):
                bb = BuildingBlock(json=data_bb_flex)
                bb.mass


    @pytest.mark.parametrize(
        "data_bb_flex, expected_m", 
        [
            ("bot_od_h", "mass_calc"),
        ],
        indirect=["data_bb_flex"]
    )
    def test_mass_3(self, data_bb_flex, expected_m, SA):
        bb = BuildingBlock(json=data_bb_flex, subassembly=SA)
        assert bb.mass == data_bb_flex[expected_m]


    @pytest.mark.parametrize(
        "data_bb_flex, expected_mi", 
        [
            ("bot_od_h", None),
            ("m", ["moment_of_inertia_x", "moment_of_inertia_y", "moment_of_inertia_z"])
        ],
        indirect=["data_bb_flex"]
    )
    def test_moment_inertia(self, data_bb_flex, expected_mi):
        bb = BuildingBlock(json=data_bb_flex)
        expected_mi_ = (
            {
                k: data_bb_flex[expected_mi[i]] for i, k in zip([0, 1, 2], ["x", "y", "z"])
            } if expected_mi is not None 
            else {
                k: None for k in ["x", "y", "z"]
            }
        )
        assert bb.moment_of_inertia == expected_mi_


    @pytest.mark.parametrize(
        "data_bb_flex, expected_out", 
        [
            ("bot_od_h", "outline"),
            ("m", None)
        ],
        indirect=["data_bb_flex"]
    )
    def test_outline(self, data_bb_flex, expected_out):
        bb = BuildingBlock(json=data_bb_flex)
        expected_outline = data_bb_flex[expected_out] if expected_out is not None else None
        assert bb.outline == expected_outline


    @pytest.mark.parametrize(
        "data_bb_flex, expected_mark", 
        [
            ("bot_od_h", None),
            ("m", "marker")
        ],
        indirect=["data_bb_flex"]
    )
    def test_marker(self, data_bb_flex, expected_mark):
        bb = BuildingBlock(json=data_bb_flex)
        expected_marker = data_bb_flex[expected_mark] if expected_mark is not None else None
        assert bb.marker == expected_marker


    @pytest.mark.parametrize(
        "data_bb_flex, expected_l", 
        [
            ("m_distr_vh", "line"),
            ("m", None)
        ],
        indirect=["data_bb_flex"]
    )
    def test_line(self, data_bb_flex, expected_l):
        bb = BuildingBlock(json=data_bb_flex)
        expected_line = data_bb_flex[expected_l] if expected_l is not None else None
        assert bb.line == expected_line


    @pytest.mark.parametrize(
        "data_bb_flex", 
        [
            ("m_distr_vh"),
            ("m"),
            ("bot_od_h")
        ],
        indirect=["data_bb_flex"]
    )
    def test_as_dict(self, data_bb_flex, SA) -> None:
        bb = BuildingBlock(json=data_bb_flex, subassembly=SA)
        assert bb.as_dict() == data_bb_flex["dict"]


    @pytest.mark.parametrize(
        "data_bb_flex", 
        [
            ("m_distr_vh"),
            ("m"),
            ("bot_od_h")
        ],
        indirect=["data_bb_flex"]
    )
    def test_str(self, data_bb_flex):
        bb = BuildingBlock(json=data_bb_flex)
        assert str(bb) == data_bb_flex["str"]


class TestSubAssembly:

    def test_init(self, sa, data_sa_init, data_mat_df) -> None:
        _assert_attributes(sa, data_sa_init, exclude=["api", "materials", "position"])	
        assert isinstance(sa.position, Position)
        _assert_attributes(sa.position, data_sa_init["position"])
        assert isinstance(sa.materials, List)
        assert len(sa.materials) == 1
        assert isinstance(sa.materials[0], Material)
        assert isinstance(sa.api, GeometryAPI)
        _assert_attributes(sa.materials[0], data_mat_df.iloc[0].to_dict())

    @pytest.mark.parametrize(
        "data_sa_flex", 
        [
            ("tw"),
            ("tp"),
            ("mp")
        ],
        indirect=["data_sa_flex"]
    )
    def test_color(self, api_root, header, data_sa_flex, data_mat_df) -> None:
        api_test = GeometryAPI(api_root, header)
        sa = SubAssembly(data_mat_df, data_sa_flex, api_test)
        assert sa.color == data_sa_flex["color"]

    def test_bb(self, sa, mock_requests_get_buildingblocks_sa) -> None:
        bb = sa.building_blocks
        assert isinstance(bb, List)
        assert len(bb) == 3
        assert isinstance(bb[0], BuildingBlock)
        assert isinstance(bb[1], BuildingBlock)
        assert isinstance(bb[2], BuildingBlock)
        assert sa.bb == bb
        #_assert_attributes(sa.bb[0], data_sa["bb"][0])

    def test_bb_exists(self, sa, mock_requests_get_buildingblocks_sa) -> None:
        sa.bb = [1, "test"]
        bb = sa.building_blocks
        assert sa.bb == bb

    def test_bb_api(self, data_sa, data_mat_df, mock_requests_get_buildingblocks_sa) -> None:
        sa = SubAssembly(data_mat_df, data_sa, None)
        with pytest.raises(ValueError):
            sa.building_blocks

    def test_bb_not_exists(self, sa, mock_requests_get_buildingblocks_sa_alt) -> None:
        with pytest.raises(ValueError):
            sa.building_blocks

    def test_height(self, sa, mock_requests_get_buildingblocks_sa, data_bb_real) -> None:
        sa.building_blocks
        h = 0
        for i in range(len(sa.bb)):
            h += data_bb_real[i]["height"]
        assert sa.height == h

    def test_mass(self, sa, mock_requests_get_buildingblocks_sa, data_bb_real) -> None:
        sa.building_blocks
        m = 0
        for i in range(len(sa.bb)):
           m += data_bb_real[i]["mass"]
        assert sa.mass == m

    def test_properties(self, sa, mock_requests_get_buildingblocks_sa, data_bb_real) -> None:
        sa.building_blocks
        m, h = 0, 0
        for i in range(len(sa.bb)):
           m += data_bb_real[i]["mass"]
           h += data_bb_real[i]["height"]
        assert sa.properties["mass"] == m
        assert sa.properties["height"] == h

    def test_outline(self, sa, mock_requests_get_buildingblocks_sa, outline_data) -> None:
         assert sa.outline == outline_data

    def test_as_df(self, sa, mock_requests_get_buildingblocks_sa, sa_as_df) -> None:
        pd_testing.assert_frame_equal(sa.as_df(), sa_as_df, rtol=1e-3, atol=1e-3)

    def test_absolute_bottom(self, sa, mock_requests_get_buildingblocks_sa, absolute_bot) -> None:
        assert sa.absolute_bottom == absolute_bot

    def test_absolute_top(self, sa, mock_requests_get_buildingblocks_sa, absolute_top) -> None:
        assert sa.absolute_top == absolute_top

    def test_repr_html(self) -> None:
        pass

    def test_str(self, sa) -> None:
        assert str(sa) == "BBG01_TW Subassembly"
