import pandas as pd
import pandas.testing as pd_testing
import pytest

from owimetadatabase_preprocessor.geometry.structures import (
    BaseStructure,
    BuildingBlock,
    Material,
    Position,
    SubAssembly,
)
from owimetadatabase_preprocessor.utils import deepcompare


class TestBaseStructure:
    def test_eq_2_obj(self) -> None:
        obj_1 = BaseStructure()
        obj_2 = BaseStructure()
        assert obj_1 == obj_2

    def test_eq_obj_dict(self) -> None:
        obj_1 = BaseStructure()
        obj_2 = obj_1.__dict__
        assert obj_1 == obj_2

    def test_eq_obj_any(self) -> None:
        obj_1 = BaseStructure()
        obj_2 = "test"
        assert not obj_1 == obj_2


class TestMaterial:
    def test_init(self, materials) -> None:
        for i in range(len(materials)):
            mat = Material(materials[i])
            assert mat == materials[i]

    def test_as_dict(self, materials, materials_dict) -> None:
        for i in range(len(materials)):
            mat = Material(materials[i])
            mat_dict = mat.as_dict()
            assert mat_dict == materials_dict[i]


class TestPosition:
    def test_init(self, position_1) -> None:
        pos = Position(**position_1)
        assert pos == position_1


class TestBuildingBlock:
    def test_init_no_sa(self, bb_in_no_mat, bb_out_no_mat) -> None:
        bb = BuildingBlock(json=bb_in_no_mat)
        assertion, message = deepcompare(bb, bb_out_no_mat)
        assert assertion, message

    def test_init_sa(self, bb_in_no_mat, bb_out_no_mat, sa_mock) -> None:
        bb = BuildingBlock(json=bb_in_no_mat, subassembly=sa_mock)
        assertion, message = deepcompare(bb, bb_out_no_mat)
        assert assertion, message

    def test_init_sa_mat(self, bb_in, bb_out, sa_mock) -> None:
        bb = BuildingBlock(json=bb_in, subassembly=sa_mock)
        assertion, message = deepcompare(bb, bb_out)
        assert assertion, message

    @pytest.mark.parametrize(
        "property",
        [
            "type",
            "wall_thickness",
            "bottom_outer_diameter",
            "top_outer_diameter",
            "diameter_str",
            "height",
            "volume",
            "mass",
            "moment_of_inertia",
            "outline",
            "marker",
            "line",
        ],
    )
    def test_building_block_properties(self, data, sa_mock, property):
        for i in range(len(data["bb_prop"])):
            bb = BuildingBlock(json=data["bb"][i], subassembly=sa_mock)
            assertion, message = deepcompare(
                getattr(bb, property), data["bb_prop"][i][property]
            )
            assert assertion, message

    def test_as_dict(self, data, sa_mock):
        for i in range(len(data["bb_prop"])):
            bb = BuildingBlock(json=data["bb"][i], subassembly=sa_mock)
            assertion, message = deepcompare(bb.as_dict(), data["bb_prop"][i]["dict_"])
            assert assertion, message

    def test_str(self, data, sa_mock):
        for i in range(len(data["bb_prop"])):
            bb = BuildingBlock(json=data["bb"][i], subassembly=sa_mock)
            assert str(bb) == data["bb_prop"][i]["str_"]


class TestSubAssembly:
    def test_init(self, api_test, materials_df, sa_in, sa_out):
        sa = SubAssembly(materials_df, sa_in, api_test)
        assertion, message = deepcompare(sa, sa_out)
        assert assertion, message

    def test_subassemblies_bb(
        self, data, bb_out_list, api_test, materials_df, mock_requests_sa_get_bb_bb
    ):
        for i in range(len(data["sa_prop"])):
            sa = SubAssembly(materials_df, data["sa"][i], api_test)
            sa.building_blocks
            assertion, message = deepcompare(sa.bb, bb_out_list[i])
            assert assertion, message

    @pytest.mark.parametrize(
        "property",
        [
            "color",
            "height",
            "mass",
            "properties",
            "outline",
            "absolute_bottom",
            "absolute_top",
        ],
    )
    def test_subassemblies_properties(
        self, data, api_test, materials_df, property, mock_requests_sa_get_bb
    ):
        for i in range(len(data["sa_prop"])):
            sa = SubAssembly(materials_df, data["sa"][i], api_test)
            if property == "properties":
                assertion, message = deepcompare(
                    getattr(sa, property),
                    {
                        "mass": data["sa_prop"][i]["mass"],
                        "height": data["sa_prop"][i]["height"],
                    },
                )
                assert assertion, message
            else:
                assertion, message = deepcompare(
                    getattr(sa, property), data["sa_prop"][i][property]
                )
                assert assertion, message

    def test_subassemblies_as_df(
        self, data, api_test, materials_df, mock_requests_sa_get_bb
    ):
        for i in range(len(data["sa_prop"])):
            sa = SubAssembly(materials_df, data["sa"][i], api_test)
            df = sa.as_df()
            df_abs = sa.as_df(include_absolute_postion=True)
            df_expected = pd.DataFrame(data["sa_prop"][i]["df"])
            df_expected.drop("absolute_position, m", axis=1, inplace=True)
            df_expected.set_index("title", inplace=True)
            df_abs_expected = pd.DataFrame(data["sa_prop"][i]["df"])
            df_abs_expected.set_index("title", inplace=True)
            pd_testing.assert_frame_equal(df, df_expected)
            pd_testing.assert_frame_equal(df_abs, df_abs_expected)

    def test_subassemblies_str(
        self, data, api_test, materials_df, mock_requests_sa_get_bb
    ):
        for i in range(len(data["sa_prop"])):
            sa = SubAssembly(materials_df, data["sa"][i], api_test)
            assert str(sa) == data["sa_prop"][i]["str_"]
