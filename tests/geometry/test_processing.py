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
from .test_utils import assert_attributes


class  TestOWT:

    def test_init(self, owt, owt_init) -> None:
        assert_attributes(owt, owt_init)

    def test_set_subassemblies_and_members(self, owt_api, data_mat_df, sa_df, sab_list, mock_requests_get_buildingblocks_sa) -> None:
        owt = OWT(owt_api, data_mat_df, sa_df)
        sa_type = ["TW", "TP", "MP"]
        sab_dict = {sa_type[i]: sab_list[i] for i in range(len(sa_type))}
        for i in range(len(sa_type)):
            sab_dict[sa_type[i]].building_blocks
        assert owt.sub_assemblies == sab_dict
        pd_testing.assert_frame_equal(owt.tower_sub_assemblies, sab_dict["TW"].as_df(), rtol=1e-3, atol=1e-3)
        pd_testing.assert_frame_equal(owt.tp_sub_assemblies, sab_dict["TP"].as_df(), rtol=1e-3, atol=1e-3)
        pd_testing.assert_frame_equal(owt.mp_sub_assemblies, sab_dict["MP"].as_df(), rtol=1e-3, atol=1e-3)

    # @pytest.mark.parametrize("idx, expected", [("tw", )])
    # def test_set_df_structure(self, owt_api, data_mat_df, sa_df, idx):
    #     owt = OWT(owt_api, data_mat_df, sa_df)
    #     df = owt.set_df_structure(self, idx)