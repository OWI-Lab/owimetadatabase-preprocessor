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


@pytest.fixture(scope="function")
def sab(api_root, header, data_mat_df, data_sa) -> SubAssembly:
    api_test = GeometryAPI(api_root, header)
    sa = SubAssembly(data_mat_df, data_sa, api_test)
    return sa


@pytest.fixture(scope="module")
def owt_api(api_root, header):
    return GeometryAPI(api_root, header)


@pytest.fixture(scope="function")
def owt_init(data_mat_df, owt_api) -> None:
    owt = {
        "api": owt_api,
        "materials": [m.to_dict() for _, m in data_mat_df.iterrows()],
        "tower_sub_assemblies": None,
        "tp_sub_assemblies": None,
        "mp_sub_assemblies": None,
        "tower_base": None,
        "pile_head": None,
        "pile_toe": None,
        "rna": None,
        "tower_geometry": None,
        "transition_piece": None,
        "monopile": None,
        "substructure": None,
        "tp_skirt": None,
        "tower_lumped_mass": None,
        "tp_lumped_mass": None,
        "mp_lumped_mass": None,
        "tp_distributed_mass": None,
        "mp_distributed_mass": None,
    }
    return owt


@pytest.fixture(scope="function")
def mock_owt_set_subassemblies(mocker: mock.Mock, sab) -> mock.Mock:
    mock = mocker.patch.object(OWT, "_set_subassemblies")
        
    def mocked_set_subassemblies(self, *args, **kwargs):
        self.sub_assemblies = {"TW": object(), "TP": object(), "MP": object()}

    mock.side_effect = mocked_set_subassemblies(mock, sab)
    return mock
    

@pytest.fixture(scope="function")
def mock_owt_set_members(mocker: mock.Mock) -> mock.Mock:
    mock = mocker.patch.object(OWT, "_set_members")
        
    def mocked_set_members(self, *args, **kwargs):
        self.tower_sub_assemblies = pd.DataFrame({"title": ["BBG01_TW"], "mass": [10000]})
        self.tp_sub_assemblies = pd.DataFrame({"title": ["BBG01_TP"], "mass": [5000]})
        self.mp_sub_assemblies = pd.DataFrame({"title": ["BBG01_MP"], "mass": [20000]})

    mock.side_effect = mocked_set_members(mock)
    return mock


@pytest.fixture(scope="function")
def owt(data_mat_df, sab, owt_api, mock_owt_set_subassemblies, mock_owt_set_members):
    mat = [m.to_dict() for _, m in data_mat_df.iterrows()]
    owt = OWT(owt_api, mat, sab)
    return owt


@pytest.fixture(scope="function")
def sa_list():
    cols = [
        "id", "title", "description", "slug", 
        "x_position", "y_position", "z_position", "vertical_position_reference_system", 
        "subassembly_type", "source", "asset", "model_definition"
    ]
    data = [
        {
            "id": 651,
            "title": "BBG01_TW",
            "description": None,
            "slug": "bbg01_tw", 
            "x_position": 0.0,
            "y_position": 0.0,
            "z_position": 17000.0,
            "vertical_position_reference_system": "LAT", 
            "subassembly_type": "TW",
            "source": "vestas_tower_dwg.pdf",
            "asset": 341,
            "model_definition": 5
        },
        {
            "id": 855,
            "title": "BBG01_TP",
            "description": None,
            "slug": "bbg01_tp",
            "x_position": 0.0,
            "y_position": 0.0,
            "z_position": -2540.0,
            "vertical_position_reference_system": "LAT", 
            "subassembly_type": "TP",
            "source": "NBW-513-004-Design Report - WTG Time Domain Fatigue Analysis",
            "asset": 341,
            "model_definition": 5
        },
        {
            "id": 64,
            "title": "BBG01_MP",
            "description": None,
            "slug": "bbg01_mp",
            "x_position": 0.0,
            "y_position": 0.0,
            "z_position": -63400.0,
            "vertical_position_reference_system": "LAT",
            "subassembly_type": "MP",
            "source": "NBW-513-004-Design Report - WTG Time Domain Fatigue Analysis",
            "asset": 341,
            "model_definition": 5
        }
    ]
    return data


@pytest.fixture(scope="function")
def sa_df(sa_list):
    return pd.DataFrame(sa_list)


@pytest.fixture(scope="function")
def sab_list(owt_api, data_mat_df, sa_list, mock_requests_get_buildingblocks_sa) -> SubAssembly:
    sa = [SubAssembly(data_mat_df, sa_dict, owt_api) for sa_dict in sa_list]
    return sa
