from copy import deepcopy
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
from owimetadatabase_preprocessor.utils import dict_generator


@pytest.fixture(scope="function")
def sa_list_in(data):
    data_list = []
    for i in range(3):
        data_ = deepcopy(data["sa"][i])
        data_list.append(
            dict_generator(
                data_,
                keys_= ["slug", "model_definition"],
                method_="exclude"
            )
        )
    return data_list


@pytest.fixture(scope="function")
def sa_list_out(data, api_root, header, materials, bb_out_list):
    data_list = []
    for i in range(3):
        data_ = deepcopy(data["sa"][i])
        data_["position"] =  {
            "x": data_["x_position"],
            "y": data_["y_position"],
            "z": data_["z_position"],
            "alpha": np.float64(0),
            "beta": np.float64(0),
            "gamma": np.float64(0),
            "reference_system": data_["vertical_position_reference_system"]
        }
        data_["bb"] = None
        if data_["subassembly_type"] == "TP":
            data_["bb"] = bb_out_list[0]
        elif data_["subassembly_type"] == "MP":
            data_["bb"] = bb_out_list[1]
        else:
            data_["bb"] = bb_out_list[2]
        data_["materials"] = materials
        data_["api"] = {
            "api_root": api_root,
            "header": header,
            "uname": None,
            "password": None,
            "auth": None
        }
        data_["type"] = data_["subassembly_type"]
        data_list.append(
            dict_generator(
                data_,
                keys_= [
                    "x_position", "y_position", "z_position", 
                    "vertical_position_reference_system", "subassembly_type",
                    "slug", "model_definition"
                ],
                method_="exclude"
            )
        )
    return data_list


@pytest.fixture(scope="function")
def sa_df(sa_list_in):
    return pd.DataFrame(sa_list_in)


@pytest.fixture(scope="function")
def owt_init(api_test, materials_df, sa_list_out, data):
    tw_sa = (
        pd.DataFrame(data["sa_prop"][2]["df"])
        .drop(columns=["absolute_position, m"], axis=1)
        .set_index("title")
    )
    tp_sa = (
        pd.DataFrame(data["sa_prop"][0]["df"])
        .drop(columns=["absolute_position, m"], axis=1)
        .set_index("title")
    )
    mp_sa = (
        pd.DataFrame(data["sa_prop"][1]["df"])
        .drop(columns=["absolute_position, m"], axis=1)
        .set_index("title")
    )
    return {
        "api": api_test,
        "materials": materials_df,
        "sub_assemblies": {
            "TW": sa_list_out[2],
            "TP": sa_list_out[0],
            "MP": sa_list_out[1]
        },
        "tower_sub_assemblies": tw_sa,
        "tp_sub_assemblies": tp_sa,
        "mp_sub_assemblies": mp_sa,
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
        "mp_distributed_mass": None  
    }