from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from owimetadatabase_preprocessor.geometry.processing import OWT, OWTs
from owimetadatabase_preprocessor.geometry.structures import Position
from owimetadatabase_preprocessor.utils import dict_generator


@pytest.fixture(scope="function")
def sa_list_in(data):
    data_list = []
    for i in range(3):
        data_ = deepcopy(data["sa"][i])
        data_list.append(
            dict_generator(data_, keys_=["slug", "model_definition"], method_="exclude")
        )
    return data_list


@pytest.fixture(scope="function")
def sa_df(sa_list_in):
    return pd.DataFrame(sa_list_in)


@pytest.fixture(scope="function")
def loc():
    return pd.DataFrame(data=[30.0], columns=["elevation"])


@pytest.fixture(scope="function")
def owt(api_test, materials_df, sa_df, loc, mock_requests_sa_get_bb):
    return OWT(api_test, materials_df, sa_df, loc)


@pytest.fixture(scope="function")
def bb_list_in(bb_in_list, bb_in_list_prop, materials_dicts_init):
    bb_list = deepcopy(bb_in_list)
    for i in range(len(bb_list)):
        bb_list[i]["json"] = bb_in_list_prop[i].copy()
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
    return bb_list[:5], bb_list[5:8], bb_list[8:12]


@pytest.fixture(scope="function")
def sa_list_out(data, api_root, header, materials_dicts_init, bb_list_in):
    data_list = []
    for i in range(3):
        data_ = deepcopy(data["sa"][i])
        data_["position"] = {
            "x": data_["x_position"],
            "y": data_["y_position"],
            "z": data_["z_position"],
            "alpha": np.float64(0),
            "beta": np.float64(0),
            "gamma": np.float64(0),
            "reference_system": data_["vertical_position_reference_system"],
        }
        data_["bb"] = None
        if data_["subassembly_type"] == "TP":
            data_["bb"] = bb_list_in[0]
        elif data_["subassembly_type"] == "MP":
            data_["bb"] = bb_list_in[1]
        else:
            data_["bb"] = bb_list_in[2]
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
            }
        }
        data_["type"] = data_["subassembly_type"]
        data_list.append(
            dict_generator(
                data_,
                keys_=[
                    "x_position",
                    "y_position",
                    "z_position",
                    "vertical_position_reference_system",
                    "subassembly_type",
                    "slug",
                    "model_definition",
                ],
                method_="exclude",
            )
        )
    return data_list


@pytest.fixture(scope="function")
def owt_init(api_test, materials_df, sa_list_out, data, loc):
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
        "_init_proc": False,
        "_init_spec_part": False,
        "_init_spec_full": False,
        "api": api_test,
        "materials": materials_df,
        "sub_assemblies": {
            "TW": sa_list_out[2],
            "TP": sa_list_out[0],
            "MP": sa_list_out[1],
        },
        "tw_sub_assemblies": tw_sa,
        "tp_sub_assemblies": tp_sa,
        "mp_sub_assemblies": mp_sa,
        "tower_base": 16.0,
        "pile_head": 7.5,
        "pile_toe": None,
        "rna": None,
        "tower": None,
        "transition_piece": None,
        "monopile": None,
        "grout": None,
        "substructure": None,
        "tp_skirt": None,
        "full_structure": None,
        "tw_lumped_mass": None,
        "tp_lumped_mass": None,
        "mp_lumped_mass": None,
        "tp_distributed_mass": None,
        "mp_distributed_mass": None,
        "water_depth": loc["elevation"].iloc[0],
    }


@pytest.fixture(scope="function")
def df_set_tube_true(request, data):
    if request.param is not None:
        idx = request.param
    if idx == 0:
        return pd.DataFrame(data["geo"]["set_tube"][0]).set_index("title")
    elif idx == 1:
        return pd.DataFrame(data["geo"]["set_tube"][1]).set_index("title")
    elif idx == 2:
        return pd.DataFrame(data["geo"]["set_tube"][2]).set_index("title")


@pytest.fixture(scope="function")
def df_proc_tube_true(request, data):
    if request.param is not None:
        idx = request.param
    if idx == 0:
        return pd.DataFrame(data["geo"]["process_tube"][0]).set_index("title")
    elif idx == 1:
        return pd.DataFrame(data["geo"]["process_tube"][1]).set_index("title")
    elif idx == 2:
        return pd.DataFrame(data["geo"]["process_tube"][2]).set_index("title")


@pytest.fixture(scope="function")
def df_rna_true(data):
    return pd.DataFrame(data["geo"]["process_rna"]).set_index("title")


@pytest.fixture(scope="function")
def df_set_lump_mass_true(request, data):
    if request.param is not None:
        idx = request.param
    if idx == 0:
        return pd.DataFrame(data["geo"]["set_appurtenances"][0]).set_index("title")
    elif idx == 1:
        return pd.DataFrame(data["geo"]["set_appurtenances"][1]).set_index("title")


@pytest.fixture(scope="function")
def df_proc_lump_mass_true(request, data):
    if request.param is not None:
        idx = request.param
    if idx == 0:
        return pd.DataFrame(data["geo"]["process_appurtenances"][0]).set_index("title")
    elif idx == 1:
        return pd.DataFrame(data["geo"]["process_appurtenances"][1]).set_index("title")


@pytest.fixture(scope="function")
def df_set_distr_true(request, data):
    if request.param is not None:
        idx = request.param
    if idx == 0:
        return pd.DataFrame(data["geo"]["set_distr"][0]).set_index("title")
    elif idx == 1:
        return pd.DataFrame(data["geo"]["set_distr"][1]).set_index("title")


@pytest.fixture(scope="function")
def df_proc_distr_true(request, data):
    if request.param is not None:
        idx = request.param
    if idx == 0:
        return pd.DataFrame(data["geo"]["process_distr"][0]).set_index("title")
    elif idx == 1:
        return pd.DataFrame(data["geo"]["process_distr"][1]).set_index("title")


@pytest.fixture(scope="function")
def df_proc_struct_true(request, data):
    if request.param is not None:
        idx = request.param
    rna, tw, tp, mp, tw_lump, tp_lump, mp_lump, tp_distr, mp_distr, grout = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    if idx == 0:
        rna = pd.DataFrame(data["geo"]["process_rna"]).set_index("title")
        tw = pd.DataFrame(data["geo"]["process_tube"][2]).set_index("title")
        tp = pd.DataFrame(data["geo"]["process_tube"][0]).set_index("title")
        mp = pd.DataFrame(data["geo"]["process_tube"][1]).set_index("title")
        tw_lump = pd.DataFrame(data["geo"]["process_appurtenances"][1]).set_index(
            "title"
        )
        tp_lump = pd.DataFrame(data["geo"]["process_appurtenances"][0]).set_index(
            "title"
        )
        tp_distr = pd.DataFrame(data["geo"]["process_distr"][0]).set_index("title")
        grout = pd.DataFrame(data["geo"]["process_distr"][1]).set_index("title")
        mp_lump = pd.DataFrame(
            columns=["title", "X [m]", "Y [m]", "Z [mLAT]", "Mass [t]", "Description"],
            dtype=np.float64,
        )
        mp_lump["title"] = mp_lump["title"].astype(str)
        mp_lump["Description"] = mp_lump["Description"].astype(str)
        mp_lump.set_index("title", inplace=True)
        mp_distr = pd.DataFrame(
            columns=[
                "title",
                "X [m]",
                "Y [m]",
                "Z [mLAT]",
                "Height [m]",
                "Mass [t]",
                "Volume [m3]",
                "Description",
            ],
            dtype=np.float64,
        )
        mp_distr["title"] = mp_distr["title"].astype(str)
        mp_distr["Description"] = mp_distr["Description"].astype(str)
        mp_distr.set_index("title", inplace=True)
    elif idx == 1:
        rna = pd.DataFrame(data["geo"]["process_rna"]).set_index("title")
        tw = pd.DataFrame(data["geo"]["process_tube"][2]).set_index("title")
        tw_lump = pd.DataFrame(data["geo"]["process_appurtenances"][1]).set_index(
            "title"
        )
    elif idx == 2:
        tp = pd.DataFrame(data["geo"]["process_tube"][0]).set_index("title")
        tp_lump = pd.DataFrame(data["geo"]["process_appurtenances"][0]).set_index(
            "title"
        )
        tp_distr = pd.DataFrame(data["geo"]["process_distr"][0]).set_index("title")
        grout = pd.DataFrame(data["geo"]["process_distr"][1]).set_index("title")
    elif idx == 3:
        mp = pd.DataFrame(data["geo"]["process_tube"][1]).set_index("title")
        mp_lump = pd.DataFrame(
            columns=["title", "X [m]", "Y [m]", "Z [mLAT]", "Mass [t]", "Description"],
            dtype=np.float64,
        )
        mp_lump["title"] = mp_lump["title"].astype(str)
        mp_lump["Description"] = mp_lump["Description"].astype(str)
        mp_lump.set_index("title", inplace=True)
        mp_distr = pd.DataFrame(
            columns=[
                "title",
                "X [m]",
                "Y [m]",
                "Z [mLAT]",
                "Height [m]",
                "Mass [t]",
                "Volume [m3]",
                "Description",
            ],
            dtype=np.float64,
        )
        mp_distr["title"] = mp_distr["title"].astype(str)
        mp_distr["Description"] = mp_distr["Description"].astype(str)
        mp_distr.set_index("title", inplace=True)
    return rna, tw, tp, mp, tw_lump, tp_lump, mp_lump, tp_distr, mp_distr, grout


@pytest.fixture(scope="function")
def can(data):
    series = pd.Series(data["sub"]["can_adjust"][0])
    return series


@pytest.fixture(scope="function")
def can_mod(request, data):
    if request.param is not None:
        idx = request.param
    if idx == 0:
        df = pd.DataFrame(data["sub"]["can_mod_bot"][0]).set_index("title")
    elif idx == 1:
        df = pd.DataFrame(data["sub"]["can_mod_top"][0]).set_index("title")
    return df


@pytest.fixture(scope="function")
def assembled_tp_mp(data):
    subs = pd.concat(
        [
            pd.DataFrame(data["sub"]["can_mod_bot"][0]).set_index("title"),
            pd.DataFrame(data["geo"]["process_tube"][1]).set_index("title"),
        ]
    )
    skirt = pd.DataFrame(data["sub"]["can_mod_top"][0]).set_index("title")
    return subs, skirt


@pytest.fixture(scope="function")
def assembled_full(data):
    struct = pd.concat(
        [
            pd.DataFrame(data["geo"]["process_tube"][2]).set_index("title"),
            pd.DataFrame(data["sub"]["can_mod_bot"][0]).set_index("title"),
            pd.DataFrame(data["geo"]["process_tube"][1]).set_index("title"),
        ]
    )
    return struct


@pytest.fixture(scope="function")
def turbines_list():
    return ["AAA01", "AAB02"]


@pytest.fixture(scope="function")
def owt_list(owt):
    return [owt, owt]


@pytest.fixture(scope="function")
def owts_init(owt, api_test, materials_df):
    dict_init = {}
    dict_init["owts"] = {"AAA01": owt, "AAB02": owt}
    dict_init["api"] = api_test
    dict_init["materials"] = materials_df
    dict_init["sub_assemblies"] = {
        "AAA01": owt.sub_assemblies,
        "AAB02": owt.sub_assemblies,
    }
    dict_init["tower_base"] = {"AAA01": owt.tower_base, "AAB02": owt.tower_base}
    dict_init["pile_head"] = {"AAA01": owt.pile_head, "AAB02": owt.pile_head}
    dict_init["water_depth"] = {"AAA01": owt.water_depth, "AAB02": owt.water_depth}
    dict_init["tw_sub_assemblies"] = pd.concat(
        [owt.tw_sub_assemblies, owt.tw_sub_assemblies]
    )
    dict_init["tp_sub_assemblies"] = pd.concat(
        [owt.tp_sub_assemblies, owt.tp_sub_assemblies]
    )
    dict_init["mp_sub_assemblies"] = pd.concat(
        [owt.mp_sub_assemblies, owt.mp_sub_assemblies]
    )
    dict_init["_init"] = False
    dict_init["pile_toe"] = []
    dict_init["rna"] = []
    dict_init["tower"] = []
    dict_init["transition_piece"] = []
    dict_init["monopile"] = []
    dict_init["tw_lumped_mass"] = []
    dict_init["tp_lumped_mass"] = []
    dict_init["mp_lumped_mass"] = []
    dict_init["tp_distributed_mass"] = []
    dict_init["mp_distributed_mass"] = []
    dict_init["grout"] = []
    dict_init["full_structure"] = []
    dict_init["tp_skirt"] = []
    dict_init["substructure"] = []
    dict_init["all_tubular_structures"] = []
    dict_init["all_distributed_mass"] = []
    dict_init["all_lumped_mass"] = []
    dict_init["all_turbines"] = []
    return dict_init


@pytest.fixture(scope="function")
def all_turb_true(data):
    return pd.DataFrame(data["turb"], index=[0])


@pytest.fixture(scope="function")
def owts(turbines_list, owt_list):
    return OWTs(turbines_list, owt_list)


@pytest.fixture(scope="function")
def owts_true(data, assembled_tp_mp, assembled_full, owts_init):
    dict_ = owts_init
    dict_["pile_toe"] = {"AAA01": -62.5, "AAB02": -62.5}
    rna = pd.DataFrame(data["geo"]["process_rna"]).set_index("title")
    tower = pd.DataFrame(data["geo"]["process_tube"][2]).set_index("title")
    tp = pd.DataFrame(data["geo"]["process_tube"][0]).set_index("title")
    mp = pd.DataFrame(data["geo"]["process_tube"][1]).set_index("title")
    tw_lump_mass = pd.DataFrame(data["geo"]["process_appurtenances"][1]).set_index(
        "title"
    )
    tp_lump_mass = pd.DataFrame(data["geo"]["process_appurtenances"][0]).set_index(
        "title"
    )
    mp_lump_mass = pd.DataFrame(
        columns=[
            "title",
            "X [m]",
            "Y [m]",
            "Z [mLAT]",
            "Mass [t]",
            "Description",
            "Subassembly",
        ],
        dtype=np.float64,
    ).set_index("title")
    tp_distr_mass = pd.DataFrame(data["geo"]["process_distr"][0]).set_index("title")
    mp_distr_mass = pd.DataFrame(
        columns=[
            "title",
            "X [m]",
            "Y [m]",
            "Z [mLAT]",
            "Height [m]",
            "Mass [t]",
            "Volume [m3]",
            "Description",
            "Subassembly",
        ],
        dtype=np.float64,
    ).set_index("title")
    grout = pd.DataFrame(data["geo"]["process_distr"][1]).set_index("title")
    rna["Subassembly"] = "TW"
    tower["Subassembly"] = "TW"
    tp["Subassembly"] = "TP"
    mp["Subassembly"] = "MP"
    tw_lump_mass["Subassembly"] = "TW"
    tp_lump_mass["Subassembly"] = "TP"
    tp_distr_mass["Subassembly"] = "TP"
    grout["Subassembly"] = "TP"
    dict_["rna"] = pd.concat([rna, rna])
    dict_["tower"] = pd.concat([tower, tower])
    dict_["transition_piece"] = pd.concat([tp, tp])
    dict_["monopile"] = pd.concat([mp, mp])
    dict_["tw_lumped_mass"] = pd.concat([tw_lump_mass, tw_lump_mass])
    dict_["tp_lumped_mass"] = pd.concat([tp_lump_mass, tp_lump_mass])
    dict_["mp_lumped_mass"] = pd.concat([mp_lump_mass, mp_lump_mass])
    dict_["tp_distributed_mass"] = pd.concat([tp_distr_mass, tp_distr_mass])
    dict_["mp_distributed_mass"] = pd.concat([mp_distr_mass, mp_distr_mass])
    dict_["grout"] = pd.concat([grout, grout])
    dict_["tp_skirt"] = pd.concat([assembled_tp_mp[1], assembled_tp_mp[1]])
    dict_["substructure"] = pd.concat([assembled_tp_mp[0], assembled_tp_mp[0]])
    dict_["full_structure"] = pd.concat([assembled_full, assembled_full])
    dict_["tp_skirt"]["Subassembly"] = "TP"
    for sa in ["TW", "TP", "MP"]:
        dict_["substructure"].loc[
            dict_["substructure"].index.str.contains(sa.lower()), "Subassembly"
        ] = sa
        dict_["full_structure"].loc[
            dict_["full_structure"].index.str.contains(sa.lower()), "Subassembly"
        ] = sa
    dict_["all_tubular_structures"] = pd.concat([tower, tp, mp, tower, tp, mp])
    dict_["all_distributed_mass"] = pd.concat(
        [tp_distr_mass, grout, mp_distr_mass, tp_distr_mass, grout, mp_distr_mass]
    )
    dict_["all_lumped_mass"] = pd.concat(
        [
            rna[
                ["X [m]", "Y [m]", "Z [mLAT]", "Mass [t]", "Description", "Subassembly"]
            ],
            tw_lump_mass,
            tp_lump_mass,
            mp_lump_mass,
            rna[
                ["X [m]", "Y [m]", "Z [mLAT]", "Mass [t]", "Description", "Subassembly"]
            ],
            tw_lump_mass,
            tp_lump_mass,
            mp_lump_mass,
        ]
    )
    dict_["all_turbines"] = pd.concat(
        [pd.DataFrame(data["turb"], index=[0]), pd.DataFrame(data["turb"], index=[1])]
    )
    dict_["all_turbines"].loc[1, "Turbine name"] = "AAB02"
    return dict_
