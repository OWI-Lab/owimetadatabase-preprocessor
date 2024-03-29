from typing import Any, Dict, Union
from unittest import mock

import pandas as pd
import pandas.testing as pd_testing
import pytest
import requests

from owimetadatabase_preprocessor.soil.io import SoilAPI


def test_init(soil_init: Dict[str, Any], header: Dict[str, str]) -> None:
    api_soil = SoilAPI(header=header)
    assert soil_init == api_soil


def test_process_data(api_soil: SoilAPI, mock_requests_get_advanced: mock.Mock) -> None:
    url_data_type = "/test/"
    url_params = {"test": "test"}
    output_type = "list"
    df, df_add = api_soil.process_data(url_data_type, url_params, output_type)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_add, dict)
    assert isinstance(df_add["existance"], bool)
    assert isinstance(df_add["response"], requests.Response)


def test_get_proximity_entities_2d(
    api_soil: SoilAPI,
    mock_requests_get_proximity_entities_2d: mock.Mock,
) -> None:
    data = api_soil.get_proximity_entities_2d(
        api_url="test", latitude=50.1, longitude=2.22, radius=0.75
    )
    df = data["data"]
    assert isinstance(df, pd.DataFrame)
    assert isinstance(data["exists"], bool)
    assert data["exists"]
    assert len(df) == 2
    print(df["col_4"], df["col_6"])
    for col, dtype_ in zip(df.columns, [int, float, bool, object, dict, object]):
        assert df[col].dtype == dtype_


def test_get_proximity_entities_2d_wrong_data(
    api_soil: SoilAPI,
    mock_requests_get_proximity_entities_2d: mock.Mock,
) -> None:
    with pytest.raises(Exception):
        api_soil.get_proximity_entities_2d(
            api_url="test", latitude=50, longitude=2.22, radius=0.75
        )


@pytest.mark.parametrize(
    "r, close_entity_true",
    [(2.0, 0), (12.0, 1), (0.5, 2)],
    indirect=["close_entity_true"],
)
def test_search_any_entity(
    api_soil: SoilAPI,
    r: float,
    close_entity_true: pd.DataFrame,
    mock_requests_search_any_entity: mock.Mock,
) -> None:
    df = api_soil._search_any_entity(
        api_url="test",
        radius_init=r,
        url_params={
            "latitude": 50.0,
            "longitude": 2.0,
        },
        radius_max=100.0,
    )
    pd_testing.assert_frame_equal(df, close_entity_true)


def test_search_any_entity_exception(
    api_soil: SoilAPI, mock_requests_search_any_entity: mock.Mock
) -> None:
    with pytest.raises(Exception):
        api_soil._search_any_entity(
            api_url="test",
            radius_init=0.75,
            url_params={
                "latitude": 50.0,
                "longitude": 2.0,
            },
            radius_max=0.5,
        )


@pytest.mark.parametrize(
    "df_gathered_inp, dict_gathered_true",
    [("regular", "regular"), ("single", "single")],
    indirect=["df_gathered_inp", "dict_gathered_true"],
)
def test_gather_data_entity(
    api_soil: SoilAPI,
    df_gathered_inp: pd.DataFrame,
    dict_gathered_true: Dict[str, Union[str, float]],
) -> None:
    dict_gathered = api_soil._gather_data_entity(df_gathered_inp)
    for key in dict_gathered:
        if key != "data":
            assert dict_gathered[key] == dict_gathered_true[key]
        else:
            pd_testing.assert_frame_equal(dict_gathered[key], dict_gathered_true[key])


def test_get_closest_entity_2d(
    api_soil: SoilAPI,
    dict_gathered_final_true: Dict[str, Union[str, float]],
    mock_requests_search_any_entity: mock.Mock,
) -> None:
    dict_ = api_soil.get_closest_entity_2d(
        api_url="test", latitude=50.0, longitude=2.0, radius_init=10.0
    )
    dict_true = dict_gathered_final_true
    print(dict_["data"], dict_true["data"])
    for key in dict_true:
        if key != "data":
            assert dict_[key] == dict_true[key]
        else:
            pd_testing.assert_frame_equal(dict_[key], dict_true[key])


# def test_get_closest_entity_3d(
#     api_soil: SoilAPI,
#     dict_gathered_final_true: Dict[str, Union[str, float]],
#     mock_requests_search_any_entity: mock.Mock,
# ) -> None:
#     dict_ = api_soil.get_closest_entity_3d(
#         api_url="test", latitude=50.0, longitude=2.0, depth=10.0
#     )
#     dict_true = dict_gathered_final_true
#     print(dict_["data"], dict_true["data"])
#     for key in dict_true:
#         if key != "data":
#             assert dict_[key] == dict_true[key]
#         else:
#             pd_testing.assert_frame_equal(dict_[key], dict_true[key])
