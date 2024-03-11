from typing import Any, Dict
from unittest import mock

import pandas as pd
import pytest
import requests

from owimetadatabase_preprocessor.soil.io import SoilAPI


def test_init(soil_init: Dict[str, Any], header: Dict[str, str]) -> None:
    api_soil = SoilAPI(header=header)
    assert soil_init == api_soil


def test_process_data(
    api_root: str, header: Dict[str, str], mock_requests_get_advanced: mock.Mock
) -> None:
    header = header
    url_data_type = "/test/"
    url_params = {"test": "test"}
    output_type = "list"
    api_test = SoilAPI(api_root, header=header)
    df, df_add = api_test.process_data(url_data_type, url_params, output_type)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_add, dict)
    assert isinstance(df_add["existance"], bool)
    assert isinstance(df_add["response"], requests.Response)


def test_get_proximity_entities_2d(
    api_root: str,
    header: Dict[str, str],
    mock_requests_get_proximity_entities_2d: mock.Mock,
) -> None:
    api_test = SoilAPI(api_root, header=header)
    data = api_test.get_proximity_entities_2d(
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
    api_root: str,
    header: Dict[str, str],
    mock_requests_get_proximity_entities_2d: mock.Mock,
) -> None:
    api_test = SoilAPI(api_root, header=header)
    with pytest.raises(Exception):
        api_test.get_proximity_entities_2d(
            api_url="test", latitude=50, longitude=2.22, radius=0.75
        )
