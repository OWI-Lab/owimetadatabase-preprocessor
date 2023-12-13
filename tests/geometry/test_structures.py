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
def api_root() -> str:
    return "https://test.api/test"


@pytest.fixture(scope="module")
def header() -> Dict[str, str]:
    return {"Authorization": "Token 12345"}


@pytest.fixture(scope="module")
def data_mat() -> Dict[str, Union[str, np.int64, np.float64]]:    
    return {
        "title": "steel",
        "id": np.int64(1),
        "description": "Structural steel",
        "young_modulus": np.float64(210.0),
        "density": np.float64(7850.0),
        "poisson_ratio": np.float64(0.3),
    }


@pytest.fixture(scope="module")
def data_mat_dict() -> Dict[str, Union[str, np.float64]]:
    return {
        "title": "steel",
        "description": "Structural steel",
        "young_modulus": np.float64(210.0),
        "poisson_ratio": np.float64(0.3),
    }


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

    def test_init_no_sa(self, data_bb, data_bb_init_no_sa) -> None:
        bb = BuildingBlock(json=data_bb)
        _assert_attributes(bb, data_bb_init_no_sa, exclude=["position"])
        assert bb.subassembly is None
        assert isinstance(bb.position, Position)
        assert bb.position.x == data_bb["x_position"]
        assert bb.position.y == data_bb["y_position"]
        assert bb.position.z == data_bb["z_position"]
        assert bb.position.alpha == data_bb["alpha"]
        assert bb.position.beta == data_bb["beta"]
        assert bb.position.gamma == data_bb["gamma"]
        assert bb.position.reference_system == data_bb["vertical_position_reference_system"]
