import json
from copy import deepcopy
from pathlib import Path

from typing import  Any, Callable, Dict, List, Tuple, Union
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import requests

from owimetadatabase_preprocessor.geometry.io import GeometryAPI
from owimetadatabase_preprocessor.geometry.processing import OWT
from owimetadatabase_preprocessor.geometry.structures import Material, Position, BuildingBlock, SubAssembly


@pytest.fixture(scope="module")
def data():
    file_dir = Path(__file__).parent.parent
    data_path = file_dir / "data"
    data_type = {
        "mat": "materials",
        "sa": "subassemblies",
        "bb": "building_blocks",
        "bb_prop": "properties_bb",
        "sa_prop": "properties_sa"
    }
    data = {}
    for d in data_type.keys():
        with open(data_path / (data_type[d] + ".json")) as f:
            data[d] = json.load(f)
    return data

@pytest.fixture(scope="function")
def material_main(data):
    data_ = {k: data["mat"][0][k] for k in data["mat"][0].keys() if k not in ["slug"]}
    return data_

@pytest.fixture(scope="function")
def material_main_dict(data):
    data_ = {k: data["mat"][0][k] for k in data["mat"][0].keys() if k not in ["id", "density", "slug"]}
    return data_
