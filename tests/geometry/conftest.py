import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from owimetadatabase_preprocessor.utils import fix_nan, fix_outline

from .fixtures.io import *  # noqa: F403, F401
from .fixtures.processing import *  # noqa: F403, F401
from .fixtures.structures import *  # noqa: F403, F401


@pytest.fixture(scope="module")
def data() -> Dict[str, List[Dict[str, Any]]]:
    file_dir = Path(__file__).parent
    data_path = file_dir / "data"
    data_type = {
        "mat": "materials",
        "sa": "subassemblies",
        "bb": "building_blocks",
        "bb_prop": "properties_bb",
        "sa_prop": "properties_sa",
    }
    data = {}
    for d in data_type.keys():
        with open(data_path / (data_type[d] + ".json")) as f:
            data_ = json.load(f)
            data[d] = fix_nan(data_)
            if d == "bb_prop" or d == "sa_prop":
                data[d] = fix_outline(data[d])
    return data
