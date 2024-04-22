from typing import Any, Dict

import pytest

from owimetadatabase_preprocessor.fatigue.io import FatigueAPI


@pytest.fixture(scope="function")
def fatigue_init(header):
    return {
        "api_root": "https://owimetadatabase.owilab.be/api/v1",
        "header": header,
        "auth": None,
        "uname": None,
        "password": None,
    }


def test_init(fatigue_init: Dict[str, Any], header: Dict[str, str]) -> None:
    api_fatigue = FatigueAPI(header=header)
    assert fatigue_init == api_fatigue
